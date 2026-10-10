import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import torch
import torch.nn.functional as F
from PIL import Image
from skimage import data

from notebooks.inference_utils import load_model_from_checkpoint, multiwm_dbscan
from watermark_anything.data.transforms import default_transform


torch.manual_seed(20261007)
random.seed(20261007)
torch.set_num_threads(4)

device = torch.device("cpu")
start = time.time()
wam = load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth")
wam = wam.to(device).eval()
print("load_seconds", time.time() - start, flush=True)

image = Image.fromarray(data.astronaut()).convert("RGB")
img = default_transform(image).unsqueeze(0).to(device)
messages = torch.randint(0, 2, (2, 32), generator=torch.Generator().manual_seed(7)).float()
masks = torch.zeros((2, 1, img.shape[-2], img.shape[-1]), dtype=img.dtype)
masks[0, :, 45:205, 45:205] = 1
masks[1, :, 307:467, 307:467] = 1

with torch.inference_mode():
    watermarked = img.clone()
    embed_start = time.time()
    for index in range(2):
        output = wam.embed(img, messages[index].unsqueeze(0))
        watermarked = output["imgs_w"] * masks[index] + watermarked * (1 - masks[index])
    print("embed_seconds", time.time() - embed_start, flush=True)
    detect_start = time.time()
    preds = wam.detect(watermarked)["preds"]
    mask_prob = torch.sigmoid(preds[:, 0])
    centroids, labels = multiwm_dbscan(
        preds[:, 1:], mask_prob, epsilon=1, min_samples=500
    )
    print("detect_cluster_seconds", time.time() - detect_start, flush=True)

print("clusters", len(centroids), flush=True)
for key, centroid in centroids.items():
    scores = (centroid.float().unsqueeze(0) == messages).float().mean(1)
    print("cluster", int(key), "best_bit_accuracy", float(scores.max()), flush=True)
print("mask_pixels", int((mask_prob > 0.5).sum()), flush=True)
