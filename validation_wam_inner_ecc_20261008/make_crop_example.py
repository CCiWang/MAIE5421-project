"""Regenerate one fixed crop for an accurately labeled diagnostic figure."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

import run_sprint as s
from run_crop import fixed_crops


def main():
    torch.set_num_threads(4)
    manifest = json.loads((s.OUT / "image_manifest_reused.json").read_text())
    image = s.load_image(manifest[0])
    masks = s.previous.make_masks(4, s.DEVICE)
    useful = s.useful_for(0, 0, 4)
    messages = s.protected_messages(useful)
    os.chdir(s.UPSTREAM)
    wam = s.load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth").to(s.DEVICE).eval()
    marked, _ = s.previous.embed_messages(wam, image, messages, masks)
    retained, variant, ratio, x0, y0, width, height = next(
        crop for crop in fixed_crops(0, 0) if crop[0] == .5 and crop[1] == 0)
    cropped = marked[:, :, y0:y0+height, x0:x0+width]
    detection = s.detect_blind(wam, cropped)
    scoring_masks = masks[:, :, y0:y0+height, x0:x0+width]
    details, _ = s.score_regions(detection, useful, scoring_masks, messages)
    original_rgb = s.previous.tensor_rgb(image)
    marked_rgb = s.previous.tensor_rgb(marked)
    cropped_rgb = s.previous.tensor_rgb(cropped)
    overlay = cropped_rgb.copy()
    union = scoring_masks.sum(0)[0].numpy() > .5
    overlay[union] = .55 * overlay[union] + .45 * np.array([1, .1, .1])
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="yellow", linewidth=3))
    axes[0, 0].set_title("Original 512 × 512; yellow = crop")
    axes[0, 1].imshow(marked_rgb)
    axes[0, 1].add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="yellow", linewidth=3))
    axes[0, 1].set_title("WAM marked; yellow = crop")
    axes[0, 2].imshow(cropped_rgb)
    axes[0, 2].set_title("Only input to blind WAM decoder")
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Retained source masks (scoring only)")
    axes[1, 1].imshow(detection["mask_probability"][0], cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title("Official WAM localization probability")
    axes[1, 2].imshow(detection["labels"], cmap="viridis")
    axes[1, 2].set_title("Official DBSCAN message clusters")
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle("Example: astronaut, seed 0, blind 50% crop; " +
                 f"{sum(row['useful_exact'] for row in details)}/4 protected messages exact")
    fig.savefig(s.OUT / "blind_crop_50pct.png", dpi=160)
    plt.close(fig)
    s.save_json("blind_crop_50pct_example.json", {
        "image": "astronaut", "payload_seed": 0, "retained_target": retained,
        "variant": variant, "aspect_ratio": ratio, "x0": x0, "y0": y0,
        "width": width, "height": height,
        "exact_useful_packets": sum(row["useful_exact"] for row in details),
        "region_details": details,
        "note": "The full original and oracle masks appear only in this scoring figure; decoder input was cropped marked pixels alone.",
    })


if __name__ == "__main__":
    main()
