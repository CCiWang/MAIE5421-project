"""Official WAM staged method-screening experiment.

Run from the pinned upstream repository with PYTHONPATH pointing at it.
The script stops at the first failed decision gate.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from notebooks.inference_utils import load_model_from_checkpoint, multiwm_dbscan
from watermark_anything.data.transforms import default_transform, unnormalize_img


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "watermark-anything"
OUT = ROOT / "official_results"
INPUTS = OUT / "inputs"
VIZ = OUT / "visualizations"
for directory in (OUT, INPUTS, VIZ):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20261007
DEVICE = torch.device("cpu")
EPSILON = 1
MIN_SAMPLES = 500
IMAGE_SIZE = 512
IMAGE_FILES = {
    "astronaut": "astronaut.png",
    "coffee": "coffee.png",
    "chelsea": "chelsea.png",
    "rocket": "rocket.jpg",
    "hubble": "hubble_deep_field.jpg",
    "immunohistochemistry": "ihc.png",
    "retina": "retina.jpg",
    "motorcycle": "motorcycle_left.png",
    "camera": "camera.png",
    "coins": "coins.png",
}
REGIONS = [
    (6, 35, 116, 240), (132, 35, 242, 240),
    (258, 35, 368, 240), (384, 35, 494, 240),
    (6, 272, 116, 477), (132, 272, 242, 477),
    (258, 272, 368, 477), (384, 272, 494, 477),
]
REGION_SELECTIONS = {2: [0, 7], 4: [0, 3, 4, 7], 8: list(range(8))}

# GF(256), primitive polynomial 0x11d. Coefficient tables are public protocol state.
GF_EXP = np.zeros(512, dtype=np.uint16)
GF_LOG = np.zeros(256, dtype=np.int16)
gf_value = 1
for gf_index in range(255):
    GF_EXP[gf_index], GF_LOG[gf_value] = gf_value, gf_index
    gf_value <<= 1
    if gf_value & 256:
        gf_value ^= 0x11D
GF_EXP[255:] = GF_EXP[np.arange(257) % 255]
GF_MUL = np.zeros((256, 256), dtype=np.uint8)
for gf_a in range(1, 256):
    for gf_b in range(1, 256):
        GF_MUL[gf_a, gf_b] = GF_EXP[int(GF_LOG[gf_a]) + int(GF_LOG[gf_b])]

IDENTITY = np.eye(4, dtype=np.uint8)
CODING_MATRICES = {
    "raw_padded": np.concatenate([IDENTITY, np.zeros((4, 4), dtype=np.uint8)]),
    "repetition": IDENTITY[[0, 1, 2, 3, 2, 3, 0, 1]],
    "rlnc": np.concatenate([
        IDENTITY,
        np.random.default_rng(SEED).integers(0, 256, (4, 4), dtype=np.uint8),
    ]),
    "mds": np.array(
        [[1, x, GF_MUL[x, x], GF_MUL[GF_MUL[x, x], x]] for x in range(1, 9)],
        dtype=np.uint8,
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def log_event(kind: str, **payload) -> None:
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, **payload}
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def load_images() -> list[dict]:
    import skimage.data

    data_dir = Path(skimage.data.__file__).parent
    records = []
    for name, filename in IMAGE_FILES.items():
        source = data_dir / filename
        image = Image.open(source).convert("RGB")
        image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)
        output = INPUTS / f"{name}.png"
        image.save(output)
        records.append({
            "name": name,
            "source": str(source),
            "source_sha256": sha256(source),
            "processed": str(output),
            "processed_sha256": sha256(output),
            "shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "provenance": "scikit-image 0.24.0 bundled public sample data",
        })
    (OUT / "image_manifest.json").write_text(json.dumps(records, indent=2))
    return records


def make_masks(count: int, device: torch.device) -> torch.Tensor:
    masks = torch.zeros((count, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32, device=device)
    for local_index, region_index in enumerate(REGION_SELECTIONS[count]):
        x0, y0, x1, y1 = REGIONS[region_index]
        masks[local_index, :, y0:y1, x0:x1] = 1
    return masks


def messages_for(image_index: int, payload_seed: int, count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + image_index * 1009 + payload_seed * 97 + count)
    messages = torch.randint(0, 2, (count, 32), generator=generator).float()
    # Ensure this screening set has distinct messages.
    if len({tuple(row.tolist()) for row in messages}) != count:
        raise RuntimeError("Unexpected duplicate message in fixed screening set")
    return messages


def embed_messages(wam, image: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor):
    watermarked = image.clone()
    start = time.perf_counter()
    with torch.inference_mode():
        for index in range(len(messages)):
            output = wam.embed(image, messages[index].unsqueeze(0).to(image.device))
            watermarked = output["imgs_w"] * masks[index] + watermarked * (1 - masks[index])
    return watermarked, time.perf_counter() - start


def tensor_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = unnormalize_img(tensor.detach()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return array


def detect_messages(wam, image: torch.Tensor, gt_messages: torch.Tensor, gt_masks: torch.Tensor | None = None):
    start = time.perf_counter()
    with torch.inference_mode():
        preds = wam.detect(image)["preds"]
    mask_prob = torch.sigmoid(preds[:, 0])
    bit_logits = preds[:, 1:]
    detected_pixels = int((mask_prob > 0.5).sum().item())
    centroids = {}
    labels = torch.full(mask_prob.shape[-2:], -1.0)
    clustering_error = None
    if detected_pixels:
        try:
            centroids, labels = multiwm_dbscan(
                bit_logits, mask_prob, epsilon=EPSILON, min_samples=MIN_SAMPLES
            )
        except Exception as error:
            clustering_error = f"{type(error).__name__}: {error}"
    cluster_items = sorted(centroids.items(), key=lambda item: int(item[0]))
    predictions = torch.stack([item[1].float() for item in cluster_items]) if cluster_items else torch.empty((0, 32))
    truths = gt_messages.detach().cpu().float()
    matched = []
    if len(predictions) and len(truths):
        hamming = (predictions[:, None, :] != truths[None, :, :]).sum(2).numpy()
        rows, columns = linear_sum_assignment(hamming)
        for row, column in zip(rows, columns):
            label_value = cluster_items[row][0]
            pixel_mask = labels == float(label_value)
            bit_prob = torch.sigmoid(bit_logits[0].detach().cpu())
            confidence = float((2 * (bit_prob[:, pixel_mask] - 0.5).abs()).mean()) if pixel_mask.any() else 0.0
            matched.append({
                "cluster": int(label_value),
                "truth": int(column),
                "hamming": int(hamming[row, column]),
                "bit_accuracy": float(1 - hamming[row, column] / 32),
                "confidence": confidence,
            })
    exact = sum(item["hamming"] == 0 for item in matched)
    false_clusters = sum(not any(torch.equal(pred, truth) for truth in truths) for pred in predictions)
    best_mean_accuracy = float(np.mean([item["bit_accuracy"] for item in matched])) if matched else 0.0
    union_precision = union_recall = union_iou = None
    if gt_masks is not None:
        resized_gt = F.interpolate(gt_masks.sum(0, keepdim=True).clamp(0, 1), size=(256, 256), mode="nearest")[0, 0] > 0.5
        predicted_union = mask_prob[0].detach().cpu() > 0.5
        gt = resized_gt.detach().cpu()
        intersection = int((predicted_union & gt).sum())
        pred_count = int(predicted_union.sum())
        gt_count = int(gt.sum())
        union = int((predicted_union | gt).sum())
        union_precision = intersection / pred_count if pred_count else 0.0
        union_recall = intersection / gt_count if gt_count else 0.0
        union_iou = intersection / union if union else 1.0
    return {
        "detected_clusters": len(predictions),
        "exact_packets": exact,
        "packet_recall": exact / len(truths),
        "mean_matched_bit_accuracy": best_mean_accuracy,
        "false_clusters": int(false_clusters),
        "detected_pixels": detected_pixels,
        "union_precision": union_precision,
        "union_recall": union_recall,
        "union_iou": union_iou,
        "mean_confidence": float(np.mean([item["confidence"] for item in matched])) if matched else 0.0,
        "runtime_seconds": time.perf_counter() - start,
        "clustering_error": clustering_error,
        "matched": matched,
        "mask_prob": mask_prob.detach().cpu(),
        "labels": labels.detach().cpu(),
        "predictions": predictions,
    }


def save_visualization(path: Path, original: torch.Tensor, watermarked: torch.Tensor, masks: torch.Tensor, detected: dict, title: str):
    original_np = tensor_rgb(original)
    watermarked_np = tensor_rgb(watermarked)
    mask_union = masks.sum(0)[0].detach().cpu().numpy()
    mask_overlay = original_np.copy()
    mask_overlay[mask_union > 0.5] = 0.55 * mask_overlay[mask_union > 0.5] + 0.45 * np.array([1.0, 0.1, 0.1])
    labels = detected["labels"].numpy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        (original_np, "Original"),
        (mask_overlay, "Embedding regions"),
        (watermarked_np, "Watermarked"),
        (np.clip(np.abs(watermarked_np - original_np) * 12, 0, 1), "12x absolute difference"),
        (detected["mask_prob"][0].numpy(), "WAM localization probability"),
        (labels, "Official DBSCAN message clusters"),
    ]
    for axis, (content, label) in zip(axes.flat, panels):
        axis.imshow(content, cmap="viridis" if content.ndim == 2 else None)
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize_clean(rows: list[dict], count: int) -> dict:
    selected = [row for row in rows if row["region_count"] == count]
    packets = sum(row["exact_packets"] for row in selected)
    total_packets = len(selected) * count
    return {
        "region_count": count,
        "cases": len(selected),
        "exact_packet_recall": packets / total_packets,
        "all_exact_image_rate": sum(row["exact_packets"] == count for row in selected) / len(selected),
        "at_least_four_exact_rate": sum(row["exact_packets"] >= 4 for row in selected) / len(selected),
        "mean_bit_accuracy": float(np.mean([row["mean_matched_bit_accuracy"] for row in selected])),
        "mean_clusters": float(np.mean([row["detected_clusters"] for row in selected])),
        "mean_false_clusters": float(np.mean([row["false_clusters"] for row in selected])),
        "mean_psnr": float(np.mean([row["psnr"] for row in selected])),
        "mean_ssim": float(np.mean([row["ssim"] for row in selected])),
        "mean_union_iou": float(np.mean([row["union_iou"] for row in selected])),
        "mean_embed_seconds": float(np.mean([row["embed_seconds"] for row in selected])),
        "mean_detect_seconds": float(np.mean([row["detect_seconds"] for row in selected])),
    }


def crop_dimensions(area: float, ratio: float) -> tuple[int, int]:
    width = min(IMAGE_SIZE, max(32, round(math.sqrt(area * IMAGE_SIZE * IMAGE_SIZE * ratio))))
    height = min(IMAGE_SIZE, max(32, round(area * IMAGE_SIZE * IMAGE_SIZE / width)))
    return width, height


def clean_phase(wam, images: list[dict]):
    rows = []
    cached_eight = []
    summaries = []
    stop = None
    for count in (2, 4, 8):
        masks = make_masks(count, DEVICE)
        for image_index, record in enumerate(images):
            image_pil = Image.open(record["processed"]).convert("RGB")
            image = default_transform(image_pil).unsqueeze(0).to(DEVICE)
            for payload_seed in range(3):
                messages = messages_for(image_index, payload_seed, count)
                watermarked, embed_seconds = embed_messages(wam, image, messages, masks)
                detected = detect_messages(wam, watermarked, messages, masks)
                original_np, marked_np = tensor_rgb(image), tensor_rgb(watermarked)
                row = {
                    "phase": "clean",
                    "image": record["name"],
                    "image_index": image_index,
                    "payload_seed": payload_seed,
                    "region_count": count,
                    "psnr": float(peak_signal_noise_ratio(original_np, marked_np, data_range=1.0)),
                    "ssim": float(structural_similarity(original_np, marked_np, channel_axis=2, data_range=1.0)),
                    "embed_seconds": embed_seconds,
                    "detect_seconds": detected["runtime_seconds"],
                    **{key: detected[key] for key in [
                        "detected_clusters", "exact_packets", "packet_recall",
                        "mean_matched_bit_accuracy", "false_clusters", "detected_pixels",
                        "union_precision", "union_recall", "union_iou", "mean_confidence", "clustering_error",
                    ]},
                    "matches_json": json.dumps(detected["matched"]),
                }
                rows.append(row)
                log_event("clean_case", image=record["name"], payload_seed=payload_seed, region_count=count,
                          exact_packets=row["exact_packets"], clusters=row["detected_clusters"])
                if image_index == 0 and payload_seed == 0:
                    save_visualization(VIZ / f"clean_{count}_regions.png", image, watermarked, masks, detected, f"Clean: {count} localized messages")
                if count == 8 and payload_seed == 0:
                    cached_eight.append({"record": record, "image": image.cpu(), "messages": messages.cpu(), "masks": masks.cpu(), "watermarked": watermarked.cpu()})
        summary = summarize_clean(rows, count)
        summaries.append(summary)
        write_csv(OUT / "clean_cases.csv", rows)
        (OUT / "clean_summary.json").write_text(json.dumps(summaries, indent=2))
        if count == 2 and not (summary["exact_packet_recall"] >= 0.95 and summary["all_exact_image_rate"] >= 0.90):
            stop = "Clean two-region gate failed"
        elif count == 4 and not (summary["exact_packet_recall"] >= 0.90 and summary["all_exact_image_rate"] >= 0.80):
            stop = "Clean four-region gate failed"
        elif count == 8 and not (summary["exact_packet_recall"] >= 0.85 and summary["at_least_four_exact_rate"] >= 0.90):
            stop = "Clean eight-region gate failed"
        if stop:
            break
    return rows, summaries, cached_eight, stop


def crop_phase(wam, cached_eight: list[dict]):
    rows = []
    ratios = [1.0, 4 / 3, 3 / 4]
    for case_index, case in enumerate(cached_eight):
        image = case["watermarked"].to(DEVICE)
        messages = case["messages"]
        masks = case["masks"]
        rng = np.random.default_rng(SEED + 50000 + case_index)
        for retained in (0.75, 0.50, 0.25):
            for ratio_index, ratio in enumerate(ratios):
                width, height = crop_dimensions(retained, ratio)
                for offset_index in range(2):
                    x0 = int(rng.integers(0, IMAGE_SIZE - width + 1))
                    y0 = int(rng.integers(0, IMAGE_SIZE - height + 1))
                    crop = image[:, :, y0:y0 + height, x0:x0 + width]
                    crop_masks = masks[:, :, y0:y0 + height, x0:x0 + width]
                    detected = detect_messages(wam, crop, messages, crop_masks)
                    row = {
                        "phase": "crop",
                        "image": case["record"]["name"],
                        "retained_target": retained,
                        "retained_actual": width * height / (IMAGE_SIZE * IMAGE_SIZE),
                        "aspect_ratio": ratio,
                        "offset_index": offset_index,
                        "x0": x0, "y0": y0, "width": width, "height": height,
                        **{key: detected[key] for key in [
                            "detected_clusters", "exact_packets", "packet_recall",
                            "mean_matched_bit_accuracy", "false_clusters", "detected_pixels",
                            "union_precision", "union_recall", "union_iou", "mean_confidence",
                            "runtime_seconds", "clustering_error",
                        ]},
                        "matches_json": json.dumps(detected["matched"]),
                    }
                    rows.append(row)
                    log_event("crop_case", image=row["image"], retained=retained, ratio=ratio,
                              exact_packets=row["exact_packets"], clusters=row["detected_clusters"])
                    if case_index == 0 and retained == 0.50 and ratio_index == 0 and offset_index == 0:
                        save_visualization(VIZ / "unknown_offset_crop_50.png", crop, crop, crop_masks, detected,
                                           "Unknown-offset 50% crop: WAM receives cropped image only")
    write_csv(OUT / "crop_cases.csv", rows)
    summary = []
    for retained in (0.75, 0.50, 0.25):
        selected = [row for row in rows if row["retained_target"] == retained]
        summary.append({
            "retained_target": retained,
            "cases": len(selected),
            "mean_exact_packets": float(np.mean([row["exact_packets"] for row in selected])),
            "at_least_four_exact_rate": float(np.mean([row["exact_packets"] >= 4 for row in selected])),
            "exact_packet_recall": sum(row["exact_packets"] for row in selected) / (8 * len(selected)),
            "mean_false_clusters": float(np.mean([row["false_clusters"] for row in selected])),
            "mean_bit_accuracy": float(np.mean([row["mean_matched_bit_accuracy"] for row in selected])),
            "mean_union_iou": float(np.mean([row["union_iou"] for row in selected])),
        })
    primary = [row for row in rows if row["retained_target"] in (0.75, 0.50)]
    primary_rate = float(np.mean([row["exact_packets"] >= 4 for row in primary]))
    result = {"by_retained_area": summary, "primary_50_75_at_least_four_rate": primary_rate, "gate_pass": primary_rate >= 0.60}
    (OUT / "crop_summary.json").write_text(json.dumps(result, indent=2))
    return rows, result


def gf_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.bitwise_xor.reduce(GF_MUL[left[:, :, None], right[None, :, :]], axis=1)


def gf_solve(coefficients: np.ndarray, values: np.ndarray) -> np.ndarray | None:
    if len(coefficients) < 4:
        return None
    augmented = np.concatenate([
        np.asarray(coefficients, dtype=np.uint8),
        np.asarray(values, dtype=np.uint8),
    ], axis=1)
    row = 0
    for column in range(4):
        pivot = next((candidate for candidate in range(row, len(augmented)) if augmented[candidate, column]), None)
        if pivot is None:
            return None
        augmented[[row, pivot]] = augmented[[pivot, row]]
        inverse = int(GF_EXP[255 - int(GF_LOG[augmented[row, column]])])
        augmented[row] = GF_MUL[inverse, augmented[row]]
        for candidate in range(len(augmented)):
            if candidate != row and augmented[candidate, column]:
                augmented[candidate] ^= GF_MUL[augmented[candidate, column], augmented[row]]
        row += 1
    # Remaining equations must be consistent with the selected pivot solution.
    if len(augmented) > 4 and np.any(augmented[4:, 4:]):
        return None
    return augmented[:4, 4:].copy()


def crc8(values: bytes) -> int:
    result = 0
    for value in values:
        result ^= value
        for _ in range(8):
            result = ((result << 1) ^ (0x07 if result & 0x80 else 0)) & 0xFF
    return result


def packet_messages(source: np.ndarray, method: str) -> torch.Tensor:
    coded = gf_matmul(CODING_MATRICES[method], source)
    packets = []
    for packet_id, symbol in enumerate(coded):
        body = bytes([packet_id, int(symbol[0]), int(symbol[1])])
        bits = np.unpackbits(np.frombuffer(body + bytes([crc8(body)]), dtype=np.uint8))
        packets.append(bits)
    return torch.tensor(np.stack(packets), dtype=torch.float32)


def parse_packet(bits: torch.Tensor):
    packed = np.packbits(bits.detach().cpu().numpy().astype(np.uint8)).tobytes()
    if len(packed) != 4 or packed[0] >= 8 or crc8(packed[:3]) != packed[3]:
        return None
    return int(packed[0]), np.frombuffer(packed[1:3], dtype=np.uint8).copy()


def recover_global(predictions: torch.Tensor, method: str, source: np.ndarray) -> dict:
    accepted = {}
    conflicting_ids = set()
    for prediction in predictions:
        parsed = parse_packet(prediction)
        if parsed is None:
            continue
        packet_id, value = parsed
        if packet_id in accepted and not np.array_equal(accepted[packet_id], value):
            conflicting_ids.add(packet_id)
        else:
            accepted[packet_id] = value
    for packet_id in conflicting_ids:
        accepted.pop(packet_id, None)
    indices = sorted(accepted)
    recovered = gf_solve(CODING_MATRICES[method][indices], np.stack([accepted[index] for index in indices])) if indices else None
    exact = bool(recovered is not None and np.array_equal(recovered, source))
    return {
        "valid_packets": len(indices),
        "conflicting_ids": len(conflicting_ids),
        "decoder_output": recovered is not None,
        "exact_global_payload": exact,
        "wrong_global_payload": bool(recovered is not None and not exact),
    }


def fixed_crops(image_index: int):
    rng = np.random.default_rng(SEED + 90000 + image_index)
    for retained in (0.75, 0.50):
        for ratio in (1.0, 4 / 3, 3 / 4):
            width, height = crop_dimensions(retained, ratio)
            x0 = int(rng.integers(0, IMAGE_SIZE - width + 1))
            y0 = int(rng.integers(0, IMAGE_SIZE - height + 1))
            yield retained, ratio, x0, y0, width, height


def coding_phase(wam, images: list[dict]):
    rows = []
    quality_rows = []
    mds_cache = []
    masks = make_masks(8, DEVICE)
    for method in ("raw_padded", "repetition", "rlnc", "mds"):
        for image_index, record in enumerate(images):
            image = default_transform(Image.open(record["processed"]).convert("RGB")).unsqueeze(0).to(DEVICE)
            source = np.random.default_rng(SEED + 70000 + image_index).integers(0, 256, (4, 2), dtype=np.uint8)
            messages = packet_messages(source, method)
            watermarked, embed_seconds = embed_messages(wam, image, messages, masks)
            original_np, marked_np = tensor_rgb(image), tensor_rgb(watermarked)
            quality = {
                "method": method,
                "image": record["name"],
                "psnr": float(peak_signal_noise_ratio(original_np, marked_np, data_range=1.0)),
                "ssim": float(structural_similarity(original_np, marked_np, channel_axis=2, data_range=1.0)),
                "embed_seconds": embed_seconds,
            }
            quality_rows.append(quality)
            detected = detect_messages(wam, watermarked, messages, masks)
            recovered = recover_global(detected["predictions"], method, source)
            rows.append({
                "method": method, "image": record["name"], "attack": "clean",
                "retained_target": 1.0, "aspect_ratio": 1.0,
                "exact_packets": detected["exact_packets"], "detected_clusters": detected["detected_clusters"],
                "false_clusters": detected["false_clusters"], "detect_seconds": detected["runtime_seconds"],
                **recovered,
            })
            if method == "mds":
                mds_cache.append({
                    "record": record, "image": image.cpu(), "watermarked": watermarked.cpu(),
                    "messages": messages.cpu(), "masks": masks.cpu(), "source": source,
                })
            for retained, ratio, x0, y0, width, height in fixed_crops(image_index):
                crop = watermarked[:, :, y0:y0 + height, x0:x0 + width]
                crop_masks = masks[:, :, y0:y0 + height, x0:x0 + width]
                detected = detect_messages(wam, crop, messages, crop_masks)
                recovered = recover_global(detected["predictions"], method, source)
                rows.append({
                    "method": method, "image": record["name"], "attack": "unknown_offset_crop",
                    "retained_target": retained, "aspect_ratio": ratio,
                    "x0": x0, "y0": y0, "width": width, "height": height,
                    "exact_packets": detected["exact_packets"], "detected_clusters": detected["detected_clusters"],
                    "false_clusters": detected["false_clusters"], "detect_seconds": detected["runtime_seconds"],
                    **recovered,
                })
            log_event("coding_image", method=method, image=record["name"])
            write_csv(OUT / "coding_cases.csv", rows)
            write_csv(OUT / "coding_quality.csv", quality_rows)
    summary = []
    for method in ("raw_padded", "repetition", "rlnc", "mds"):
        q = [row for row in quality_rows if row["method"] == method]
        for attack, retained in (("clean", 1.0), ("unknown_offset_crop", 0.75), ("unknown_offset_crop", 0.50)):
            selected = [row for row in rows if row["method"] == method and row["attack"] == attack and row["retained_target"] == retained]
            summary.append({
                "method": method, "attack": attack, "retained_target": retained, "cases": len(selected),
                "exact_global_recovery_rate": float(np.mean([row["exact_global_payload"] for row in selected])),
                "wrong_global_outputs": int(sum(row["wrong_global_payload"] for row in selected)),
                "mean_valid_packets": float(np.mean([row["valid_packets"] for row in selected])),
                "mean_exact_packets": float(np.mean([row["exact_packets"] for row in selected])),
                "mean_psnr": float(np.mean([row["psnr"] for row in q])),
                "mean_ssim": float(np.mean([row["ssim"] for row in q])),
            })
    def recovery(method):
        selected = [row for row in rows if row["method"] == method and row["attack"] == "unknown_offset_crop"]
        return float(np.mean([row["exact_global_payload"] for row in selected]))
    rep_q = next(row for row in summary if row["method"] == "repetition" and row["attack"] == "clean")
    mds_q = next(row for row in summary if row["method"] == "mds" and row["attack"] == "clean")
    gate = {
        "repetition_crop_recovery": recovery("repetition"),
        "mds_crop_recovery": recovery("mds"),
        "mds_minus_repetition": recovery("mds") - recovery("repetition"),
        "psnr_difference_db": mds_q["mean_psnr"] - rep_q["mean_psnr"],
        "ssim_difference": mds_q["mean_ssim"] - rep_q["mean_ssim"],
    }
    gate["pass"] = bool(gate["mds_minus_repetition"] > 0 and abs(gate["psnr_difference_db"]) <= 0.5 and abs(gate["ssim_difference"]) <= 0.005)
    result = {"rows": summary, "gate": gate}
    (OUT / "coding_summary.json").write_text(json.dumps(result, indent=2))
    return rows, result, mds_cache


def tensor_from_rgb(array: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(np.clip(np.rint(array * 255), 0, 255).astype(np.uint8), mode="RGB")
    return default_transform(image).unsqueeze(0)


def jpeg_attack(tensor: torch.Tensor, quality: int) -> torch.Tensor:
    image = Image.fromarray(np.clip(np.rint(tensor_rgb(tensor) * 255), 0, 255).astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    return default_transform(Image.open(buffer).convert("RGB")).unsqueeze(0)


def noise_attack(tensor: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    rgb = tensor_rgb(tensor)
    return tensor_from_rgb(np.clip(rgb + rng.normal(0, sigma / 255, rgb.shape), 0, 1))


def rotate_attack(tensor: torch.Tensor, angle: float) -> torch.Tensor:
    image = Image.fromarray(np.clip(np.rint(tensor_rgb(tensor) * 255), 0, 255).astype(np.uint8), mode="RGB")
    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    return default_transform(rotated).unsqueeze(0)


def diagnostic_phase(wam, mds_cache: list[dict]):
    rows = []
    attacks = ["jpeg75", "jpeg50", "noise2", "noise5"] + [f"rotate{angle:+d}" for angle in (-10, -5, -2, -1, 1, 2, 5, 10)]
    for case_index, case in enumerate(mds_cache):
        for attack in attacks:
            if attack.startswith("jpeg"):
                attacked = jpeg_attack(case["watermarked"], int(attack[4:]))
            elif attack.startswith("noise"):
                attacked = noise_attack(case["watermarked"], float(attack[5:]), SEED + case_index)
            else:
                attacked = rotate_attack(case["watermarked"], float(attack[6:]))
            detected = detect_messages(wam, attacked, case["messages"], None)
            recovered = recover_global(detected["predictions"], "mds", case["source"])
            rows.append({
                "image": case["record"]["name"], "attack": attack,
                "exact_packets": detected["exact_packets"], "detected_clusters": detected["detected_clusters"],
                "false_clusters": detected["false_clusters"], "mean_bit_accuracy": detected["mean_matched_bit_accuracy"],
                "mean_confidence": detected["mean_confidence"], "runtime_seconds": detected["runtime_seconds"],
                **recovered,
            })
        # One combined diagnostic requested by the protocol.
        retained, ratio, x0, y0, width, height = next(fixed_crops(case_index))
        cropped = case["watermarked"][:, :, y0:y0 + height, x0:x0 + width]
        attacked = jpeg_attack(cropped, 75)
        detected = detect_messages(wam, attacked, case["messages"], None)
        recovered = recover_global(detected["predictions"], "mds", case["source"])
        rows.append({
            "image": case["record"]["name"], "attack": "crop75+jpeg75",
            "exact_packets": detected["exact_packets"], "detected_clusters": detected["detected_clusters"],
            "false_clusters": detected["false_clusters"], "mean_bit_accuracy": detected["mean_matched_bit_accuracy"],
            "mean_confidence": detected["mean_confidence"], "runtime_seconds": detected["runtime_seconds"],
            **recovered,
        })
        log_event("diagnostic_image", image=case["record"]["name"])
        write_csv(OUT / "diagnostic_cases.csv", rows)
    summary = []
    for attack in attacks + ["crop75+jpeg75"]:
        selected = [row for row in rows if row["attack"] == attack]
        summary.append({
            "attack": attack, "cases": len(selected),
            "exact_global_recovery_rate": float(np.mean([row["exact_global_payload"] for row in selected])),
            "mean_exact_packets": float(np.mean([row["exact_packets"] for row in selected])),
            "mean_bit_accuracy": float(np.mean([row["mean_bit_accuracy"] for row in selected])),
            "wrong_global_outputs": int(sum(row["wrong_global_payload"] for row in selected)),
        })
    rotation_failures = any(row["exact_global_recovery_rate"] < 1.0 for row in summary if row["attack"].startswith("rotate"))
    search_rows = []
    if rotation_failures:
        for case in mds_cache[:3]:
            attacked = rotate_attack(case["watermarked"], 2)
            candidates = []
            for correction in (-10, -5, -2, 0, 2, 5, 10):
                corrected = rotate_attack(attacked, correction)
                detected = detect_messages(wam, corrected, case["messages"], None)
                recovered = recover_global(detected["predictions"], "mds", case["source"])
                candidates.append({
                    "correction": correction,
                    "valid_packets": recovered["valid_packets"],
                    "exact_packets": detected["exact_packets"],
                    "confidence": detected["mean_confidence"],
                    "exact_global_payload": recovered["exact_global_payload"],
                })
            best = max(candidates, key=lambda row: (row["valid_packets"], row["exact_packets"], row["confidence"]))
            search_rows.append({"image": case["record"]["name"], "attack_angle": 2, **best, "all_candidates_json": json.dumps(candidates)})
        write_csv(OUT / "rotation_search.csv", search_rows)
    result = {"attacks": summary, "rotation_search": search_rows}
    (OUT / "diagnostic_summary.json").write_text(json.dumps(result, indent=2))
    return rows, result


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    start = time.time()
    checkpoint = UPSTREAM / "checkpoints" / "wam_mit.pth"
    environment = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "mps_available": bool(torch.backends.mps.is_available()),
        "wam_commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "seed": SEED,
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2))
    images = load_images()
    os.chdir(UPSTREAM)
    wam = load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth").to(DEVICE).eval()
    log_event("model_loaded", parameters=sum(p.numel() for p in wam.parameters()), device=str(DEVICE))

    clean_rows, clean_summary, cached_eight, stop = clean_phase(wam, images)
    final = {
        "decision": "NO-GO",
        "last_completed_phase": "clean",
        "stop_reason": stop,
        "clean_summary": clean_summary,
        "crop_summary": None,
        "coding_summary": None,
        "diagnostics": "NOT RUN",
        "total_seconds": time.time() - start,
    }
    if not stop:
        crop_rows, crop_summary = crop_phase(wam, cached_eight)
        final["last_completed_phase"] = "unknown_offset_crop"
        final["crop_summary"] = crop_summary
        if not crop_summary["gate_pass"]:
            final["stop_reason"] = "Unknown-offset crop gate failed"
        else:
            coding_rows, coding_summary, mds_cache = coding_phase(wam, images)
            diagnostics_rows, diagnostics = diagnostic_phase(wam, mds_cache)
            final["last_completed_phase"] = "coding_and_diagnostics"
            final["coding_summary"] = coding_summary
            final["diagnostics"] = diagnostics
            if coding_summary["gate"]["pass"]:
                final["decision"] = "GO"
                final["stop_reason"] = None
            else:
                final["stop_reason"] = "MDS did not improve exact global recovery over strong repetition at the quality gate"
    final["total_seconds"] = time.time() - start
    (OUT / "final_result.json").write_text(json.dumps(final, indent=2))
    log_event("run_complete", decision=final["decision"], stop_reason=final["stop_reason"], seconds=final["total_seconds"])
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
