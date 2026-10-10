"""Staged official-WAM inner-ECC method validation; stops at first failed gate."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "validation_wam_20261007"
UPSTREAM = OLD / "watermark-anything"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(OLD))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from notebooks.inference_utils import load_model_from_checkpoint, multiwm_dbscan
from watermark_anything.data.transforms import default_transform
import run_validation as previous
from bch31 import decode_int, encode_int, int_to_bits, bits_to_int

SEED = 20261008
DEVICE = torch.device("cpu")
IMAGE_SIZE = 512
EXPECTED_COMMIT = "2c08af04d037d5667c02f6ddebbda9ff04581c3e"
EXPECTED_CHECKPOINT = "90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def save_json(name: str, data):
    (OUT / name).write_text(json.dumps(data, indent=2, default=str) + "\n")


def save_csv(name: str, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def log(kind: str, **values):
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, **values}
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, default=str) + "\n")
    print(json.dumps(record, default=str), flush=True)


def verify_environment():
    checkpoint = UPSTREAM / "checkpoints" / "wam_mit.pth"
    manifest = json.loads((OLD / "official_results" / "image_manifest.json").read_text())
    for record in manifest:
        for label in ("source", "processed"):
            assert sha256(Path(record[label])) == record[f"{label}_sha256"], record["name"]
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_executable": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "numpy": np.__version__, "pillow": Image.__version__,
        "scikit_image": __import__("skimage").__version__,
        "wam_commit": command_output("git", "-C", str(UPSTREAM), "rev-parse", "HEAD"),
        "wam_git_status": command_output("git", "-C", str(UPSTREAM), "status", "--short"),
        "project_git_status": command_output("git", "-C", str(ROOT.parent), "status", "--short"),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "mps_available": bool(torch.backends.mps.is_available()), "device": str(DEVICE),
        "image_manifest_sha256": sha256(OLD / "official_results" / "image_manifest.json"),
        "previous_clean_cases_sha256": sha256(OLD / "official_results" / "clean_cases.csv"),
    }
    assert record["wam_commit"] == EXPECTED_COMMIT
    assert record["wam_git_status"] == ""
    assert record["checkpoint_sha256"] == EXPECTED_CHECKPOINT
    assert len(manifest) == 10
    save_json("environment.json", record)
    save_json("image_manifest_reused.json", manifest)
    log("environment_verified", commit=record["wam_commit"], checkpoint_sha256=record["checkpoint_sha256"])
    return manifest


def useful_for(image_index: int, payload_seed: int, count: int) -> list[int]:
    rng = np.random.default_rng(SEED + image_index * 1009 + payload_seed * 97 + count)
    values = [int(x) for x in rng.integers(0, 1 << 16, count, dtype=np.uint16)]
    assert len(set(values)) == count
    return values


def protected_messages(useful: list[int]) -> torch.Tensor:
    return torch.tensor([int_to_bits(encode_int(value)) for value in useful], dtype=torch.float32)


def detect_blind(wam, image: torch.Tensor) -> dict:
    start = time.perf_counter()
    with torch.inference_mode():
        preds = wam.detect(image)["preds"]
    mask_probability = torch.sigmoid(preds[:, 0]).detach().cpu()
    bit_logits = preds[:, 1:].detach().cpu()
    centroids = {}
    labels = torch.full(mask_probability.shape[-2:], -1.0)
    error = None
    if bool((mask_probability > 0.5).any()):
        try:
            centroids, labels = multiwm_dbscan(bit_logits, mask_probability, epsilon=1, min_samples=500)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    clusters = []
    for label, bits in sorted(centroids.items(), key=lambda pair: int(pair[0])):
        binary = bits.detach().cpu().numpy().astype(np.uint8)
        word = bits_to_int(binary)
        t0 = time.perf_counter()
        decoded = decode_int(word)
        ecc_seconds = time.perf_counter() - t0
        pixel_mask = labels == float(label)
        bit_prob = torch.sigmoid(bit_logits[0])
        confidence = float((2 * (bit_prob[:, pixel_mask] - 0.5).abs()).mean()) if pixel_mask.any() else 0.0
        clusters.append({
            "label": int(label), "raw_word_hex": f"{word:08x}", "raw_word": word,
            "mask": pixel_mask, "decoded": decoded, "confidence": confidence,
            "ecc_decode_seconds": ecc_seconds,
        })
    return {"clusters": clusters, "labels": labels, "mask_probability": mask_probability,
            "runtime_seconds": time.perf_counter() - start, "clustering_error": error}


def score_regions(detected: dict, useful: list[int], masks: torch.Tensor, messages: torch.Tensor,
                  retained_fractions: list[float] | None = None) -> tuple[list[dict], dict]:
    """Use oracle masks/payloads only after blind detection and BCH decoding."""
    count = len(useful)
    clusters = detected["clusters"]
    ground = F.interpolate(masks.detach().cpu(), size=detected["labels"].shape[-2:], mode="nearest")[:, 0] > 0.5
    overlaps = np.zeros((len(clusters), count), dtype=np.int32)
    for ci, cluster in enumerate(clusters):
        for ri in range(count):
            overlaps[ci, ri] = int((cluster["mask"] & ground[ri]).sum().item())
    paired = {}
    if len(clusters):
        candidate_clusters, candidate_regions = linear_sum_assignment(-overlaps)
        for ci, ri in zip(candidate_clusters, candidate_regions):
            if overlaps[ci, ri] > 0:
                paired[int(ri)] = int(ci)
    details = []
    for ri in range(count):
        fraction = retained_fractions[ri] if retained_fractions else 1.0
        row = {"region": ri, "retained_fraction": fraction, "localized": ri in paired,
               "raw_hamming": None, "raw_exact": False, "useful_exact": False,
               "silent_wrong_decode": False, "bch_status": "missing", "corrected_total_bits": None,
               "corrected_bch_bits": None, "corrected_parity_bit": None,
               "cluster_label": None, "confidence": None, "overlap_pixels_256": 0,
               "ecc_decode_seconds": None}
        if ri in paired:
            cluster = clusters[paired[ri]]
            truth_word = bits_to_int(messages[ri].int().tolist())
            hamming = (cluster["raw_word"] ^ truth_word).bit_count()
            decoded = cluster["decoded"]
            row.update({
                "raw_hamming": hamming, "raw_exact": hamming == 0,
                "useful_exact": decoded["status"] == "valid" and decoded["payload"] == useful[ri],
                "silent_wrong_decode": decoded["status"] == "valid" and decoded["payload"] != useful[ri],
                "bch_status": decoded["status"], "corrected_total_bits": decoded["corrected_total_bits"],
                "corrected_bch_bits": decoded["corrected_bch_bits"],
                "corrected_parity_bit": decoded["corrected_parity_bit"],
                "cluster_label": cluster["label"], "confidence": cluster["confidence"],
                "overlap_pixels_256": int(overlaps[paired[ri], ri]),
                "ecc_decode_seconds": cluster["ecc_decode_seconds"],
            })
        details.append(row)
    union_gt = ground.any(dim=0)
    union_pred = detected["mask_probability"][0] > 0.5
    union = int((union_gt | union_pred).sum())
    intersection = int((union_gt & union_pred).sum())
    metrics = {"detected_clusters": len(clusters), "matched_clusters": len(paired),
               "unmatched_clusters": len(clusters) - len(paired),
               "union_iou": intersection / union if union else 1.0,
               "clustering_error": detected["clustering_error"]}
    return details, metrics


def quality(original: torch.Tensor, marked: torch.Tensor) -> tuple[float, float]:
    a, b = previous.tensor_rgb(original), previous.tensor_rgb(marked)
    return (float(peak_signal_noise_ratio(a, b, data_range=1.0)),
            float(structural_similarity(a, b, channel_axis=2, data_range=1.0)))


def load_image(record: dict) -> torch.Tensor:
    return default_transform(Image.open(record["processed"]).convert("RGB")).unsqueeze(0).to(DEVICE)


def run_clean(wam, manifest: list[dict]):
    rows = []
    packets = []
    masks = previous.make_masks(4, DEVICE)
    for image_index, record in enumerate(manifest):
        image = load_image(record)
        for payload_seed in range(3):
            useful = useful_for(image_index, payload_seed, 4)
            start_ecc = time.perf_counter()
            messages = protected_messages(useful)
            ecc_encode_seconds = time.perf_counter() - start_ecc
            marked, embed_seconds = previous.embed_messages(wam, image, messages, masks)
            detection = detect_blind(wam, marked)
            details, metrics = score_regions(detection, useful, masks, messages)
            psnr, ssim = quality(image, marked)
            row = {
                "image": record["name"], "image_index": image_index, "payload_seed": payload_seed,
                "regions": 4, "raw_exact_packets": sum(item["raw_exact"] for item in details),
                "useful_exact_packets": sum(item["useful_exact"] for item in details),
                "silent_wrong_decodes": sum(item["silent_wrong_decode"] for item in details),
                "raw_bit_errors_matched": sum(item["raw_hamming"] for item in details if item["raw_hamming"] is not None),
                "matched_packets": sum(item["localized"] for item in details),
                "psnr": psnr, "ssim": ssim,
                "embed_seconds": embed_seconds, "detect_seconds": detection["runtime_seconds"],
                "ecc_encode_seconds": ecc_encode_seconds,
                "ecc_decode_seconds": sum(cluster["ecc_decode_seconds"] for cluster in detection["clusters"]),
                **metrics,
            }
            rows.append(row)
            for item in details:
                packets.append({"image": record["name"], "image_index": image_index,
                                "payload_seed": payload_seed, **item})
            log("clean_case", image=record["name"], seed=payload_seed,
                raw_exact=row["raw_exact_packets"], protected_exact=row["useful_exact_packets"],
                clusters=row["detected_clusters"])
            if image_index == 0 and payload_seed == 0:
                viz_detect = {"labels": detection["labels"], "mask_prob": detection["mask_probability"]}
                previous.save_visualization(OUT / "clean_four_regions.png", image, marked, masks,
                                            viz_detect, "Protected BCH(31,16)+parity: 4 localized WAM messages")
            save_csv("clean_cases.csv", rows)
            save_csv("clean_packets.csv", packets)
    old_rows = list(csv.DictReader((OLD / "official_results" / "clean_cases.csv").open()))
    baseline = {(row["image"], int(row["payload_seed"])): row for row in old_rows if int(row["region_count"]) == 4}
    assert len(baseline) == 30
    paired = []
    for row in rows:
        old = baseline[(row["image"], row["payload_seed"])]
        paired.append({"image": row["image"], "payload_seed": row["payload_seed"],
                       "prior_raw_exact_packets": int(old["exact_packets"]),
                       "new_raw_exact_packets": row["raw_exact_packets"],
                       "new_protected_exact_packets": row["useful_exact_packets"],
                       "prior_psnr": float(old["psnr"]), "new_psnr": row["psnr"],
                       "prior_ssim": float(old["ssim"]), "new_ssim": row["ssim"]})
    save_csv("paired_comparison.csv", paired)
    protected_exact = sum(row["useful_exact_packets"] for row in rows)
    raw_exact = sum(row["raw_exact_packets"] for row in rows)
    silent = sum(row["silent_wrong_decodes"] for row in rows)
    matched = sum(row["matched_packets"] for row in rows)
    raw_errors = sum(row["raw_bit_errors_matched"] for row in rows)
    prior_psnr = np.mean([float(row["prior_psnr"]) for row in paired])
    prior_ssim = np.mean([float(row["prior_ssim"]) for row in paired])
    new_psnr = float(np.mean([row["psnr"] for row in rows]))
    new_ssim = float(np.mean([row["ssim"] for row in rows]))
    corrections = Counter(str(item["corrected_total_bits"]) if item["bch_status"] == "valid" else item["bch_status"] for item in packets)
    summary = {
        "cases": len(rows), "packets": len(packets), "detected_and_spatially_matched": matched,
        "raw_exact_packets": raw_exact, "raw_exact_rate": raw_exact / len(packets),
        "raw_ber_among_matched": raw_errors / (32 * matched) if matched else None,
        "raw_ber_count_missing_as_32_errors": (raw_errors + 32 * (len(packets) - matched)) / (32 * len(packets)),
        "useful_exact_packets": protected_exact, "useful_exact_rate": protected_exact / len(packets),
        "all_four_useful_exact_images": sum(row["useful_exact_packets"] == 4 for row in rows),
        "all_four_useful_exact_rate": np.mean([row["useful_exact_packets"] == 4 for row in rows]),
        "silent_wrong_decodes": silent, "silent_wrong_rate": silent / len(packets),
        "codec_status_and_correction_distribution": dict(corrections),
        "mean_detected_clusters": float(np.mean([row["detected_clusters"] for row in rows])),
        "mean_union_iou": float(np.mean([row["union_iou"] for row in rows])),
        "prior_mean_psnr": float(prior_psnr), "new_mean_psnr": new_psnr,
        "psnr_change_db": new_psnr - prior_psnr,
        "prior_mean_ssim": float(prior_ssim), "new_mean_ssim": new_ssim,
        "ssim_change": new_ssim - prior_ssim,
        "mean_embed_seconds": float(np.mean([row["embed_seconds"] for row in rows])),
        "mean_detect_seconds": float(np.mean([row["detect_seconds"] for row in rows])),
        "mean_ecc_encode_seconds": float(np.mean([row["ecc_encode_seconds"] for row in rows])),
        "mean_ecc_decode_seconds": float(np.mean([row["ecc_decode_seconds"] for row in rows])),
    }
    gate = {
        "protected_packet_recovery_ge_98pct": bool(summary["useful_exact_rate"] >= 0.98),
        "all_four_images_ge_90pct": bool(summary["all_four_useful_exact_rate"] >= 0.90),
        "silent_wrong_le_1pct": bool(summary["silent_wrong_rate"] <= 0.01),
        "psnr_drop_le_0p5_db": bool(summary["psnr_change_db"] >= -0.5),
        "ssim_drop_le_0p005": bool(summary["ssim_change"] >= -0.005),
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    save_json("clean_summary.json", summary)
    log("clean_complete", gate=gate, protected_exact=protected_exact, total=len(packets))
    return summary


def main():
    start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    manifest = verify_environment()
    assert (OUT / "codec_verification.json").exists(), "run codec verification first"
    os.chdir(UPSTREAM)
    wam = load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth").to(DEVICE).eval()
    log("model_loaded", parameters=sum(p.numel() for p in wam.parameters()), device=str(DEVICE))
    clean = run_clean(wam, manifest)
    result = {"phase2": clean, "phase3": "NOT RUN", "phase4": "NOT RUN", "phase5": "NOT RUN",
              "last_completed_phase": "Phase 2 clean protected four-region",
              "decision": "MODIFY" if not clean["gate"]["pass"] else "PENDING",
              "stop_reason": "Phase 2 gate failed" if not clean["gate"]["pass"] else None,
              "elapsed_seconds": time.time() - start}
    save_json("final_result.json", result)
    log("phase2_decision", decision=result["decision"], gate_pass=clean["gate"]["pass"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
