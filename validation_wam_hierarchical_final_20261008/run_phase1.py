"""Official-WAM four-region CRC/BCH clean and blind-crop validation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "validation_wam_20261007"
INNER = ROOT.parent / "validation_wam_inner_ecc_20261008"
UPSTREAM = OLD / "watermark-anything"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, str(INNER))
sys.path.insert(0, str(UPSTREAM))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

import run_sprint as prior
from run_crop import fixed_crops
from notebooks.inference_utils import load_model_from_checkpoint
from packet_codec import decode_packet, encode_bits, encode_packet, packet16

SEED = 20261008
DEVICE = torch.device("cpu")
EXPECTED_COMMIT = "2c08af04d037d5667c02f6ddebbda9ff04581c3e"
EXPECTED_CHECKPOINT = "90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(name: str, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def write_csv(name: str, rows: list[dict]):
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def log(kind: str, **values):
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, **values}
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def verify_environment():
    checkpoint = UPSTREAM / "checkpoints" / "wam_mit.pth"
    manifest_path = OLD / "official_results" / "image_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest:
        for key in ("source", "processed"):
            assert hash_file(Path(item[key])) == item[f"{key}_sha256"]
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_executable": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__, "numpy": np.__version__,
        "pillow": Image.__version__, "scikit_image": __import__("skimage").__version__,
        "device": str(DEVICE), "mps_available": bool(torch.backends.mps.is_available()),
        "wam_commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "wam_git_status": subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--short"], text=True).strip(),
        "project_git_status": subprocess.check_output(["git", "-C", str(ROOT.parent), "status", "--short"], text=True).strip(),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": hash_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "reused_image_manifest_sha256": hash_file(manifest_path),
        "previous_clean_packets_sha256": hash_file(INNER / "results" / "clean_packets.csv"),
        "previous_crop_packets_sha256": hash_file(INNER / "results" / "crop_packets.csv"),
    }
    assert record["wam_commit"] == EXPECTED_COMMIT
    assert not record["wam_git_status"]
    assert record["checkpoint_sha256"] == EXPECTED_CHECKPOINT
    assert len(manifest) == 10
    write_json("environment.json", record)
    write_json("reused_images.json", manifest)
    log("environment_verified", commit=record["wam_commit"], checkpoint=record["checkpoint_sha256"])
    return manifest


def symbols_for(image_index: int, payload_seed: int, count: int) -> list[int]:
    rng = np.random.default_rng(SEED + image_index*1009 + payload_seed*97 + count)
    return [int(x) for x in rng.integers(0, 256, count, dtype=np.uint8)]


def packet_messages(symbols: list[int]) -> torch.Tensor:
    return torch.tensor([encode_bits(index, value) for index, value in enumerate(symbols)], dtype=torch.float32)


def infer_packets(wam, marked: torch.Tensor) -> dict:
    detection = prior.detect_blind(wam, marked)
    for cluster in detection["clusters"]:
        start = time.perf_counter()
        cluster["packet"] = decode_packet(cluster["raw_word"])
        cluster["packet_decode_seconds"] = time.perf_counter()-start
    return detection


def score(detection: dict, symbols: list[int], messages: torch.Tensor, oracle_masks: torch.Tensor,
          physical_fractions: list[float]) -> tuple[list[dict], dict]:
    """Oracle overlap is exclusively for evaluation, after blind inference."""
    count = len(symbols)
    clusters = detection["clusters"]
    ground = F.interpolate(oracle_masks.detach().cpu(), size=detection["labels"].shape[-2:], mode="nearest")[:, 0] > .5
    overlap = np.zeros((len(clusters), count), dtype=np.int32)
    for ci, cluster in enumerate(clusters):
        for region in range(count):
            overlap[ci, region] = int((cluster["mask"] & ground[region]).sum())
    assigned = {}
    if len(clusters):
        rows, columns = linear_sum_assignment(-overlap)
        for ci, region in zip(rows, columns):
            if overlap[ci, region] > 0:
                assigned[int(region)] = int(ci)
    region_rows = []
    for region in range(count):
        item = {"region": region, "physical_retention": physical_fractions[region],
                "localized": region in assigned, "raw_hamming": None,
                "raw_exact": False, "bch_exact_16": False,
                "bch_status": "missing", "bch_corrected_bits": None,
                "crc_status": "missing", "accepted_packet_id": None,
                "accepted_symbol": None, "classification": "ERASURE",
                "confidence": None, "cluster_label": None, "overlap_pixels_256": 0,
                "packet_decode_seconds": 0.0}
        if region in assigned:
            cluster = clusters[assigned[region]]
            packet = cluster["packet"]
            expected = encode_packet(region, symbols[region])
            hamming = (cluster["raw_word"] ^ expected).bit_count()
            bch = cluster["decoded"]
            bch_exact = bch["status"] == "valid" and bch["payload"] == packet16(region, symbols[region])
            classification = "ERASURE"
            if packet["status"] == "ACCEPTED":
                classification = "VALID" if (packet["packet_id"], packet["symbol"]) == (region, symbols[region]) else "SILENT WRONG"
            item.update({
                "raw_hamming": hamming, "raw_exact": hamming == 0,
                "bch_exact_16": bch_exact, "bch_status": bch["status"],
                "bch_corrected_bits": bch["corrected_total_bits"],
                "crc_status": packet["reason"] or "accepted",
                "accepted_packet_id": packet["packet_id"], "accepted_symbol": packet["symbol"],
                "classification": classification, "confidence": cluster["confidence"],
                "cluster_label": cluster["label"], "overlap_pixels_256": int(overlap[assigned[region], region]),
                "packet_decode_seconds": cluster["packet_decode_seconds"],
            })
        region_rows.append(item)
    assigned_cluster_indices = set(assigned.values())
    unmatched = [cluster for ci, cluster in enumerate(clusters) if ci not in assigned_cluster_indices]
    metrics = {"detected_clusters": len(clusters), "matched_clusters": len(assigned),
               "unmatched_clusters": len(unmatched),
               "unmatched_accepted_clusters": sum(cluster["packet"]["status"] == "ACCEPTED" for cluster in unmatched),
               "clustering_error": detection["clustering_error"],
               "packet_decode_seconds": sum(cluster["packet_decode_seconds"] for cluster in clusters)}
    return region_rows, metrics


def run_phase1(wam, manifest):
    clean_rows, clean_packets, crop_rows, crop_packets = [], [], [], []
    masks = prior.previous.make_masks(4, DEVICE)
    for image_index, image_record in enumerate(manifest):
        image = prior.load_image(image_record)
        for payload_seed in range(3):
            symbols = symbols_for(image_index, payload_seed, 4)
            start = time.perf_counter()
            messages = packet_messages(symbols)
            codec_encode_seconds = time.perf_counter()-start
            marked, embed_seconds = prior.previous.embed_messages(wam, image, messages, masks)
            detection = infer_packets(wam, marked)
            details, metrics = score(detection, symbols, messages, masks, [1.0]*4)
            psnr, ssim = prior.quality(image, marked)
            clean_row = {"image": image_record["name"], "image_index": image_index,
                         "payload_seed": payload_seed, "psnr": psnr, "ssim": ssim,
                         "embed_seconds": embed_seconds, "detect_seconds": detection["runtime_seconds"],
                         "codec_encode_seconds": codec_encode_seconds,
                         "raw_exact": sum(x["raw_exact"] for x in details),
                         "bch_exact": sum(x["bch_exact_16"] for x in details),
                         "valid_exact": sum(x["classification"] == "VALID" for x in details),
                         "erased": sum(x["classification"] == "ERASURE" for x in details),
                         "silent_wrong": sum(x["classification"] == "SILENT WRONG" for x in details),
                         **metrics}
            clean_rows.append(clean_row)
            for item in details:
                clean_packets.append({"image": image_record["name"], "image_index": image_index,
                                      "payload_seed": payload_seed, **item})
            for retained, variant, ratio, x0, y0, width, height in fixed_crops(image_index, payload_seed):
                cropped = marked[:, :, y0:y0+height, x0:x0+width]
                detection = infer_packets(wam, cropped)  # receives crop pixels only
                oracle_masks = masks[:, :, y0:y0+height, x0:x0+width]
                fractions = [float(oracle_masks[i].sum()/masks[i].sum()) for i in range(4)]
                details, metrics = score(detection, symbols, messages, oracle_masks, fractions)
                crop_row = {"image": image_record["name"], "image_index": image_index,
                            "payload_seed": payload_seed, "retained_target": retained,
                            "retained_actual": width*height/512**2,
                            "variant": variant, "aspect_ratio": ratio, "x0": x0, "y0": y0,
                            "width": width, "height": height,
                            "raw_exact": sum(x["raw_exact"] for x in details),
                            "bch_exact": sum(x["bch_exact_16"] for x in details),
                            "valid_exact": sum(x["classification"] == "VALID" for x in details),
                            "erased": sum(x["classification"] == "ERASURE" for x in details),
                            "silent_wrong": sum(x["classification"] == "SILENT WRONG" for x in details),
                            "large_surviving": sum(x["physical_retention"] >= .75 for x in details),
                            "large_valid_exact": sum(x["physical_retention"] >= .75 and x["classification"] == "VALID" for x in details),
                            "detect_seconds": detection["runtime_seconds"], **metrics}
                crop_rows.append(crop_row)
                for item in details:
                    crop_packets.append({"image": image_record["name"], "image_index": image_index,
                                         "payload_seed": payload_seed, "retained_target": retained,
                                         "variant": variant, "aspect_ratio": ratio,
                                         "x0": x0, "y0": y0, "width": width, "height": height,
                                         **item})
                log("crop_case", image=image_record["name"], seed=payload_seed,
                    area=retained, variant=variant, valid=crop_row["valid_exact"],
                    erased=crop_row["erased"], silent=crop_row["silent_wrong"])
            log("clean_case", image=image_record["name"], seed=payload_seed,
                valid=clean_row["valid_exact"], silent=clean_row["silent_wrong"])
            write_csv("clean_cases.csv", clean_rows)
            write_csv("clean_packets.csv", clean_packets)
            write_csv("crop_cases.csv", crop_rows)
            write_csv("crop_packets.csv", crop_packets)
    old_quality = json.loads((INNER / "results" / "clean_summary.json").read_text())
    all_crop = crop_packets
    large = [row for row in all_crop if row["physical_retention"] >= .75]
    failures = [row for row in all_crop if row["classification"] != "VALID"]
    silent = sum(row["classification"] == "SILENT WRONG" for row in all_crop)
    erasures = sum(row["classification"] == "ERASURE" for row in all_crop)
    accepted_unmatched = sum(row["unmatched_accepted_clusters"] for row in crop_rows)
    mean_psnr = float(np.mean([row["psnr"] for row in clean_rows]))
    mean_ssim = float(np.mean([row["ssim"] for row in clean_rows]))
    summary = {
        "clean": {"cases": len(clean_rows), "region_cases": len(clean_packets),
                  "raw_exact": sum(row["raw_exact"] for row in clean_rows),
                  "bch_exact": sum(row["bch_exact"] for row in clean_rows),
                  "valid_exact": sum(row["valid_exact"] for row in clean_rows),
                  "erased": sum(row["erased"] for row in clean_rows),
                  "silent_wrong": sum(row["silent_wrong"] for row in clean_rows),
                  "mean_psnr": mean_psnr, "mean_ssim": mean_ssim,
                  "previous_protected_mean_psnr": old_quality["new_mean_psnr"],
                  "previous_protected_mean_ssim": old_quality["new_mean_ssim"],
                  "psnr_change_db": mean_psnr-old_quality["new_mean_psnr"],
                  "ssim_change": mean_ssim-old_quality["new_mean_ssim"],
                  "mean_embed_seconds": float(np.mean([row["embed_seconds"] for row in clean_rows])),
                  "mean_detect_seconds": float(np.mean([row["detect_seconds"] for row in clean_rows])),
                  "mean_codec_encode_seconds": float(np.mean([row["codec_encode_seconds"] for row in clean_rows])),
                  "mean_packet_decode_seconds": float(np.mean([row["packet_decode_seconds"] for row in clean_rows])),
                 },
        "crop": {"cases": len(crop_rows), "region_cases": len(all_crop),
                 "raw_exact": sum(row["raw_exact"] for row in crop_rows),
                 "bch_exact": sum(row["bch_exact"] for row in crop_rows),
                 "valid_exact": sum(row["valid_exact"] for row in crop_rows),
                 "erased": erasures, "silent_wrong": silent,
                 "silent_wrong_rate": silent/len(all_crop),
                 "unmatched_accepted_clusters": accepted_unmatched,
                 "large_surviving": len(large),
                 "large_valid_exact": sum(row["classification"] == "VALID" for row in large),
                 "large_valid_exact_rate": sum(row["classification"] == "VALID" for row in large)/len(large),
                 "failed_packets": len(failures),
                 "erasure_share_of_failures": erasures/len(failures) if failures else 1.0,
                 "by_area": [],
                 "classification_counts": dict(Counter(row["classification"] for row in all_crop)),
                 "crc_status_counts": dict(Counter(row["crc_status"] for row in all_crop)),
                 "bch_status_counts": dict(Counter(row["bch_status"] for row in all_crop)),
                },
    }
    for area in (.75, .5):
        rows = [row for row in crop_rows if row["retained_target"] == area]
        packs = [row for row in all_crop if row["retained_target"] == area]
        largely = [row for row in packs if row["physical_retention"] >= .75]
        summary["crop"]["by_area"].append({
            "retained_target": area, "crops": len(rows), "region_cases": len(packs),
            "raw_exact": sum(row["raw_exact"] for row in packs),
            "bch_exact": sum(row["bch_exact_16"] for row in packs),
            "valid_exact": sum(row["classification"] == "VALID" for row in packs),
            "erased": sum(row["classification"] == "ERASURE" for row in packs),
            "silent_wrong": sum(row["classification"] == "SILENT WRONG" for row in packs),
            "large_surviving": len(largely),
            "large_valid_exact": sum(row["classification"] == "VALID" for row in largely),
            "large_valid_exact_rate": sum(row["classification"] == "VALID" for row in largely)/len(largely) if largely else None,
            "mean_clusters": float(np.mean([row["detected_clusters"] for row in rows])),
            "mean_detect_seconds": float(np.mean([row["detect_seconds"] for row in rows])),
        })
    gate = {
        "silent_wrong_le_0p5pct": summary["crop"]["silent_wrong_rate"] <= .005,
        "large_region_recovery_ge_95pct": summary["crop"]["large_valid_exact_rate"] >= .95,
        "erasure_share_of_failures_ge_95pct": summary["crop"]["erasure_share_of_failures"] >= .95,
        "psnr_drop_le_0p5db": summary["clean"]["psnr_change_db"] >= -.5,
        "ssim_drop_le_0p005": summary["clean"]["ssim_change"] >= -.005,
        "no_accepted_unmatched_clusters": accepted_unmatched == 0,
    }
    gate["pass"] = all(gate.values())
    summary["gate"] = gate
    write_json("phase1_summary.json", summary)
    log("phase1_complete", gate=gate, silent=silent, region_cases=len(all_crop))
    return summary


def main():
    start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    assert (OUT / "packet_codec_verification.json").exists()
    manifest = verify_environment()
    os.chdir(UPSTREAM)
    wam = load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth").to(DEVICE).eval()
    log("model_loaded", parameters=sum(parameter.numel() for parameter in wam.parameters()))
    summary = run_phase1(wam, manifest)
    final = {"phase1": summary, "phase2": "NOT RUN", "phase3": "NOT RUN",
             "phase4": "NOT RUN", "phase5": "NOT RUN",
             "last_completed_phase": "four-region CRC/BCH blind crop",
             "decision": "PENDING" if summary["gate"]["pass"] else "MODIFY",
             "stop_reason": None if summary["gate"]["pass"] else "Phase 1 integrity gate failed",
             "phase1_elapsed_seconds": time.time()-start}
    write_json("final_result.json", final)
    log("phase1_decision", decision=final["decision"], gate_pass=summary["gate"]["pass"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
