"""Frozen official-WAM robustness study; no model or old-result modifications."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OLD = PROJECT / "validation_wam_20261007"
PREV = PROJECT / "validation_wam_hierarchical_final_20261008"
UPSTREAM = OLD / "watermark-anything"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
for folder in (RESULTS, FIGURES, LOGS):
    folder.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, str(PREV))
sys.path.insert(0, str(OLD))

import numpy as np
import torch
from PIL import Image
import run_phase1 as p
from run_final_comparison import source_for
from outer_codes import collect_accepted_packets, decode_repetition, encode_repetition
from run_validation import jpeg_attack, noise_attack, rotate_attack, tensor_rgb

EXPECTED_COMMIT = "2c08af04d037d5667c02f6ddebbda9ff04581c3e"
EXPECTED_CHECKPOINT = "90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2"
ANGLES = (-10, -5, -2, -1, 1, 2, 5, 10)
SEARCH_ANGLES = (-10, -5, -2, -1, 0, 1, 2, 5, 10)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def event(kind, **fields):
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, **fields}
    with (LOGS / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def environment():
    manifest_path = OLD / "official_results" / "image_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 10
    for record in manifest:
        for kind in ("source", "processed"):
            assert sha256(record[kind]) == record[f"{kind}_sha256"]
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    checkpoint = UPSTREAM / "checkpoints" / "wam_mit.pth"
    checkpoint_hash = sha256(checkpoint)
    assert commit == EXPECTED_COMMIT
    assert checkpoint_hash == EXPECTED_CHECKPOINT
    assert not subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--short"], text=True).strip()
    record = {
        "python_executable": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__, "numpy": np.__version__,
        "pillow": Image.__version__, "scipy": __import__("scipy").__version__,
        "scikit_image": __import__("skimage").__version__,
        "wam_commit": commit, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "image_manifest": str(manifest_path), "image_manifest_sha256": sha256(manifest_path),
        "input_images": [{"name": x["name"], "processed_sha256": x["processed_sha256"],
                          "source_sha256": x["source_sha256"]} for x in manifest],
        "prior_packet_codec_sha256": sha256(PREV / "packet_codec.py"),
        "prior_outer_codes_sha256": sha256(PREV / "outer_codes.py"),
        "git_status_project": subprocess.check_output(["git", "-C", str(PROJECT), "status", "--short"], text=True).strip(),
    }
    save_json(RESULTS / "environment.json", record)
    event("environment_verified", commit=commit, checkpoint_sha256=checkpoint_hash)
    return manifest


def decode_global(detection):
    start = time.perf_counter()
    packets = [cluster["packet"] for cluster in detection["clusters"]]
    unique, counts = collect_accepted_packets(packets)
    decoded = decode_repetition(unique)
    return {**counts, "global_status": decoded["status"],
            "global_payload": decoded["payload"],
            "outer_decode_seconds": time.perf_counter() - start}


def oracle_masks_rotated(masks, angle):
    arrays = []
    for i in range(len(masks)):
        image = Image.fromarray((masks[i, 0].cpu().numpy() * 255).astype(np.uint8), mode="L")
        image = image.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
        arrays.append(torch.from_numpy((np.asarray(image).copy() > 127).astype(np.float32))[None])
    return torch.stack(arrays)


def infer_case(wam, attacked, source, symbols, messages, oracle_masks):
    detection = p.infer_packets(wam, attacked)
    decoded = decode_global(detection)  # Only attacked pixels and packet IDs enter decoding.
    region_rows, metrics = p.score(detection, list(symbols), messages, oracle_masks, [1.0] * 8)
    exact = decoded["global_status"] == "decoded" and decoded["global_payload"] == source
    wrong = decoded["global_status"] == "decoded" and not exact
    corrected = sum(row["classification"] == "VALID" and (row["bch_corrected_bits"] or 0) > 0
                    for row in region_rows)
    matched = sum(row["localized"] for row in region_rows)
    bit_errors = sum(row["raw_hamming"] or 0 for row in region_rows)
    row = {"global_exact": bool(exact), "wrong_global_output": bool(wrong),
           "global_status": decoded["global_status"],
           "detected_clusters": metrics["detected_clusters"],
           "matched_packets": matched, "raw_bit_errors_matched": bit_errors,
           "raw_ber_matched": bit_errors / (32 * matched) if matched else None,
           "raw_exact_packets": sum(row["raw_exact"] for row in region_rows),
           "bch_exact_packets": sum(row["bch_exact_16"] for row in region_rows),
           "bch_corrected_exact_packets": corrected,
           "exact_crc_valid_packets": sum(row["classification"] == "VALID" for row in region_rows),
           "erased_packets": sum(row["classification"] == "ERASURE" for row in region_rows),
           "silent_wrong_packets": sum(row["classification"] == "SILENT WRONG" for row in region_rows),
           "detect_seconds": detection["runtime_seconds"],
           "packet_decode_seconds": metrics["packet_decode_seconds"],
           "unmatched_accepted_clusters": metrics["unmatched_accepted_clusters"],
           **{key: decoded[key] for key in ("accepted_clusters", "unique_valid_ids", "conflicting_ids",
                                            "erased_id_count", "outer_decode_seconds")}}
    return row, region_rows


def save_example(original, marked, attacks):
    import matplotlib.pyplot as plt
    items = [("Original", original), ("Marked", marked)] + list(attacks)
    fig, axes = plt.subplots(1, len(items), figsize=(3.1 * len(items), 3.3))
    for axis, (label, image) in zip(axes, items):
        axis.imshow(tensor_rgb(image))
        axis.set_title(label)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "representative_attacks.png", dpi=160)
    plt.close(fig)


def run_baseline(wam, manifest):
    cases, packet_rows, negatives = [], [], []
    masks = p.prior.previous.make_masks(8, p.DEVICE)
    for image_index, record in enumerate(manifest):
        original = p.prior.load_image(record)
        # Negative control: full blind decoder on unmarked natural image.
        unmarked = p.infer_packets(wam, original)
        neg_decoded = decode_global(unmarked)
        negatives.append({"image": record["name"], "detected_clusters": len(unmarked["clusters"]),
                          "crc_valid_clusters": neg_decoded["accepted_clusters"],
                          "unique_valid_ids": neg_decoded["unique_valid_ids"],
                          "global_output": neg_decoded["global_status"] == "decoded",
                          "detect_seconds": unmarked["runtime_seconds"]})
        save_csv(RESULTS / "negative_control.csv", negatives)
        for seed in range(3):
            source = source_for(image_index, seed)
            symbols = encode_repetition(source)
            start_encode = time.perf_counter()
            messages = p.packet_messages(list(symbols))
            encode_seconds = time.perf_counter() - start_encode
            marked, embed_seconds = p.prior.previous.embed_messages(wam, original, messages, masks)
            psnr, ssim = p.prior.quality(original, marked)
            case_id = f"{image_index:02d}_{seed}"
            variants = [("clean", marked, masks, None)]
            variants += [(f"jpeg{q}", jpeg_attack(marked, q), masks, None) for q in (75, 50)]
            variants += [(f"noise{s}", noise_attack(marked, s, p.SEED + image_index * 1009 + seed * 97 + s), masks, None)
                         for s in (2, 5)]
            variants += [(f"rotate{angle:+d}", rotate_attack(marked, angle),
                          oracle_masks_rotated(masks, angle), angle) for angle in ANGLES]
            crop = next(item for item in p.fixed_crops(image_index, seed) if item[0] == .75)
            _, crop_variant, _, x0, y0, width, height = crop
            cropped = marked[:, :, y0:y0+height, x0:x0+width]
            variants.append(("crop75_jpeg75", jpeg_attack(cropped, 75),
                             masks[:, :, y0:y0+height, x0:x0+width], None))
            if image_index == 0 and seed == 0:
                save_example(original, marked, [("JPEG 50", variants[2][1]), ("Noise σ5", variants[4][1]),
                                               ("Rotate +5°", next(x[1] for x in variants if x[0] == "rotate+5")),
                                               ("Crop 75% + JPEG", variants[-1][1])])
            for condition, attacked, oracle, angle in variants:
                measured, details = infer_case(wam, attacked, source, symbols, messages, oracle)
                row = {"case_id": case_id, "image": record["name"], "image_index": image_index,
                       "payload_seed": seed, "condition": condition, "attack_angle": angle,
                       "psnr_clean": psnr, "ssim_clean": ssim, "embed_seconds": embed_seconds,
                       "codec_encode_seconds": encode_seconds, **measured}
                cases.append(row)
                for detail in details:
                    packet_rows.append({"case_id": case_id, "condition": condition, **detail})
                event("case", id=case_id, condition=condition, exact=row["global_exact"],
                      valid=row["unique_valid_ids"], silent=row["silent_wrong_packets"])
            save_csv(RESULTS / "cases.csv", cases)
            save_csv(RESULTS / "packets.csv", packet_rows)
    event("baseline_complete", cases=len(cases), negatives=len(negatives))
    return cases


def needs_search(cases):
    values = {}
    for magnitude in (1, 2):
        group = [row for row in cases if row["condition"] in (f"rotate+{magnitude}", f"rotate-{magnitude}")]
        assert len(group) == 60
        values[magnitude] = sum(row["global_exact"] for row in group) / len(group)
    pooled = (values[1] + values[2]) / 2
    decision = pooled < .9 or any(value < .8 for value in values.values())
    save_json(RESULTS / "search_gate.json", {"rate_by_magnitude": values, "pooled": pooled,
                                               "search_triggered": decision})
    return decision


def search_score(result, candidate_angle, order):
    # Predeclared, decoder-visible only. Truth and attack angle are absent.
    return (int(result["global_status"] == "decoded"), result["unique_valid_ids"],
            -len(result["conflicting_ids"]), -abs(candidate_angle), -order)


def run_search(wam, manifest):
    masks = p.prior.previous.make_masks(8, p.DEVICE)
    results = []
    for image_index, record in enumerate(manifest):
        original = p.prior.load_image(record)
        for seed in range(3):
            source = source_for(image_index, seed)
            symbols = encode_repetition(source)
            messages = p.packet_messages(list(symbols))
            marked, _ = p.prior.previous.embed_messages(wam, original, messages, masks)
            for attack_angle in (-2, -1, 1, 2):
                attacked = rotate_attack(marked, attack_angle)
                candidates = []
                start = time.perf_counter()
                for order, candidate_angle in enumerate(SEARCH_ANGLES):
                    corrected = rotate_attack(attacked, candidate_angle) if candidate_angle else attacked
                    detection = p.infer_packets(wam, corrected)
                    decoded = decode_global(detection)
                    candidates.append({"candidate_angle": candidate_angle, "order": order,
                                       "detect_seconds": detection["runtime_seconds"], **decoded})
                selected = max(candidates, key=lambda x: search_score(x, x["candidate_angle"], x["order"]))
                exact = selected["global_status"] == "decoded" and selected["global_payload"] == source
                wrong = selected["global_status"] == "decoded" and not exact
                results.append({"image": record["name"], "image_index": image_index, "payload_seed": seed,
                                "attack_angle": attack_angle, "selected_inverse_angle": selected["candidate_angle"],
                                "selected_global_exact": exact, "wrong_global_output": wrong,
                                "selected_status": selected["global_status"],
                                "selected_valid_ids": selected["unique_valid_ids"],
                                "search_wall_seconds": time.perf_counter() - start,
                                "sum_detection_seconds": sum(item["detect_seconds"] for item in candidates),
                                "candidate_scores": json.dumps([{
                                    "angle": c["candidate_angle"], "status": c["global_status"],
                                    "valid_ids": c["unique_valid_ids"], "conflicts": len(c["conflicting_ids"]),
                                    "score": search_score(c, c["candidate_angle"], c["order"])}
                                    for c in candidates])})
                save_csv(RESULTS / "rotation_search.csv", results)
                event("search_case", image=record["name"], seed=seed, attack=attack_angle,
                      selected=selected["candidate_angle"], exact=exact)
    event("search_complete", cases=len(results))


def main():
    torch.set_num_threads(4)
    torch.manual_seed(p.SEED)
    np.random.seed(p.SEED)
    manifest = environment()
    os.chdir(UPSTREAM)
    wam = p.load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth").to(p.DEVICE).eval()
    event("model_loaded", parameters=sum(x.numel() for x in wam.parameters()))
    if (RESULTS / "cases.csv").exists():
        with (RESULTS / "cases.csv").open() as stream:
            cases = list(csv.DictReader(stream))
        if len(cases) != 420:
            raise RuntimeError("Incomplete baseline exists; inspect before rerun")
        for row in cases:
            row["global_exact"] = row["global_exact"] == "True"
    else:
        cases = run_baseline(wam, manifest)
    if needs_search(cases):
        if not (RESULTS / "rotation_search.csv").exists():
            run_search(wam, manifest)
    event("experiment_complete")


if __name__ == "__main__":
    main()
