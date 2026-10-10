"""Phase 3 only: blind arbitrary-offset crop localization and protected decoding."""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter

import numpy as np
import torch

import run_sprint as s


def crop_dimensions(area: float, ratio: float) -> tuple[int, int]:
    width = min(s.IMAGE_SIZE, max(32, round(math.sqrt(area * s.IMAGE_SIZE**2 * ratio))))
    height = min(s.IMAGE_SIZE, max(32, round(area * s.IMAGE_SIZE**2 / width)))
    return width, height


def fixed_crops(image_index: int, payload_seed: int):
    for retained in (0.75, 0.50):
        for variant, ratio in enumerate((1.0, 4/3, 3/4)):
            width, height = crop_dimensions(retained, ratio)
            rng = np.random.default_rng(s.SEED + 50000 + image_index*1009 + payload_seed*97 + variant*13 + int(retained*100))
            x0 = int(rng.integers(0, s.IMAGE_SIZE - width + 1))
            y0 = int(rng.integers(0, s.IMAGE_SIZE - height + 1))
            yield retained, variant, ratio, x0, y0, width, height


def run_crop(wam, manifest):
    rows, packets = [], []
    masks = s.previous.make_masks(4, s.DEVICE)
    for image_index, record in enumerate(manifest):
        image = s.load_image(record)
        for payload_seed in range(3):
            useful = s.useful_for(image_index, payload_seed, 4)
            messages = s.protected_messages(useful)
            marked, _ = s.previous.embed_messages(wam, image, messages, masks)
            for retained, variant, ratio, x0, y0, width, height in fixed_crops(image_index, payload_seed):
                crop = marked[:, :, y0:y0+height, x0:x0+width]
                # Oracle masks are created AFTER blind WAM/BCH inference and are used only for scoring.
                detection = s.detect_blind(wam, crop)
                scoring_masks = masks[:, :, y0:y0+height, x0:x0+width]
                fractions = [float(scoring_masks[i].sum() / masks[i].sum()) for i in range(4)]
                details, metrics = s.score_regions(detection, useful, scoring_masks, messages, fractions)
                row = {
                    "image": record["name"], "image_index": image_index, "payload_seed": payload_seed,
                    "retained_target": retained, "retained_actual": width*height/s.IMAGE_SIZE**2,
                    "variant": variant, "aspect_ratio": ratio, "x0": x0, "y0": y0,
                    "width": width, "height": height,
                    "localized_regions": sum(item["localized"] for item in details),
                    "raw_exact_packets": sum(item["raw_exact"] for item in details),
                    "useful_exact_packets": sum(item["useful_exact"] for item in details),
                    "silent_wrong_decodes": sum(item["silent_wrong_decode"] for item in details),
                    "largely_surviving_regions": sum(item["retained_fraction"] >= .75 for item in details),
                    "largely_surviving_exact": sum(item["retained_fraction"] >= .75 and item["useful_exact"] for item in details),
                    "detect_seconds": detection["runtime_seconds"],
                    "ecc_decode_seconds": sum(cluster["ecc_decode_seconds"] for cluster in detection["clusters"]),
                    **metrics,
                }
                rows.append(row)
                for item in details:
                    packets.append({"image": record["name"], "image_index": image_index,
                                    "payload_seed": payload_seed, "retained_target": retained,
                                    "variant": variant, "aspect_ratio": ratio,
                                    "x0": x0, "y0": y0, "width": width, "height": height,
                                    **item})
                s.log("crop_case", image=record["name"], seed=payload_seed,
                      retained=retained, variant=variant,
                      protected_exact=row["useful_exact_packets"],
                      largely_surviving=row["largely_surviving_regions"],
                      clusters=row["detected_clusters"])
                if image_index == 0 and payload_seed == 0 and retained == .5 and variant == 0:
                    viz_detect = {"labels": detection["labels"], "mask_prob": detection["mask_probability"]}
                    s.previous.save_visualization(s.OUT / "blind_crop_50pct.png", crop, crop,
                                                  scoring_masks, viz_detect,
                                                  "Unknown-offset 50% crop: masks shown for scoring only")
                s.save_csv("crop_cases.csv", rows)
                s.save_csv("crop_packets.csv", packets)
    by_area = []
    for area in (.75, .5):
        selected_rows = [row for row in rows if row["retained_target"] == area]
        selected_packets = [row for row in packets if row["retained_target"] == area]
        large = [row for row in selected_packets if row["retained_fraction"] >= .75]
        failed = [row for row in selected_packets if not row["useful_exact"]]
        statuses = Counter(row["bch_status"] for row in selected_packets)
        by_area.append({
            "retained_target": area, "crops": len(selected_rows), "regions": len(selected_packets),
            "crops_with_at_least_one_cluster": sum(row["detected_clusters"] >= 1 for row in selected_rows),
            "crops_with_at_least_one_cluster_rate": np.mean([row["detected_clusters"] >= 1 for row in selected_rows]),
            "mean_detected_clusters": float(np.mean([row["detected_clusters"] for row in selected_rows])),
            "mean_localized_regions": float(np.mean([row["localized_regions"] for row in selected_rows])),
            "mean_exact_useful_packets": float(np.mean([row["useful_exact_packets"] for row in selected_rows])),
            "overall_exact_useful_packets": sum(row["useful_exact"] for row in selected_packets),
            "largely_surviving_regions": len(large),
            "largely_surviving_localized": sum(row["localized"] for row in large),
            "largely_surviving_exact": sum(row["useful_exact"] for row in large),
            "largely_surviving_exact_rate": sum(row["useful_exact"] for row in large)/len(large) if large else None,
            "silent_wrong_decodes": sum(row["silent_wrong_decode"] for row in selected_packets),
            "failed_packets": len(failed),
            "failed_fully_removed": sum(row["retained_fraction"] == 0 for row in failed),
            "failed_less_than_25pct_remaining": sum(row["retained_fraction"] < .25 for row in failed),
            "failed_less_than_75pct_remaining": sum(row["retained_fraction"] < .75 for row in failed),
            "failed_largely_surviving": sum(row["retained_fraction"] >= .75 for row in failed),
            "bch_status": dict(statuses),
            "mean_detection_seconds": float(np.mean([row["detect_seconds"] for row in selected_rows])),
            "mean_ecc_decode_seconds": float(np.mean([row["ecc_decode_seconds"] for row in selected_rows])),
        })
    total_failures = sum(row["failed_packets"] for row in by_area)
    physically_erased = sum(row["failed_less_than_75pct_remaining"] for row in by_area)
    gate = {
        "large_region_exact_ge_75pct_each_level": all(row["largely_surviving_exact_rate"] is not None and row["largely_surviving_exact_rate"] >= .75 for row in by_area),
        "at_least_one_cluster_ge_80pct_each_level": all(row["crops_with_at_least_one_cluster_rate"] >= .80 for row in by_area),
        "silent_wrong_le_1pct": sum(row["silent_wrong_decodes"] for row in by_area) <= .01 * len(packets),
        "majority_failed_packets_below_75pct_survival": physically_erased > total_failures/2,
    }
    gate["pass"] = all(gate.values())
    result = {"by_area": by_area, "gate": gate,
              "physical_erasure_definition": "less than 75% of original region survives; full removal is separately reported",
              "total_failed_packets": total_failures,
              "failed_packets_below_75pct_survival": physically_erased}
    s.save_json("crop_summary.json", result)
    s.log("crop_complete", gate=gate, failed=total_failures, below_75_survival=physically_erased)
    return result


def main():
    start = time.time()
    torch.manual_seed(s.SEED)
    np.random.seed(s.SEED)
    torch.set_num_threads(4)
    clean = json.loads((s.OUT / "clean_summary.json").read_text())
    assert clean["gate"]["pass"], "Phase 2 gate must pass"
    manifest = s.verify_environment()
    os.chdir(s.UPSTREAM)
    wam = s.load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth").to(s.DEVICE).eval()
    s.log("model_loaded_for_crop", parameters=sum(p.numel() for p in wam.parameters()))
    crop = run_crop(wam, manifest)
    final = json.loads((s.OUT / "final_result.json").read_text())
    final.update({"phase3": crop, "last_completed_phase": "Phase 3 blind arbitrary-offset crop",
                  "decision": "PENDING" if crop["gate"]["pass"] else "MODIFY",
                  "stop_reason": None if crop["gate"]["pass"] else "Phase 3 blind-crop gate failed",
                  "phase3_elapsed_seconds": time.time()-start})
    s.save_json("final_result.json", final)
    s.log("phase3_decision", decision=final["decision"], gate_pass=crop["gate"]["pass"])
    print(json.dumps(crop, indent=2), flush=True)


if __name__ == "__main__":
    main()
