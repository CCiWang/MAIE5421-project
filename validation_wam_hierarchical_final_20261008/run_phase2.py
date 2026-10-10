"""Phase 2 only: eight simultaneous CRC/BCH WAM packets, no outer code."""
from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np
import torch

import run_phase1 as p


def run_eight(wam, manifest):
    rows, packets = [], []
    masks = p.prior.previous.make_masks(8, p.DEVICE)
    for image_index, record in enumerate(manifest):
        image = p.prior.load_image(record)
        for seed in range(3):
            symbols = p.symbols_for(image_index, seed, 8)
            start = time.perf_counter()
            messages = p.packet_messages(symbols)
            encode_seconds = time.perf_counter()-start
            marked, embed_seconds = p.prior.previous.embed_messages(wam, image, messages, masks)
            detection = p.infer_packets(wam, marked)
            details, metrics = p.score(detection, symbols, messages, masks, [1.0]*8)
            psnr, ssim = p.prior.quality(image, marked)
            row = {"image": record["name"], "image_index": image_index, "payload_seed": seed,
                   "raw_exact": sum(item["raw_exact"] for item in details),
                   "bch_exact": sum(item["bch_exact_16"] for item in details),
                   "valid_exact": sum(item["classification"] == "VALID" for item in details),
                   "erased": sum(item["classification"] == "ERASURE" for item in details),
                   "silent_wrong": sum(item["classification"] == "SILENT WRONG" for item in details),
                   "psnr": psnr, "ssim": ssim, "embed_seconds": embed_seconds,
                   "detect_seconds": detection["runtime_seconds"], "encode_seconds": encode_seconds,
                   **metrics}
            rows.append(row)
            for item in details:
                packets.append({"image": record["name"], "image_index": image_index,
                                "payload_seed": seed, **item})
            p.log("eight_clean_case", image=record["name"], seed=seed,
                  clusters=row["detected_clusters"], valid=row["valid_exact"],
                  erased=row["erased"], silent=row["silent_wrong"])
            if image_index == 0 and seed == 0:
                p.prior.previous.save_visualization(p.OUT / "eight_clean_visualization.png", image,
                    marked, masks, {"labels": detection["labels"],
                                    "mask_prob": detection["mask_probability"]},
                    "Eight CRC/BCH-protected WAM packets, clean image")
            p.write_csv("eight_clean_cases.csv", rows)
            p.write_csv("eight_clean_packets.csv", packets)
    distribution = {str(k): sum(row["valid_exact"] == k for row in rows) for k in range(9)}
    silent = sum(row["silent_wrong"] for row in rows)
    result = {
        "cases": len(rows), "region_cases": len(packets), "distribution_valid_exact_0_to_8": distribution,
        "images_with_at_least_4_valid": sum(row["valid_exact"] >= 4 for row in rows),
        "p_at_least_4_valid": float(np.mean([row["valid_exact"] >= 4 for row in rows])),
        "mean_valid_exact": float(np.mean([row["valid_exact"] for row in rows])),
        "raw_exact": sum(row["raw_exact"] for row in rows),
        "bch_exact": sum(row["bch_exact"] for row in rows),
        "valid_exact": sum(row["valid_exact"] for row in rows),
        "erased": sum(row["erased"] for row in rows),
        "silent_wrong": silent, "silent_wrong_rate": silent/len(packets),
        "unmatched_accepted_clusters": sum(row["unmatched_accepted_clusters"] for row in rows),
        "mean_clusters": float(np.mean([row["detected_clusters"] for row in rows])),
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
        "mean_embed_seconds": float(np.mean([row["embed_seconds"] for row in rows])),
        "mean_detect_seconds": float(np.mean([row["detect_seconds"] for row in rows])),
        "mean_packet_decode_seconds": float(np.mean([row["packet_decode_seconds"] for row in rows])),
        "bch_status_counts": dict(Counter(row["bch_status"] for row in packets)),
        "crc_status_counts": dict(Counter(row["crc_status"] for row in packets)),
    }
    gate = {"p_at_least_4_ge_99pct": result["p_at_least_4_valid"] >= .99,
            "mean_valid_ge_7": result["mean_valid_exact"] >= 7,
            "silent_wrong_le_0p5pct": result["silent_wrong_rate"] <= .005,
            "mean_psnr_ge_40db": result["mean_psnr"] >= 40,
            "mean_ssim_ge_0p97": result["mean_ssim"] >= .97,
            "no_accepted_unmatched_clusters": result["unmatched_accepted_clusters"] == 0}
    gate["pass"] = all(gate.values())
    result["gate"] = gate
    p.write_json("phase2_summary.json", result)
    p.log("phase2_complete", gate=gate, distribution=distribution)
    return result


def main():
    start = time.time()
    torch.manual_seed(p.SEED)
    np.random.seed(p.SEED)
    torch.set_num_threads(4)
    phase1 = json.loads((p.OUT / "phase1_summary.json").read_text())
    assert phase1["gate"]["pass"]
    manifest = p.verify_environment()
    os.chdir(p.UPSTREAM)
    wam = p.load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth").to(p.DEVICE).eval()
    p.log("model_loaded_for_eight", parameters=sum(parameter.numel() for parameter in wam.parameters()))
    result = run_eight(wam, manifest)
    final = json.loads((p.OUT / "final_result.json").read_text())
    final.update({"phase2": result, "last_completed_phase": "eight-region clean feasibility",
                  "decision": "PENDING" if result["gate"]["pass"] else "MODIFY",
                  "stop_reason": None if result["gate"]["pass"] else "Phase 2 eight-region clean gate failed",
                  "phase2_elapsed_seconds": time.time()-start})
    p.write_json("final_result.json", final)
    p.log("phase2_decision", decision=final["decision"], gate_pass=result["gate"]["pass"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
