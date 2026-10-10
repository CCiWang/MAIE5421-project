"""Paired official-WAM CRC/BCH MDS-versus-repetition blind crop experiment."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

import run_phase1 as p
from outer_codes import (collect_accepted_packets, decode_mds, decode_repetition,
                         encode_mds, encode_repetition)


def source_for(image_index: int, seed: int) -> tuple[int, int, int, int]:
    rng = np.random.default_rng(p.SEED + 70000 + image_index*1009 + seed*97)
    return tuple(int(value) for value in rng.integers(0, 256, 4, dtype=np.uint8))


def decode_global(detection: dict, method: str, source: tuple[int, ...]):
    start = time.perf_counter()
    decoded_packets = [cluster["packet"] for cluster in detection["clusters"]]
    unique, packet_counts = collect_accepted_packets(decoded_packets)
    outer = decode_mds(unique) if method == "mds" else decode_repetition(unique)
    outer_seconds = time.perf_counter()-start
    exact = outer["status"] == "decoded" and outer["payload"] == source
    wrong = outer["status"] == "decoded" and outer["payload"] != source
    return {"global_status": outer["status"], "global_payload_exact": bool(exact),
            "wrong_global_output": bool(wrong), "outer_decode_seconds": outer_seconds,
            **packet_counts}


def infer_and_score(wam, image: torch.Tensor, method: str, source: tuple[int, ...],
                    symbols: tuple[int, ...], messages: torch.Tensor,
                    oracle_masks: torch.Tensor, fractions: list[float]):
    # Blind decoding happens first; oracle geometry and truth are applied only afterwards.
    detection = p.infer_packets(wam, image)
    global_result = decode_global(detection, method, source)
    details, metrics = p.score(detection, list(symbols), messages, oracle_masks, fractions)
    raw_errors = sum(row["raw_hamming"] for row in details if row["raw_hamming"] is not None)
    row = {"method": method, "raw_exact_packets": sum(item["raw_exact"] for item in details),
           "bch_exact_packets": sum(item["bch_exact_16"] for item in details),
           "exact_valid_packets": sum(item["classification"] == "VALID" for item in details),
           "erased_packets": sum(item["classification"] == "ERASURE" for item in details),
           "silent_wrong_packets": sum(item["classification"] == "SILENT WRONG" for item in details),
           "raw_bit_errors_matched": raw_errors,
           "matched_region_packets": sum(item["localized"] for item in details),
           "detect_seconds": detection["runtime_seconds"],
           **metrics, **global_result}
    return row, details


def run_comparison(wam, manifest):
    masks = p.prior.previous.make_masks(8, p.DEVICE)
    crop_rows, crop_packets, clean_rows, clean_packets = [], [], [], []
    for image_index, record in enumerate(manifest):
        original = p.prior.load_image(record)
        for seed in range(3):
            source = source_for(image_index, seed)
            crop_windows = list(p.fixed_crops(image_index, seed))
            for method in ("repetition", "mds"):
                symbols = encode_repetition(source) if method == "repetition" else encode_mds(source)
                start_codec = time.perf_counter()
                messages = p.packet_messages(list(symbols))
                codec_encode_seconds = time.perf_counter()-start_codec
                marked, embed_seconds = p.prior.previous.embed_messages(wam, original, messages, masks)
                psnr, ssim = p.prior.quality(original, marked)
                clean, details = infer_and_score(wam, marked, method, source, symbols,
                                                 messages, masks, [1.0]*8)
                clean.update({"image": record["name"], "image_index": image_index,
                              "payload_seed": seed, "psnr": psnr, "ssim": ssim,
                              "embed_seconds": embed_seconds, "codec_encode_seconds": codec_encode_seconds})
                clean_rows.append(clean)
                for item in details:
                    clean_packets.append({"method": method, "image": record["name"],
                                          "image_index": image_index, "payload_seed": seed, **item})
                for retained, variant, ratio, x0, y0, width, height in crop_windows:
                    cropped = marked[:, :, y0:y0+height, x0:x0+width]
                    oracle_masks = masks[:, :, y0:y0+height, x0:x0+width]
                    fractions = [float(oracle_masks[index].sum()/masks[index].sum()) for index in range(8)]
                    crop, details = infer_and_score(wam, cropped, method, source, symbols,
                                                    messages, oracle_masks, fractions)
                    crop.update({"image": record["name"], "image_index": image_index,
                                 "payload_seed": seed, "retained_target": retained,
                                 "retained_actual": width*height/512**2,
                                 "variant": variant, "aspect_ratio": ratio,
                                 "x0": x0, "y0": y0, "width": width, "height": height,
                                 "psnr_full_marked": psnr, "ssim_full_marked": ssim})
                    crop_rows.append(crop)
                    for item in details:
                        crop_packets.append({"method": method, "image": record["name"],
                                             "image_index": image_index, "payload_seed": seed,
                                             "retained_target": retained, "variant": variant,
                                             "x0": x0, "y0": y0, "width": width, "height": height,
                                             **item})
                    p.log("paired_crop_case", method=method, image=record["name"], seed=seed,
                          area=retained, variant=variant,
                          exact_global=crop["global_payload_exact"],
                          valid_ids=crop["unique_valid_ids"],
                          silent_wrong=crop["silent_wrong_packets"])
                p.log("paired_clean_case", method=method, image=record["name"], seed=seed,
                      exact_global=clean["global_payload_exact"], valid_ids=clean["unique_valid_ids"])
                p.write_csv("final_clean_cases.csv", clean_rows)
                p.write_csv("final_clean_packets.csv", clean_packets)
                p.write_csv("final_crop_cases.csv", crop_rows)
                p.write_csv("final_crop_packets.csv", crop_packets)
    paired = []
    index = {(row["method"], row["image_index"], row["payload_seed"], row["retained_target"], row["variant"]): row
             for row in crop_rows}
    for image_index in range(len(manifest)):
        for seed in range(3):
            for area in (.75, .5):
                for variant in range(3):
                    repetition = index[("repetition", image_index, seed, area, variant)]
                    mds = index[("mds", image_index, seed, area, variant)]
                    assert all(repetition[key] == mds[key] for key in ("x0", "y0", "width", "height"))
                    paired.append({"image": repetition["image"], "image_index": image_index,
                                   "payload_seed": seed, "retained_target": area, "variant": variant,
                                   "x0": repetition["x0"], "y0": repetition["y0"],
                                   "width": repetition["width"], "height": repetition["height"],
                                   "repetition_exact": repetition["global_payload_exact"],
                                   "mds_exact": mds["global_payload_exact"],
                                   "difference": int(mds["global_payload_exact"])-int(repetition["global_payload_exact"]),
                                   "repetition_valid_ids": repetition["unique_valid_ids"],
                                   "mds_valid_ids": mds["unique_valid_ids"]})
    p.write_csv("paired_global_outcomes.csv", paired)
    by_method = []
    for method in ("repetition", "mds"):
        clean = [row for row in clean_rows if row["method"] == method]
        for area in (.75, .5):
            crops = [row for row in crop_rows if row["method"] == method and row["retained_target"] == area]
            matched = sum(row["matched_region_packets"] for row in crops)
            errors = sum(row["raw_bit_errors_matched"] for row in crops)
            by_method.append({
                "method": method, "retained_target": area, "crops": len(crops),
                "exact_global": sum(row["global_payload_exact"] for row in crops),
                "exact_global_rate": float(np.mean([row["global_payload_exact"] for row in crops])),
                "wrong_global_outputs": sum(row["wrong_global_output"] for row in crops),
                "mean_valid_ids": float(np.mean([row["unique_valid_ids"] for row in crops])),
                "mean_erased_ids": float(np.mean([row["erased_id_count"] for row in crops])),
                "exact_valid_packets": sum(row["exact_valid_packets"] for row in crops),
                "erased_region_packets": sum(row["erased_packets"] for row in crops),
                "silent_wrong_region_packets": sum(row["silent_wrong_packets"] for row in crops),
                "unmatched_accepted_clusters": sum(row["unmatched_accepted_clusters"] for row in crops),
                "conflicting_ids": sum(len(row["conflicting_ids"]) for row in crops),
                "raw_ber_among_matched": errors/(32*matched) if matched else None,
                "mean_clusters": float(np.mean([row["detected_clusters"] for row in crops])),
                "mean_psnr": float(np.mean([row["psnr"] for row in clean])),
                "mean_ssim": float(np.mean([row["ssim"] for row in clean])),
                "mean_embed_seconds": float(np.mean([row["embed_seconds"] for row in clean])),
                "mean_crop_detect_seconds": float(np.mean([row["detect_seconds"] for row in crops])),
                "mean_outer_decode_seconds": float(np.mean([row["outer_decode_seconds"] for row in crops])),
            })
    differences = {str(area): float(np.mean([row["difference"] for row in paired if row["retained_target"] == area]))
                   for area in (.75, .5)}
    combined_difference = float(np.mean([row["difference"] for row in paired]))
    rep_quality = [row for row in clean_rows if row["method"] == "repetition"]
    mds_quality = [row for row in clean_rows if row["method"] == "mds"]
    psnr_difference = float(np.mean([row["psnr"] for row in mds_quality])-
                            np.mean([row["psnr"] for row in rep_quality]))
    ssim_difference = float(np.mean([row["ssim"] for row in mds_quality])-
                            np.mean([row["ssim"] for row in rep_quality]))
    silent_rates = {method: sum(row["silent_wrong_packets"]
                                for row in crop_rows if row["method"] == method)/(180*8)
                    for method in ("repetition", "mds")}
    unmatched_accepted_rates = {method: sum(row["unmatched_accepted_clusters"]
                                            for row in crop_rows if row["method"] == method)/(180*8)
                                for method in ("repetition", "mds")}
    conservative_rates = {method: silent_rates[method]+unmatched_accepted_rates[method]
                          for method in ("repetition", "mds")}
    gate = {"combined_gain_ge_10pp": combined_difference >= .10,
            "nonnegative_gain_each_area": all(value >= 0 for value in differences.values()),
            "psnr_difference_le_0p5db": abs(psnr_difference) <= .5,
            "ssim_difference_le_0p005": abs(ssim_difference) <= .005,
            "silent_or_unmatched_accepted_le_0p5pct_each_method":
                all(value <= .005 for value in conservative_rates.values())}
    gate["pass"] = all(gate.values())
    result = {"paired_crops": len(paired), "by_method_and_area": by_method,
              "paired_differences_by_area": differences,
              "combined_paired_difference": combined_difference,
              "psnr_mds_minus_repetition_db": psnr_difference,
              "ssim_mds_minus_repetition": ssim_difference,
              "silent_wrong_rates": silent_rates,
              "unmatched_accepted_rates": unmatched_accepted_rates,
              "conservative_accepted_error_or_unmatched_rates": conservative_rates,
              "clean_global_by_method": {method: {
                  "cases": len([row for row in clean_rows if row["method"] == method]),
                  "exact_global": sum(row["global_payload_exact"] for row in clean_rows if row["method"] == method),
                  "wrong_global_outputs": sum(row["wrong_global_output"] for row in clean_rows if row["method"] == method)}
                  for method in ("repetition", "mds")},
              "gate": gate}
    p.write_json("final_comparison_summary.json", result)
    p.log("final_comparison_complete", gate=gate, combined_gain=combined_difference,
          gain_by_area=differences)
    return result


def main():
    start = time.time()
    torch.manual_seed(p.SEED)
    np.random.seed(p.SEED)
    torch.set_num_threads(4)
    phase1 = json.loads((p.OUT / "phase1_summary.json").read_text())
    phase2 = json.loads((p.OUT / "phase2_summary.json").read_text())
    algebra = json.loads((p.OUT / "outer_code_verification.json").read_text())
    assert phase1["gate"]["pass"] and phase2["gate"]["pass"]
    assert algebra["subsets"] == 70 and algebra["all_exact"]
    manifest = p.verify_environment()
    os.chdir(p.UPSTREAM)
    wam = p.load_model_from_checkpoint("checkpoints/params.json", "checkpoints/wam_mit.pth").to(p.DEVICE).eval()
    p.log("model_loaded_for_final", parameters=sum(parameter.numel() for parameter in wam.parameters()))
    result = run_comparison(wam, manifest)
    final = json.loads((p.OUT / "final_result.json").read_text())
    final.update({"phase3": algebra, "phase4": "strong repetition implemented and unit-tested",
                  "phase5": result, "last_completed_phase": "paired MDS versus repetition blind crop",
                  "decision": "GO" if result["gate"]["pass"] else "MODIFY",
                  "stop_reason": None if result["gate"]["pass"] else "Final paired gain/quality/silent-error gate failed",
                  "phase5_elapsed_seconds": time.time()-start})
    p.write_json("final_result.json", final)
    p.log("final_decision", decision=final["decision"], gate_pass=result["gate"]["pass"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
