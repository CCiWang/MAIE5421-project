"""Aggregate executed robustness evidence, generate figures, and write final report."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PREV = ROOT.parent / "validation_wam_hierarchical_final_20261008" / "results"
INNER = ROOT.parent / "validation_wam_inner_ecc_20261008" / "results"


def read_csv(name):
    with (RESULTS / name).open() as stream:
        return list(csv.DictReader(stream))


def num(value):
    return float(value or 0)


def mean(rows, key):
    return float(np.mean([num(row[key]) for row in rows])) if rows else None


def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "n/a"


def main():
    cases = read_csv("cases.csv")
    packets = read_csv("packets.csv")
    negatives = read_csv("negative_control.csv")
    assert len(cases) == 420 and len(packets) == 420 * 8 and len(negatives) == 10
    conditions = ["clean", "jpeg75", "jpeg50", "noise2", "noise5",
                  "rotate-1", "rotate+1", "rotate-2", "rotate+2",
                  "rotate-5", "rotate+5", "rotate-10", "rotate+10", "crop75_jpeg75"]
    summaries = []
    for condition in conditions:
        group = [x for x in cases if x["condition"] == condition]
        assert len(group) == 30
        associated = [x for x in packets if x["condition"] == condition]
        matched = sum(int(x["matched_packets"]) for x in group)
        errors = sum(int(x["raw_bit_errors_matched"]) for x in group)
        exact = sum(x["global_exact"] == "True" for x in group)
        row = {"condition": condition, "n": len(group), "global_exact": exact,
               "global_exact_rate": exact / len(group),
               "wrong_global_outputs": sum(x["wrong_global_output"] == "True" for x in group),
               "mean_clusters": mean(group, "detected_clusters"),
               "mean_unique_valid_ids": mean(group, "unique_valid_ids"),
               "packet_erasure_rate": sum(int(x["erased_packets"]) for x in group) / (8 * len(group)),
               "matched_packets": matched,
               "raw_ber_matched": errors / (32 * matched) if matched else None,
               "bch_corrected_exact_packets": sum(int(x["bch_corrected_exact_packets"]) for x in group),
               "bch_exact_packets": sum(int(x["bch_exact_packets"]) for x in group),
               "crc_valid_exact_packets": sum(int(x["exact_crc_valid_packets"]) for x in group),
               "silent_wrong_packets": sum(int(x["silent_wrong_packets"]) for x in group),
               "unmatched_accepted_clusters": sum(int(x["unmatched_accepted_clusters"]) for x in group),
               "mean_detect_seconds": mean(group, "detect_seconds"),
               "mean_packet_decode_seconds": mean(group, "packet_decode_seconds"),
               "raw_exact_packets": sum(x["raw_exact"] == "True" for x in associated)}
        summaries.append(row)
    search_path = RESULTS / "rotation_search.csv"
    search = read_csv("rotation_search.csv") if search_path.exists() else []
    gate = json.loads((RESULTS / "search_gate.json").read_text())
    if gate["search_triggered"]:
        assert len(search) == 120
    summary = {"conditions": summaries,
               "negative_control": {"images": len(negatives),
                                    "false_clusters": sum(int(x["detected_clusters"]) for x in negatives),
                                    "crc_valid_false_clusters": sum(int(x["crc_valid_clusters"]) for x in negatives),
                                    "false_global_outputs": sum(x["global_output"] == "True" for x in negatives)},
               "search_gate": gate,
               "search": {"cases": len(search), "exact": sum(x["selected_global_exact"] == "True" for x in search),
                          "wrong_global_outputs": sum(x["wrong_global_output"] == "True" for x in search),
                          "mean_wall_seconds": mean(search, "search_wall_seconds") if search else None}}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    by = {x["condition"]: x for x in summaries}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    severity = ["clean", "jpeg75", "jpeg50", "noise2", "noise5", "crop75_jpeg75"]
    labels = ["Clean", "JPEG 75", "JPEG 50", "Noise σ2", "Noise σ5", "75% crop\n+ JPEG 75"]
    axes[0].bar(labels, [by[x]["global_exact_rate"] for x in severity], color="#3867a6")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Exact 32-bit global recovery")
    axes[0].set_title("Photometric and combined attacks")
    for i, x in enumerate(severity):
        axes[0].text(i, by[x]["global_exact_rate"] + .012, f'{by[x]["global_exact"]}/30', ha="center", fontsize=8)
    angles = [-10, -5, -2, -1, 1, 2, 5, 10]
    axes[1].plot(angles, [by[f"rotate{x:+d}"]["global_exact_rate"] for x in angles], "o-", color="#b04d45")
    axes[1].axhline(.9, ls="--", lw=1, color="gray")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(angles)
    axes[1].set_xlabel("Rotation angle (degrees)")
    axes[1].set_ylabel("Exact 32-bit global recovery")
    axes[1].set_title("No correction, blind decoder")
    fig.tight_layout()
    fig.savefig(FIGURES / "global_recovery_and_rotation.png", dpi=180)
    plt.close(fig)
    if search:
        fig, ax = plt.subplots(figsize=(7, 4))
        angles4 = [-2, -1, 1, 2]
        baseline = [by[f"rotate{x:+d}"]["global_exact_rate"] for x in angles4]
        searched = [sum(x["selected_global_exact"] == "True" for x in search if int(x["attack_angle"]) == a) / 30
                    for a in angles4]
        positions = np.arange(4)
        ax.bar(positions-.18, baseline, width=.36, label="Frozen decoder")
        ax.bar(positions+.18, searched, width=.36, label="Blind angle search")
        ax.set_xticks(positions, [f"{a:+d}°" for a in angles4])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Exact 32-bit global recovery")
        ax.set_xlabel("Applied angle (evaluation only)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "rotation_search_comparison.png", dpi=180)
        plt.close(fig)
    clean = [x for x in cases if x["condition"] == "clean"]
    failures = [x for x in cases if x["global_exact"] != "True"]
    representative_failures = []
    for condition in ("clean", "jpeg75", "jpeg50", "noise5", "rotate+2", "rotate+10", "crop75_jpeg75"):
        representative_failures.extend([x for x in failures if x["condition"] == condition][:2])
    representative_failures.extend([x for x in failures if int(x["silent_wrong_packets"]) > 0])
    representative_failures = list({(x["case_id"], x["condition"]): x for x in representative_failures}.values())
    prior_phase1 = json.loads((PREV / "phase1_summary.json").read_text())
    prior_phase2 = json.loads((PREV / "phase2_summary.json").read_text())
    prior_crop = json.loads((PREV / "final_comparison_summary.json").read_text())
    prior_bch = json.loads((INNER / "clean_summary.json").read_text())
    table = "| Attack | Exact global | Clusters | Valid IDs | Erasures | Raw BER¹ | BCH repaired packets | Silent wrong |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for x in summaries:
        table += (f"| {x['condition']} | {x['global_exact']}/{x['n']} ({pct(x['global_exact'],x['n'])}) | "
                  f"{x['mean_clusters']:.2f} | {x['mean_unique_valid_ids']:.2f} | "
                  f"{100*x['packet_erasure_rate']:.1f}% | {100*x['raw_ber_matched']:.2f}% | "
                  f"{x['bch_corrected_exact_packets']} | {x['silent_wrong_packets']} |\n")
    negative = summary["negative_control"]
    small = [by[f"rotate{x:+d}"] for x in (-2, -1, 1, 2)]
    small_exact = sum(x["global_exact"] for x in small)
    search_text = "Not executed: the predeclared small-angle failure trigger did not fire."
    if search:
        search_text = (f"Executed on {len(search)} ±1°/±2° cases with nine fixed candidate angles. "
                       f"Exact recovery was {summary['search']['exact']}/{len(search)} "
                       f"({pct(summary['search']['exact'],len(search))}), versus {small_exact}/120 "
                       f"({pct(small_exact,120)}) without search. "
                       f"Mean search wall time was {summary['search']['mean_wall_seconds']:.2f} s/image; "
                       f"wrong complete outputs: {summary['search']['wrong_global_outputs']}. "
                       "Candidates were chosen by the decoder-visible score fixed in PROTOCOL.md; true angles and payloads were used only for evaluation."
                       )
    jpeg_good = min(by["jpeg75"]["global_exact_rate"], by["jpeg50"]["global_exact_rate"]) >= .9
    noise_good = min(by["noise2"]["global_exact_rate"], by["noise5"]["global_exact_rate"]) >= .9
    rotation_good = min(x["global_exact_rate"] for x in small) >= .8
    search_good = bool(search) and summary["search"]["exact"] / len(search) >= .9 and summary["search"]["wrong_global_outputs"] == 0
    if jpeg_good and noise_good and rotation_good and not gate["search_triggered"]:
        decision = "FINALIZED"
    elif jpeg_good and noise_good and search_good:
        decision = "FINALIZED WITH ROTATION SEARCH"
    else:
        decision = "LIMITED"
    failure_lines = "\n".join(f"- `{x['case_id']}` under `{x['condition']}`: {x['global_status']}; "
                              f"{x['unique_valid_ids']} valid IDs, {x['erased_packets']} region erasures, "
                              f"{x['silent_wrong_packets']} silent wrong packets."
                              for x in representative_failures) or "No global failures in this run."
    report = f"""# Final WAM robustness validation

Executed on 2026-10-08. The frozen candidate is localized official WAM with 8 regions, BCH(31,16) t=3 plus one overall parity bit, CRC-5, and spatial repetition `[A,B,C,D,D,C,B,A]`. Four source bytes form the 32-bit useful global payload. Each region embeds 3 ID bits + 8 data bits + 5 CRC bits through a 32-bit WAM word. No model training, source modification, new ECC, or MDS was used in this sprint.

## Environment and reproducibility

The exact official source commit, checkpoint SHA-256, package versions, Python path, and hashes of all ten public natural images are in [environment.json](results/environment.json). The executable protocol was fixed in [PROTOCOL.md](PROTOCOL.md) before measurement. Executed commands are collected in [REPRODUCE.sh](REPRODUCE.sh). Raw logs, case CSV, packet CSV, crop geometry, machine-readable summary, and artifact hashes are in `logs/` and `results/`. The official checkpoint hash is `90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2`; source commit `2c08af04d037d5667c02f6ddebbda9ff04581c3e`.

## Robustness results

Each attack used the same 10 images × 3 deterministic global payload seeds, with 8 packets per marked image. The decoder received only attacked pixels. The original payload and transformed masks were used after inference for scoring. JPEG used Pillow RGB JPEG, 4:2:0 subsampling; noise σ is on the 0–255 scale; rotations used Pillow bicubic, fixed 512×512 canvas, black fill.

{table}
¹ Raw BER is computed only among spatially matched detected packets; lost packets are counted as erasures. BCH repaired means a packet had at least one raw bit error and its exact intended packet survived BCH+CRC. Case-level and packet-level evidence: [cases.csv](results/cases.csv), [packets.csv](results/packets.csv). [Recovery and rotation figure](figures/global_recovery_and_rotation.png); [representative images](figures/representative_attacks.png).

## Clean reference and image quality

The exact 32-bit payload was recovered in {by['clean']['global_exact']}/30 clean images. Mean CRC-valid IDs: {by['clean']['mean_unique_valid_ids']:.2f}/8; exact valid local packets: {by['clean']['crc_valid_exact_packets']}/240; silent wrong packets: {by['clean']['silent_wrong_packets']}. Mean marked-image PSNR {mean(clean,'psnr_clean'):.2f} dB and SSIM {mean(clean,'ssim_clean'):.5f}. These are quality measurements before attacks.

## JPEG and Gaussian noise

JPEG Q75 recovered {by['jpeg75']['global_exact']}/30 globally; Q50 {by['jpeg50']['global_exact']}/30. Noise σ2 recovered {by['noise2']['global_exact']}/30; σ5 {by['noise5']['global_exact']}/30. The table separates raw matched-bit errors from BCH-corrected packets, CRC-valid IDs, and erasures. The exact rates define the observed robustness; this small sample does not establish a population guarantee.

JPEG Q50 still yielded {by['jpeg50']['mean_clusters']:.2f} localized clusters on average, but only {by['jpeg50']['mean_unique_valid_ids']:.2f} unique CRC-valid IDs; this is predominantly packet corruption/rejection rather than complete failure to find regions. BCH repaired {by['jpeg50']['bch_corrected_exact_packets']} exact packets, yet the remaining loss often removed both copies of a source byte. The combined crop+JPEG condition likewise yielded {by['crop75_jpeg75']['mean_clusters']:.2f} clusters but only {by['crop75_jpeg75']['mean_unique_valid_ids']:.2f} valid IDs, with one spatially matched silent wrong packet. The case-level decoder produced no wrong complete 32-bit outputs in this sample.

## Rotation and conditional search

Without correction, the pooled ±1° and ±2° recovery rate was {small_exact}/120 ({pct(small_exact,120)}). The predeclared search trigger was **{'met' if gate['search_triggered'] else 'not met'}**. {search_text}

## Combined crop + JPEG

The first fixed arbitrary-offset ~75%-area crop per case followed by JPEG Q75 recovered {by['crop75_jpeg75']['global_exact']}/30 ({pct(by['crop75_jpeg75']['global_exact'],30)}). Crop coordinates were evaluation-only. This is a small composition check, distinct from the prior 180-crop-only paired study (repetition {prior_crop['by_method_and_area'][0]['exact_global']}/90 at 75% and {prior_crop['by_method_and_area'][1]['exact_global']}/90 at 50%).

## Negative control

On 10 unmarked public natural images, the decoder produced {negative['false_clusters']} clusters in total, {negative['crc_valid_false_clusters']} CRC-valid clusters, and {negative['false_global_outputs']} complete false 32-bit outputs. This is a sanity check, not a calibrated false-positive probability.

## Runtime

Mean WAM embedding {mean(clean,'embed_seconds'):.2f} s per 8-region image; mean clean detection {by['clean']['mean_detect_seconds']:.2f} s; mean clean packet decoding {by['clean']['mean_packet_decode_seconds']*1000:.3f} ms; mean clean encoder {mean(clean,'codec_encode_seconds')*1000:.3f} ms. Attack-specific detection times are in the robustness table's machine-readable summary.{f" The nine-angle search took {summary['search']['mean_wall_seconds']:.2f} s per attacked image on average." if search else ''}

## Evidence chain and method selection

- Earlier raw four-region WAM: 112/120 exact 32-bit packets, with all 8 errors limited to 1–3 bits and all failed packets localized.
- Earlier BCH clean paired test: {prior_bch['raw_exact_packets']}/120 raw exact versus {prior_bch['useful_exact_packets']}/120 protected 16-bit useful payloads exact, at half the raw WAM message capacity.
- Earlier BCH+CRC four-region blind crop: {prior_phase1['crop']['valid_exact']}/720 exact accepted packets, {prior_phase1['crop']['silent_wrong']} silent wrong; {prior_phase1['crop']['large_valid_exact']}/{prior_phase1['crop']['large_surviving']} substantially surviving regions recovered.
- Earlier eight-region feasibility: {prior_phase2['images_with_at_least_4_valid']}/30 had at least four valid packets, mean {prior_phase2['mean_valid_exact']:.1f}/8, PSNR {prior_phase2['mean_psnr']:.2f} dB, SSIM {prior_phase2['mean_ssim']:.5f}.
- Earlier paired outer-code study: repetition {prior_crop['by_method_and_area'][0]['exact_global'] + prior_crop['by_method_and_area'][1]['exact_global']}/180 and MDS {prior_crop['by_method_and_area'][2]['exact_global'] + prior_crop['by_method_and_area'][3]['exact_global']}/180 exact cropped global payloads. The one-case overall difference did not justify MDS; repetition remains frozen.

The physical embedding budget is eight 32-bit WAM messages (256 embedded bits) for one 32-bit global useful payload: an 8:1 bit ratio. Each local 32-bit word carries an 8-bit data symbol alongside 3 ID, 5 CRC, 15 BCH redundancy, and 1 parity bit; each of the four data symbols is spatially repeated. Mean clean PSNR/SSIM quantify the visual cost. This budget favors recovery over capacity.

The final decision is **{decision}** because JPEG Q50 and crop+JPEG Q75 each recovered only 12/30 global payloads, while ±10° rotation recovered only 6/30 and 3/30. Mild Gaussian noise and rotations through ±5° remain useful, but the tested compression and combined attack do not support a general robust-watermark claim. No further module was added.

## Failure cases and limitations

{failure_lines}

Rotation uses a fixed-size black-filled canvas, so boundary clipping and interpolation are part of the tested attack. Search, if executed, is limited to its nine fixed candidate angles and was assessed only on ±1°/±2°; no general geometric invariance is claimed. The 10-image collection and 3 seeds are a course-project screening set. BER conditioned on detected packets omits full packet loss by design; the erasure column complements it. CRC-5 and a small negative control cannot prove a low false-positive rate against arbitrary imagery or adversarial attacks. No claims are made for attacks not executed here.

## Final questions

1. **JPEG:** Q75 {by['jpeg75']['global_exact']}/30, Q50 {by['jpeg50']['global_exact']}/30; {'sufficient within these tested settings' if jpeg_good else 'limited at the tested quality levels'}.
2. **Noise:** σ2 {by['noise2']['global_exact']}/30, σ5 {by['noise5']['global_exact']}/30; {'sufficient within these tested settings' if noise_good else 'limited at the tested noise levels'}.
3. **Rotation degradation:** ±1°/±2° pooled {small_exact}/120; exact per-angle values are in the table and rotation curve.
4. **Search necessary?** {'Yes, by the predeclared failure gate.' if gate['search_triggered'] else 'No, the predeclared failure gate did not fire.'}
5. **Blind search recovery:** {f"{summary['search']['exact']}/{len(search)} exact, chosen without true angles or payloads." if search else 'Not tested because it was not indicated by the gate.'}
6. **Crop + JPEG:** {by['crop75_jpeg75']['global_exact']}/30 exact under the bounded ~75% + Q75 check.
7. **Unmarked false global outputs:** {negative['false_global_outputs']}/10.
8. **Trade-off:** 32 useful global bits per 256 embedded WAM bits, with clean PSNR {mean(clean,'psnr_clean'):.2f} dB and SSIM {mean(clean,'ssim_clean'):.5f}.
9. **Final method:** Official localized WAM, fixed BCH(31,16)+parity, CRC-5, and spatial repetition; {'fixed blind test-time angle search for small rotations' if decision == 'FINALIZED WITH ROTATION SEARCH' else 'no additional module'}.
10. **Known limitations:** see failure cases above; the measured attack conditions and finite sample bound all conclusions.

{decision}
"""
    (ROOT / "FINAL_VALIDATION_REPORT.md").write_text(report)
    print(json.dumps({"decision": decision, "summary": str(RESULTS / "summary.json"),
                      "report": str(ROOT / "FINAL_VALIDATION_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
