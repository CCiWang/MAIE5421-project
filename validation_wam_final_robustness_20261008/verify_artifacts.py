"""Independent structural checks and provenance manifest for final artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PREV = ROOT.parent / "validation_wam_hierarchical_final_20261008" / "results"


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cases = rows(RESULTS / "cases.csv")
    packets = rows(RESULTS / "packets.csv")
    negative = rows(RESULTS / "negative_control.csv")
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert len(cases) == 420 and len(packets) == 3360 and len(negative) == 10
    assert all(count == 30 for count in Counter(x["condition"] for x in cases).values())
    assert len(set((x["case_id"], x["condition"]) for x in cases)) == 420
    assert len(set((x["case_id"], x["condition"], x["region"]) for x in packets)) == 3360
    assert all(int(x["erased_packets"]) + int(x["exact_crc_valid_packets"]) +
               int(x["silent_wrong_packets"]) == 8 for x in cases)
    assert all(not (x["global_exact"] == "True" and x["wrong_global_output"] == "True") for x in cases)
    for condition in {x["condition"] for x in cases}:
        subset = [x for x in cases if x["condition"] == condition]
        aggregate = next(x for x in summary["conditions"] if x["condition"] == condition)
        assert aggregate["global_exact"] == sum(x["global_exact"] == "True" for x in subset)
    old_crops = rows(PREV / "final_crop_cases.csv")
    geometry = [{"image": x["image"], "image_index": x["image_index"], "payload_seed": x["payload_seed"],
                 "retained_actual": x["retained_actual"], "variant": x["variant"],
                 "x0": x["x0"], "y0": x["y0"], "width": x["width"], "height": x["height"]}
                for x in old_crops if x["method"] == "repetition" and
                x["retained_target"] == "0.75" and x["variant"] == "0"]
    assert len(geometry) == 30
    assert set((x["image_index"], x["payload_seed"]) for x in geometry) == \
           set((x["image_index"], x["payload_seed"]) for x in cases if x["condition"] == "crop75_jpeg75")
    with (RESULTS / "crop_geometry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(geometry[0]))
        writer.writeheader()
        writer.writerows(geometry)
    assets = [ROOT / "PROTOCOL.md", ROOT / "REPRODUCE.sh", ROOT / "run_robustness.py", ROOT / "finalize.py",
              ROOT / "verify_artifacts.py", ROOT / "FINAL_VALIDATION_REPORT.md"]
    assets += sorted(path for folder in (RESULTS, ROOT / "figures", ROOT / "logs")
                     for path in folder.iterdir() if path.is_file() and
                     path.name not in ("artifact_manifest.json", "verification.log"))
    manifest = {str(path.relative_to(ROOT)): {"sha256": digest(path), "bytes": path.stat().st_size}
                for path in assets}
    (RESULTS / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"checks_passed": True, "cases": len(cases), "packet_rows": len(packets),
                      "negative_images": len(negative), "crop_geometry_rows": len(geometry),
                      "artifacts": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
