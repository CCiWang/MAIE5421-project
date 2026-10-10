"""Clarify matched silent errors versus accepted unmatched clusters from saved CSV."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"
with (ROOT / "final_crop_cases.csv").open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 360
summary_path = ROOT / "final_comparison_summary.json"
summary = json.loads(summary_path.read_text())
matched, unmatched, conservative = {}, {}, {}
for method in ("repetition", "mds"):
    cases = [row for row in rows if row["method"] == method]
    assert len(cases) == 180
    matched[method] = sum(int(row["silent_wrong_packets"]) for row in cases)/(180*8)
    unmatched[method] = sum(int(row["unmatched_accepted_clusters"]) for row in cases)/(180*8)
    conservative[method] = matched[method]+unmatched[method]
summary["silent_wrong_rates"] = matched
summary["unmatched_accepted_rates"] = unmatched
summary["conservative_accepted_error_or_unmatched_rates"] = conservative
summary["gate"].pop("silent_wrong_le_0p5pct_each_method", None)
summary["gate"]["silent_or_unmatched_accepted_le_0p5pct_each_method"] = all(
    value <= .005 for value in conservative.values())
summary["gate"]["pass"] = all(value for key,value in summary["gate"].items() if key != "pass")
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
final_path = ROOT / "final_result.json"
final = json.loads(final_path.read_text())
final["phase5"] = summary
final["decision"] = "GO" if summary["gate"]["pass"] else "MODIFY"
final["stop_reason"] = None if summary["gate"]["pass"] else "Final paired gain/quality/accepted-error gate failed"
final_path.write_text(json.dumps(final, indent=2) + "\n")
print(json.dumps({"matched_silent_wrong_rates": matched,
                  "accepted_unmatched_rates": unmatched,
                  "conservative_rates": conservative,
                  "gate": summary["gate"]}, indent=2))
