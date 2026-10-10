# Research evidence and repository scope

This repository retains four evidence stages. Numerical result files are unchanged; repository cleanup only selects which artifacts are appropriate for public version control.

## 1. Official WAM baseline

`validation_wam_20261007/` established that the pinned official Watermark Anything implementation and checkpoint run successfully with localized messages. In the clean four-region experiment, all failed packets were localized and their message errors were limited to one to three bits. This motivated an inner error-correcting code.

Retained evidence includes the protocol, implementation, reports, clean case summary, and final result summary. The local Python environments, upstream checkout, checkpoint, raw logs, and regenerable visualization intermediates are ignored.

## 2. Inner BCH validation

`validation_wam_inner_ecc_20261008/` implements and verifies binary BCH(31,16), designed distance 7, with an overall parity bit. Exhaustive code tests prove correction of every 0–3 bit error pattern and rejection of every 4-bit pattern for the extended 32-bit codeword. Behavior outside the guaranteed radius is measured rather than assumed.

The paired clean WAM test recovered 120/120 protected 16-bit payloads, compared with 115/120 exact raw 32-bit WAM messages. Blind crop experiments then showed that physically surviving regions were usually recovered and that failures were primarily erasures.

## 3. CRC, eight regions, and outer-code selection

`validation_wam_hierarchical_final_20261008/` adds the fixed CRC-5 packet format, validates eight localized protected packets, and compares MDS(8,4) with a strong spatial repetition baseline under the same physical embedding budget.

MDS recovered 162/180 paired blind-crop payloads and repetition recovered 161/180. This 0.56 percentage-point combined difference failed the predeclared selection gate, and repetition performed better at 75% retained area. The project therefore keeps repetition and records MDS as a negative result.

## 4. Final robustness evaluation

`validation_wam_final_robustness_20261008/` freezes the selected system and evaluates clean images, JPEG Q75/Q50, Gaussian noise σ2/σ5, rotations from ±1° through ±10°, one crop-plus-JPEG condition, and unmarked controls. The final decision is `LIMITED`: mild noise and small rotations are useful, while strong JPEG, large rotation, and crop-plus-compression remain weak.

The final report, per-case CSV, per-packet CSV, summary JSON, and final figures are retained. Raw console logs, machine-specific environment snapshots, and caches are ignored.

## Public repository exclusions

- `wam_mit.pth` and all model checkpoint patterns
- the nested official WAM clone and its `.git` history
- local virtual/Conda environments
- raw logs and event streams containing machine paths
- `.DS_Store`, Python caches, Matplotlib caches, and scratch directories
- redundant packet-level intermediates from earlier sprints

The upstream WAM commit and checkpoint hash are recorded in the root README so these exclusions do not remove reproducibility information.
