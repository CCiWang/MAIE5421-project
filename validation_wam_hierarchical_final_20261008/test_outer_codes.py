"""Algebraic MDS and repetition validation before any outer-code WAM run."""
import json
import random
from pathlib import Path

from outer_codes import (collect_accepted_packets, decode_mds, decode_repetition,
                         encode_mds, encode_repetition, exhaustive_mds_verification)


def main():
    rng = random.Random(20261008)
    sources = [(0, 0, 0, 0), (255, 255, 255, 255), (1, 2, 3, 4)]
    sources += [tuple(rng.randrange(256) for _ in range(4)) for _ in range(256)]
    result = exhaustive_mds_verification(sources)
    for source in sources:
        mds = encode_mds(source)
        repetition = encode_repetition(source)
        assert decode_mds(dict(enumerate(mds))) == {"status": "decoded", "payload": source}
        assert decode_repetition(dict(enumerate(repetition))) == {"status": "decoded", "payload": source}
    source = (11, 22, 33, 44)
    repeated = encode_repetition(source)
    assert decode_repetition({0: repeated[0], 7: repeated[7], 1: repeated[1], 6: repeated[6],
                              2: repeated[2], 5: repeated[5], 3: repeated[3], 4: repeated[4]})["payload"] == source
    assert decode_repetition({0: 11, 7: 12, 1: 22, 2: 33, 3: 44})["status"] == "repetition_conflict"
    unique, info = collect_accepted_packets([
        {"status": "ACCEPTED", "packet_id": 0, "symbol": 11},
        {"status": "ACCEPTED", "packet_id": 0, "symbol": 12},
        {"status": "ACCEPTED", "packet_id": 1, "symbol": 22},
        {"status": "ERASURE", "packet_id": None, "symbol": None},
    ])
    assert unique == {1: 22} and info["conflicting_ids"] == [0]
    result["repetition_source_by_region"] = [0, 1, 2, 3, 3, 2, 1, 0]
    result["conflict_and_missing_packet_tests"] = "passed"
    path = Path(__file__).resolve().parent / "results" / "outer_code_verification.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
