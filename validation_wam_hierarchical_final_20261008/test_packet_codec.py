"""Exhaustive 0–6 bit pattern test for the full CRC/BCH/parity packet codec."""
import json
import random
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from packet_codec import crc5, decode_packet, encode_packet, packet16


def run():
    start = time.perf_counter()
    for packet_id in range(8):
        for symbol in range(256):
            word = encode_packet(packet_id, symbol)
            result = decode_packet(word)
            assert result["status"] == "ACCEPTED"
            assert (result["packet_id"], result["symbol"]) == (packet_id, symbol)
            assert packet16(packet_id, symbol) >> 5 == (packet_id << 8) | symbol
    assert crc5(0) == 0
    counts = {}
    for weight in range(7):
        cases = Counter()
        for positions in combinations(range(32), weight):
            error = sum(1 << bit for bit in positions)
            result = decode_packet(error)  # zero packet; linear code covers every input.
            if result["bch"]["status"] == "valid" and result["bch"]["payload"] != 0:
                cases["bch_silent_wrong_before_crc"] += 1
                if result["reason"] == "crc_reject":
                    cases["crc_rejected_bch_silent_wrong"] += 1
            if result["status"] == "ERASURE":
                cases["erasure"] += 1
                cases[result["reason"]] += 1
            elif (result["packet_id"], result["symbol"]) == (0, 0):
                cases["exact"] += 1
            else:
                cases["silent_wrong"] += 1
        assert cases["erasure"] + cases["exact"] + cases["silent_wrong"] == __import__("math").comb(32, weight)
        if weight <= 3:
            assert cases["exact"] == __import__("math").comb(32, weight)
        if weight == 4:
            assert cases["erasure"] == __import__("math").comb(32, weight)
        counts[str(weight)] = dict(cases)
    rng = random.Random(20261008)
    random_trials = 0
    for weight in range(7):
        for _ in range(1000):
            packet_id, symbol = rng.randrange(8), rng.randrange(256)
            error = sum(1 << bit for bit in rng.sample(range(32), weight))
            result = decode_packet(encode_packet(packet_id, symbol) ^ error)
            if weight <= 3:
                assert result["status"] == "ACCEPTED"
                assert (result["packet_id"], result["symbol"]) == (packet_id, symbol)
            if weight == 4:
                assert result["status"] == "ERASURE"
            random_trials += 1
    output = {"crc_definition": "MSB first, poly x^5+x^2+1 (0x25), init 0, augment 5 zero bits, xorout 0, no reflection",
              "all_2048_zero_error_inputs": True,
              "exhaustive_zero_packet_bit_error_patterns": counts,
              "random_nonzero_input_trials": random_trials,
              "elapsed_seconds": time.perf_counter()-start}
    out = Path(__file__).resolve().parent / "results" / "packet_codec_verification.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    run()
