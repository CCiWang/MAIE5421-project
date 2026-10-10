"""Strong, deterministic codec verification before any WAM measurement."""
import json
import random
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from bch31 import COSETS, GENERATOR, bits_to_int, decode_int, encode_int, int_to_bits


def run():
    start = time.perf_counter()
    rng = random.Random(20261008)
    assert COSETS == [[1, 2, 4, 8, 16], [3, 6, 12, 24, 17], [5, 10, 20, 9, 18]]
    min_distance_31 = min((encode_int(message) >> 1).bit_count() for message in range(1, 1 << 16))
    min_distance = min(encode_int(message).bit_count() for message in range(1, 1 << 16))
    assert min_distance_31 == 7, min_distance_31
    assert min_distance == 8, min_distance
    for message in range(1 << 16):
        encoded = encode_int(message)
        assert bits_to_int(int_to_bits(encoded)) == encoded
        decoded = decode_int(encoded)
        assert decoded["status"] == "valid" and decoded["payload"] == message
        assert decoded["corrected_total_bits"] == 0
    patterns = Counter()
    for weight in (1, 2, 3, 4):
        for positions in combinations(range(32), weight):
            error = sum(1 << position for position in positions)
            decoded = decode_int(error)  # Zero codeword; linearity covers all messages.
            patterns[weight] += 1
            if weight <= 3:
                assert decoded["status"] == "valid" and decoded["payload"] == 0
                assert decoded["corrected_total_bits"] == weight
            else:
                assert decoded["status"] != "valid", (positions, decoded)
    random_trials = 0
    for weight in (0, 1, 2, 3, 4, 5, 6):
        for _ in range(1000):
            message = rng.randrange(1 << 16)
            error = sum(1 << position for position in rng.sample(range(32), weight))
            decoded = decode_int(encode_int(message) ^ error)
            if weight <= 3:
                assert decoded["status"] == "valid" and decoded["payload"] == message
            elif weight == 4:
                assert decoded["status"] != "valid"
            random_trials += 1
    # Five or more errors are outside the guaranteed radius and can silently
    # miscorrect. Record actual behavior rather than implying perfect detection.
    beyond_radius = Counter()
    for weight in (5, 6):
        for _ in range(10000):
            message = rng.randrange(1 << 16)
            error = sum(1 << position for position in rng.sample(range(32), weight))
            decoded = decode_int(encode_int(message) ^ error)
            if decoded["status"] == "valid":
                key = "silent_miscorrection" if decoded["payload"] != message else "correct"
            else:
                key = "detected_failure"
            beyond_radius[f"{weight}_{key}"] += 1
    result = {
        "generator_hex": hex(GENERATOR), "generator_degree": GENERATOR.bit_length() - 1,
        "cosets": COSETS, "minimum_distance_bch_31": min_distance_31,
        "minimum_distance_extended_32": min_distance,
        "all_65536_messages_zero_error_roundtrip": True,
        "exhaustive_error_patterns_on_zero_codeword": dict(patterns),
        "random_trials_0_to_6_errors": random_trials,
        "beyond_radius_5_and_6_bit_behavior": dict(beyond_radius),
        "elapsed_seconds": time.perf_counter() - start,
    }
    output = Path(__file__).parent / "results" / "codec_verification.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
