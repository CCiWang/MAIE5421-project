"""CRC-5 + previously verified extended BCH(31,16) WAM packet codec."""
from __future__ import annotations

from pathlib import Path
import sys

PREVIOUS = Path(__file__).resolve().parent.parent / "validation_wam_inner_ecc_20261008"
sys.path.insert(0, str(PREVIOUS))
from bch31 import decode_int, encode_int, int_to_bits, bits_to_int

CRC_POLY = 0x25  # x^5 + x^2 + 1, including the x^5 term.


def crc5(data11: int) -> int:
    if not 0 <= data11 < (1 << 11):
        raise ValueError("CRC input must be 11 bits")
    remainder = data11 << 5  # five zero augmentation bits, init=0.
    for bit in range(15, 4, -1):
        if remainder & (1 << bit):
            remainder ^= CRC_POLY << (bit - 5)
    assert 0 <= remainder < 32
    return remainder


def packet16(packet_id: int, symbol: int) -> int:
    if not 0 <= packet_id < 8 or not 0 <= symbol < 256:
        raise ValueError("packet ID must be 3 bits and symbol 8 bits")
    data11 = (packet_id << 8) | symbol
    return (data11 << 5) | crc5(data11)


def encode_packet(packet_id: int, symbol: int) -> int:
    return encode_int(packet16(packet_id, symbol))


def decode_packet(word32: int) -> dict:
    bch = decode_int(word32)
    if bch["status"] != "valid":
        return {"status": "ERASURE", "reason": "bch_" + bch["status"],
                "packet_id": None, "symbol": None, "bch": bch}
    packet = bch["payload"]
    data11, check = packet >> 5, packet & 31
    if crc5(data11) != check:
        return {"status": "ERASURE", "reason": "crc_reject",
                "packet_id": None, "symbol": None, "bch": bch}
    return {"status": "ACCEPTED", "reason": None,
            "packet_id": data11 >> 8, "symbol": data11 & 255, "bch": bch}


def encode_bits(packet_id: int, symbol: int) -> list[int]:
    return int_to_bits(encode_packet(packet_id, symbol))
