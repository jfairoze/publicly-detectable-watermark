import hashlib
from struct import unpack
from typing import Any

from bitstring import BitArray
from bls.scheme import aggregate_sigma, aggregate_vk, setup, sign, ttp_keygen, verify
from bplib.bp import G1Elem, G2Elem
from petlib.pack import decode, encode
from reedsolo import RSCodec

# Constants for primitives
SIGNATURE_LENGTH: int = 328
REED_SOLO_CONSTANT: int = 8

# Constants for asymmetric watermarking
DEFAULT_SIGNATURE_SEGMENT_LENGTH: int = 16
DEFAULT_BIT_SIZE: int = 2
DEFAULT_MESSAGE_LENGTH: int = DEFAULT_SIGNATURE_SEGMENT_LENGTH // DEFAULT_BIT_SIZE
DEFAULT_MAX_PLANTED_ERRORS: int = 2

# Constant for symmetric watermarking
DEFAULT_SECURITY_PARAMETER: int = 16


def get_signature_codeword_length(max_planted_errors: int, bit_size: int) -> int:
    """Signature codeword length is signature length plus Reed-Solomon error correction bits."""

    signature_codeword_length = SIGNATURE_LENGTH + (
        REED_SOLO_CONSTANT * max_planted_errors * 2
    )
    assert (
        signature_codeword_length % bit_size == 0
    ), "Signature codeword length must be divisible by bit size."
    return signature_codeword_length


def unkeyed_hash_to_float(input_bytes: bytes) -> float:
    """Hashes a bytearray to a float in [0,1]."""
    return float(unpack("L", hashlib.sha256(input_bytes).digest()[:8])[0]) / 2**64


def unkeyed_hash_to_bits(input_bytes: bytes, bit_size: int) -> str | Any:
    """Hashes a bytearray to a fixed number of specified bits."""
    assert bit_size <= 256  # Using 256-bit hashes
    return BitArray(bytes=hashlib.sha256(input_bytes).digest()).bin[0:bit_size]


def bytes_to_binary_codeword(input_bytes: bytes, max_planted_errors: int) -> str | Any:
    """Converts a bytearray to a binary codeword with Reed-Solomon error correction."""

    if max_planted_errors == 0:
        return BitArray(bytes=input_bytes).bin
    rsc = RSCodec(max_planted_errors * 2)
    return BitArray(bytes=rsc.encode(input_bytes)).bin


def binary_codeword_to_bytes(
    binary_codeword: str, max_planted_errors: int
) -> bytes | Any:
    """Converts a binary codeword with Reed-Solomon error correction to a bytearray."""

    if max_planted_errors == 0:
        return BitArray(bin=binary_codeword).bytes
    rsc = RSCodec(max_planted_errors * 2)
    return rsc.decode(BitArray(bin=binary_codeword).bytes)[0]


def bls_generate_openssl() -> tuple[list, G2Elem, tuple]:
    """Initialize keys and parameters for BLS signature scheme."""

    params = setup()
    (sk, vk) = ttp_keygen(params, 1, 1)
    pk = aggregate_vk(params, vk, threshold=False)
    return sk, pk, params


def bls_sign_openssl(message: bytes, sk: list, params: tuple) -> bytes | Any:
    """Sign a message using BLS signature scheme."""

    sigs = [sign(params, ski, message) for ski in sk]
    sigma = aggregate_sigma(params, sigs, threshold=False)
    return encode(sigma)


def bls_verify_openssl(
    message: bytes, signature: bytes, pk: G2Elem, params: tuple
) -> bool | Any:
    """Verify a message using BLS signature scheme."""
    return verify(params, pk, decode(signature), message)


def sign_and_encode_openssl(
    sk: list, message: bytes, params: tuple, max_planted_errors: int
) -> str | Any:
    """Sign a message using BLS signature scheme, error-correct it, and mask it with a one-time pad."""

    signature: bytes = bls_sign_openssl(message, sk, params)
    h: bytes = hashlib.sha512(message).digest()
    signature_codeword = bytes_to_binary_codeword(signature, max_planted_errors)
    return BitArray(
        bytes=bytes(
            a ^ b
            for a, b in zip(BitArray(bin=signature_codeword).bytes, h, strict=False)
        )
    ).bin


def decode_and_verify_openssl(
    pk: G2Elem,
    message: bytes,
    binary_codeword: str,
    params: tuple,
    max_planted_errors: int,
) -> bool:
    """Unmask a binary codeword, correct errors, and verify it using BLS signature scheme."""

    h: bytes = hashlib.sha512(message).digest()
    unmasked_codeword = BitArray(
        bytes=bytes(
            a ^ b for a, b in zip(BitArray(bin=binary_codeword).bytes, h, strict=False)
        )
    ).bin
    signature = binary_codeword_to_bytes(unmasked_codeword, max_planted_errors)
    ok: bool = bls_verify_openssl(message, signature, pk, params)
    return ok
