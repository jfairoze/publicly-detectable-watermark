import argparse
import hashlib
import logging
import pickle

import numpy as np
from bls.utils import *
from bplib.bp import G2Elem
from petlib.pack import decode

import crypto

logging.basicConfig(filename="logging.log", encoding="utf-8", level=logging.INFO)


def main(args: argparse.Namespace) -> None:
    # Load in candidate watermarked text.
    with open(args.document, "r") as f:
        text = f.read()
        text = text.rstrip("\n")

    if args.pk:
        with open(args.pk, "rb") as f:
            pk = decode(pickle.load(f))
    if args.params:
        with open(args.params, "rb") as g:
            G = decode(pickle.load(g))
            params = (G, G.order(), G.gen1(), G.gen2(), G.pair)

    if args.gen_type == "asymmetric":
        watermarked = search_for_asymmetric_watermark(
            pk,
            params,
            text,
            args.signature_segment_length,
            args.bit_size,
            args.max_planted_errors,
        )
    elif args.gen_type == "symmetric":
        watermarked = detect_symmetric_watermark(text, args.security_parameter)
    logging.info(f"watermarked: {watermarked}")
    print(watermarked)


def detect_symmetric_watermark(bitstring: str, security_parameter: float) -> bool:
    """Detects a watermark in a bitstring using the symmetric watermark detection algorithm, Algorithm 4 from Christ et al. (2023)."""

    L = len(bitstring)

    for i in range(L):
        r = bitstring[: (i + 1)]

        v = []
        for j in range(i + 1, L):
            unkeyed_hash = crypto.unkeyed_hash_to_float(
                bytes(r, "utf-8") + bytes(bin(j), "utf-8")
            )
            v.append(
                int(bitstring[j]) * unkeyed_hash
                + (1 - int(bitstring[j])) * (1 - unkeyed_hash)
            )

        lhs = sum([np.log(1 / vj) for vj in v])
        rhs = (L - i) + security_parameter * np.sqrt(L - i)
        if lhs > rhs:
            return True

    return False


def search_for_asymmetric_watermark(
    pk: G2Elem,
    params: tuple,
    text: str,
    message_length: int,
    signature_segment_length: int,
    bit_size: int,
    max_planted_errors: int,
) -> bool:
    """Searches for a watermark in a text using the asymmetric watermark detection algorithm, Algorithm 2 from our paper."""

    # For each rotation of the text, check if the watermark is present.
    for i in range(len(text)):
        rotated_text = text[i:] + text[:i]
        if detect_asymmetric_watermark(
            pk,
            params,
            rotated_text,
            message_length,
            signature_segment_length,
            bit_size,
            max_planted_errors,
        ):
            return True
    return False


def detect_asymmetric_watermark(
    pk: G2Elem,
    params: tuple,
    text: str,
    message_length: int,
    signature_segment_length: int,
    bit_size: int,
    max_planted_errors: int,
) -> bool:
    """Check one window of the text for an asymmetric watermark."""

    try:
        message = text[:message_length]
        message_hash = hashlib.sha256(bytes(message, "utf-8")).digest()

        signature_str = text[message_length:]
        signature = ""
        num_signature_segments = 0
        signature_codeword_length = crypto.get_signature_codeword_length(
            max_planted_errors, bit_size
        )
        for i in range(0, len(signature_str), signature_segment_length):
            if num_signature_segments >= signature_codeword_length // bit_size:
                break
            signature_segment = signature_str[i : i + signature_segment_length]
            unkeyed_hash = crypto.unkeyed_hash_to_bits(
                bytes(message, "utf-8")
                + bytes(signature, "utf-8")
                + bytes(signature_segment, "utf-8"),
                bit_size,
            )
            signature += unkeyed_hash
            num_signature_segments += 1

        logging.info(f"extracted signature\n{signature}")

        watermarked = crypto.decode_and_verify_openssl(
            pk, message_hash, signature, params, max_planted_errors
        )
        return watermarked
    except Exception as e:
        logging.error(e)
        return False


def validate_args(args: argparse.Namespace) -> None:
    if args.gen_type == "asymmetric" and (args.pk is None or args.params is None):
        raise ValueError(
            f"pk and params must be provided for asymmetric watermark detection"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="check for a watermark in text")
    parser.add_argument(
        "document",
        type=str,
        help="a file containing the text to be checked",
    )
    parser.add_argument(
        "--gen-type",
        default="asymmetric",
        type=str,
        help="the algorithm used for generation, currently supported: 'asymmetric'",
        choices=("asymmetric"),
    )

    # The following arguments are used in asymmetric detection only: pk, params, signature_segment_length, bit_size, and max_planted_errors.
    # pk and params must be specified for asymmetric watermark detection.
    parser.add_argument(
        "--pk",
        default=None,
        type=str,
        help="the input path to public detection key pickle file",
    )
    parser.add_argument(
        "--params", default=None, type=str, help="the input path to params pickle file"
    )
    parser.add_argument(
        "--signature-segment-length",
        default=crypto.DEFAULT_SIGNATURE_SEGMENT_LENGTH,
        type=int,
        help="the length of each signature segment in characters",
    )
    parser.add_argument(
        "--bit-size",
        default=crypto.DEFAULT_BIT_SIZE,
        type=int,
        help="the number of signature bits in each segment",
    )
    parser.add_argument(
        "--max-planted-errors",
        default=crypto.DEFAULT_MAX_PLANTED_ERRORS,
        type=int,
        help="the number of error correcting symbols to use",
    )

    args = parser.parse_args()
    if args.gen_type == "asymmetric" and (args.pk is None or args.params is None):
        parser.error("--gen-type asymmetric requires --pk and --params")

    # The following argument is used in symmetric watermarking only: security_parameter
    parser.add_argument(
        "--security-parameter",
        default=crypto.DEFAULT_SECURITY_PARAMETER,
        type=int,
        help="the security parameter for the symmetric watermarking algorithm",
    )

    main(args)
