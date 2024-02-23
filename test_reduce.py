import math
import unittest
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

import reduce


class TestReduce(unittest.TestCase):
    def test_sample_bits(self) -> None:
        @dataclass
        class TestCase:
            model_name: str
            generate_distribution_func: Callable

        cases = {
            "opt 1.3b model with dirichlet distribution": TestCase(
                model_name="facebook/opt-1.3b",
                generate_distribution_func=reduce.generate_dirichlet_distribution,
            ),
            "opt 1.3b model with uniform distribution": TestCase(
                model_name="facebook/opt-1.3b",
                generate_distribution_func=reduce.generate_uniform_distribution,
            ),
        }

        # num_samples is the number of samples to get for each test case
        # for the given model_name and probability distribution.
        num_samples = 10

        for name, c in cases.items():
            with self.subTest(name):
                tokens = reduce.generate_tokens(c.model_name)
                p = c.generate_distribution_func(len(tokens))
                (
                    encode,
                    decode,
                    padded_encoding,
                    max_bit_length,
                ) = reduce.simple_encoder(tokens)
                for i in range(num_samples):
                    bits = reduce.sample_bits(
                        p, padded_encoding, max_bit_length, "multinomial"
                    )
                    self.assertIn(
                        bits,
                        padded_encoding,
                        "sampled bits not found in padded encoding",
                    )
                    self.assertEqual(
                        max_bit_length,
                        len(bits),
                        "length of sampled bits is not equal to max bit length",
                    )

                    decoded = decode(bits)
                    self.assertIn(
                        decoded,
                        tokens,
                        "result of decoding sampled bits not found in original tokens",
                    )

    def test_get_probability_distribution_for_bit(self) -> None:
        @dataclass
        class TestCase:
            p: List[np.float64]
            encoding: List[str]
            bit_index: int
            max_bit_length: int
            previous_bits: str
            expected_probability_distribution: tuple[float, float]

        cases = {
            "two bits: get probability distribution for first bit": TestCase(
                p=[np.float64(e) for e in [0.1, 0.2, 0.3, 0.4]],
                encoding=["00", "01", "10", "11"],
                bit_index=0,
                max_bit_length=2,
                previous_bits="",
                expected_probability_distribution=(0.3, 0.7),
            ),
            "two bits: get probability distribution for second bit given first is 0": TestCase(
                p=[np.float64(e) for e in [0.1, 0.2, 0.3, 0.4]],
                encoding=["00", "01", "10", "11"],
                bit_index=1,
                max_bit_length=2,
                previous_bits="0",
                expected_probability_distribution=(
                    0.1 / (0.1 + 0.2),
                    0.2 / (0.1 + 0.2),
                ),
            ),
            "two bits: get probability distribution for second bit given first is 1": TestCase(
                p=[np.float64(e) for e in [0.1, 0.2, 0.3, 0.4]],
                encoding=["00", "01", "10", "11"],
                bit_index=1,
                max_bit_length=2,
                previous_bits="1",
                expected_probability_distribution=(
                    0.3 / (0.3 + 0.4),
                    0.4 / (0.3 + 0.4),
                ),
            ),
            "three bits: get probability distribution for second bit given first bit is 0": TestCase(
                p=[
                    np.float64(e)
                    for e in [0.25, 0.05, 0.1, 0.2, 0.03, 0.07, 0.12, 0.18]
                ],
                encoding=["000", "001", "010", "011", "100", "101", "110", "111"],
                bit_index=1,
                max_bit_length=3,
                previous_bits="0",
                expected_probability_distribution=(
                    (0.25 + 0.05) / (0.25 + 0.05 + 0.1 + 0.2),
                    (0.1 + 0.2) / (0.25 + 0.05 + 0.1 + 0.2),
                ),
            ),
            "three bits: get probability distribution for third bit given first and second bits are 01": TestCase(
                p=[
                    np.float64(e)
                    for e in [0.25, 0.05, 0.1, 0.2, 0.03, 0.07, 0.12, 0.18]
                ],
                encoding=["000", "001", "010", "011", "100", "101", "110", "111"],
                bit_index=2,
                max_bit_length=3,
                previous_bits="01",
                expected_probability_distribution=(
                    (0.1) / (0.1 + 0.2),
                    (0.2) / (0.1 + 0.2),
                ),
            ),
            "five bits: get probability distribution for third bit given first and second bits are 00": TestCase(
                p=[np.float64(e) for e in [0.25, 0.05, 0.1, 0.6]],
                encoding=["00000", "00001", "00010", "00011"],
                bit_index=2,
                max_bit_length=5,
                previous_bits="00",
                expected_probability_distribution=(
                    1,
                    0,
                ),
            ),
        }

        for name, c in cases.items():
            with self.subTest(name):
                probability_distribution = reduce.get_probability_distribution_for_bit(
                    c.p, c.encoding, c.bit_index, c.max_bit_length, c.previous_bits
                )
                self.assertTrue(
                    np.isclose(
                        c.expected_probability_distribution[0],
                        probability_distribution[0],
                    ),
                    f"probability of 0 for {c.bit_index}-th bit is off from expected: {c.expected_probability_distribution[0]},  got: {probability_distribution[0]}",
                )
                self.assertTrue(
                    np.isclose(
                        c.expected_probability_distribution[1],
                        probability_distribution[1],
                    ),
                    f"probability of 1 for {c.bit_index}-th bit is off from expected: {c.expected_probability_distribution[1]},  got: {probability_distribution[1]}",
                )

    def test_get_probability_distribution_for_bit_compared_to_original(self) -> None:
        @dataclass
        class TestCase:
            p: List[np.float64]
            encoding: List[str]
            num_bits: int

        cases = {
            "three bits with deterministic original probability distribution": TestCase(
                p=[
                    np.float64(e)
                    for e in [0.25, 0.05, 0.1, 0.2, 0.03, 0.07, 0.12, 0.18]
                ],
                encoding=["000", "001", "010", "011", "100", "101", "110", "111"],
                num_bits=3,
            ),
            "random 7 bits with random original probability distribution": TestCase(
                p=reduce.generate_dirichlet_distribution(2**7),
                encoding=[bin(i)[2:].zfill(7) for i in range(2**7)],
                num_bits=7,
            ),
        }

        for name, c in cases.items():
            with self.subTest(name):
                # Get all possible previous bits for each bit index.
                # Each entry in keys is a tuple of (bit_index, previous_bits).
                keys = []
                for bit_index in range(c.num_bits):
                    possible_previous_bits = list(
                        set([bitstring[:bit_index] for bitstring in c.encoding])
                    )
                    for previous_bits in possible_previous_bits:
                        keys.append((bit_index, previous_bits))

                values = []
                for key in keys:
                    bit_index, previous_bits = key
                    values.append(
                        reduce.get_probability_distribution_for_bit(
                            c.p, c.encoding, bit_index, c.num_bits, previous_bits
                        )
                    )

                probabilities_given_previous_bits = dict(zip(keys, values))

                # Using probabilities_given_previous_bits, compute the probability of each bitstring in c.encoding.
                # Example calculations:
                # 000: P(0, None) * P(0, 0) * P(0, 00)
                # 001: P(0, None) * P(0, 0) * P(1, 00)
                # 010: P(0, None) * P(1, 0) * P(0, 01)
                # 011: P(0, None) * P(1, 0) * P(1, 01)
                token_probabilities_from_bits: list = [
                    1 for _ in range(len(c.encoding))
                ]
                for k, v in probabilities_given_previous_bits.items():
                    bit_index, previous_bits = k
                    pr0, pr1 = v
                    for index, bitstring in enumerate(c.encoding):
                        if bitstring[:bit_index] == previous_bits:
                            if bitstring[bit_index] == "0":
                                token_probabilities_from_bits[index] *= pr0
                            else:
                                token_probabilities_from_bits[index] *= pr1

                for i in range(len(c.p)):
                    self.assertTrue(
                        np.isclose(
                            c.p[i],
                            token_probabilities_from_bits[i],
                        ),
                        f"probability for {c.encoding[i]} is off from expected: {c.p[i]},  got: {token_probabilities_from_bits[i]}",
                    )

    def test_encoders(self) -> None:
        @dataclass
        class TestCase:
            encoder_func: Callable
            tokens: List[str]

        tokens = reduce.generate_tokens("facebook/opt-1.3b")

        cases = {
            "simple encoder: encode then decode": TestCase(
                encoder_func=reduce.simple_encoder,
                tokens=tokens,
            ),
        }

        for name, c in cases.items():
            with self.subTest(name):
                encode, decode, padded_encoding, max_bit_length = c.encoder_func(
                    c.tokens
                )

                encoded = list(map(encode, c.tokens))
                self.assertNotEqual(
                    c.tokens,
                    encoded,
                    "encoded should be different from input tokens",
                )
                self.assertEqual(
                    len(set(encoded)),
                    len(c.tokens),
                    "all encodings should be unique",
                )
                self.assertEqual(
                    math.ceil(math.log2(len(c.tokens))),
                    len(max(encoded, key=len)),
                    "the longest bit string encoding should be ceil(log2(total number of tokens))",
                )

                decoded = list(map(decode, encoded))
                self.assertEqual(
                    c.tokens,
                    decoded,
                    "encoding then decoding should return the original tokens",
                )


if __name__ == "__main__":
    unittest.main()
