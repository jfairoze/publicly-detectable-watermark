import hashlib
import random
import unittest
from dataclasses import dataclass

import numpy as np
from bls.scheme import *
from scipy.stats import kstest

import crypto


class TestCrypto(unittest.TestCase):
    def test_unkeyed_hash_to_float_between_zero_and_one(self) -> None:
        @dataclass
        class TestCase:
            input_bytes: bytes

        # Generate 100 random bytearrays of random length between 1 and 100
        bytearrs = [random.randbytes(random.randint(1, 100)) for _ in range(100)]
        cases = {bytearr: TestCase(input_bytes=bytearr) for bytearr in bytearrs}

        for name, c in cases.items():
            with self.subTest(name):
                output = crypto.unkeyed_hash_to_float(c.input_bytes)
                self.assertLessEqual(0, output)
                self.assertLessEqual(output, 1)

    def test_unkeyed_hash_to_float_is_uniform(self) -> None:
        # Generate 1 million random bytearrays of random length between 1 and 100
        bytearrs = [random.randbytes(random.randint(1, 100)) for _ in range(1000000)]
        hashes = [crypto.unkeyed_hash_to_float(bytearr) for bytearr in bytearrs]
        average = sum(hashes) / len(hashes)
        self.assertTrue(
            np.isclose(average, 0.5, atol=0.01),
            f"the average is not within 0.01 of 0.5: average={average}",
        )

        # Run the Kolmorogov-Smirnov test to check if the hashes are uniformly distributed.
        ks_statistic, p_value = kstest(hashes, "uniform")
        self.assertGreaterEqual(
            p_value,
            0.01,
            f"the hashes do not appear to be uniformly distributed: ks_statistic={ks_statistic}, p_value={p_value}",
        )

    def test_unkeyed_hash_to_float_is_deterministic(self) -> None:
        r = "00000000100100010000110010011100000000000000110001100001010010010100010100101011000000000000010000000001011101100000000000111111000000010011010000000100010101100000000000000110001010100100000100000000000100000010000001111"
        j = 221
        input_bytes = bytes(r, "utf-8") + bytes(bin(j), "utf-8")

        hash1 = crypto.unkeyed_hash_to_float(input_bytes)
        for i in range(1000):
            with self.subTest(i):
                hash2 = crypto.unkeyed_hash_to_float(input_bytes)
                self.assertEqual(hash1, hash2)

    def test_reedsolo_error_correction(self) -> None:
        # use bytes_to_binary_codeword and binary_codeword_to_bytes to encode and decode in unit test
        for max_planted_errors in range(2, 32, 2):
            for i in range(100):
                with self.subTest(f"{max_planted_errors}, {i}"):
                    input_bytes = random.randbytes(random.randint(32, 64))
                    codeword = crypto.bytes_to_binary_codeword(
                        input_bytes, max_planted_errors
                    )

                    # flip DEFAULT_MAX_PLANTED_ERRORS bits in the codeword
                    old_codeword = codeword
                    codeword_len = len(codeword)
                    indices = random.sample(
                        range(codeword_len), max_planted_errors // 2
                    )

                    for i in indices:
                        codeword = (
                            codeword[:i]
                            + ("1" if codeword[i] == "0" else "0")
                            + codeword[i + 1 :]
                        )

                    for i in range(codeword_len):
                        if i in indices:
                            self.assertNotEqual(codeword[i], old_codeword[i])
                        else:
                            self.assertEqual(codeword[i], old_codeword[i])

                    output_bytes = crypto.binary_codeword_to_bytes(
                        codeword, max_planted_errors
                    )
                    self.assertEqual(input_bytes, output_bytes)

    def test_bls_rsc_combination(self) -> None:
        for max_planted_errors in range(10):
            for i in range(100):
                with self.subTest(f"{max_planted_errors}, {i}"):
                    sk, pk, params = crypto.bls_generate_openssl()
                    message = random.randbytes(random.randint(32, 64))
                    signature = crypto.sign_and_encode_openssl(
                        sk, message, params, max_planted_errors
                    )
                    self.assertTrue(
                        crypto.decode_and_verify_openssl(
                            pk, message, signature, params, max_planted_errors
                        )
                    )

    def test_bls_openssl(self) -> None:
        for i in range(100):
            with self.subTest(i):
                sk, pk, params = crypto.bls_generate_openssl()
                message = random.randbytes(random.randint(32, 64))
                signature = crypto.bls_sign_openssl(message, sk, params)
                self.assertTrue(len(signature) == crypto.SIGNATURE_LENGTH // 8)
                self.assertTrue(
                    crypto.bls_verify_openssl(message, signature, pk, params)
                )

    def test_bls_rsc_combination_with_hashing_openssl(self) -> None:
        for max_planted_errors in range(10):
            for i in range(100):
                with self.subTest(f"{max_planted_errors}, {i}"):
                    sk, pk, params = crypto.bls_generate_openssl()
                    message = random.randbytes(random.randint(32, 64))
                    message_hash = hashlib.sha256(message).digest()
                    signature = crypto.sign_and_encode_openssl(
                        sk, message_hash, params, max_planted_errors
                    )
                    self.assertTrue(
                        crypto.decode_and_verify_openssl(
                            pk,
                            message_hash,
                            signature,
                            params,
                            max_planted_errors,
                        )
                    )
