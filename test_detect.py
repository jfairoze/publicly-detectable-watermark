import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

import crypto
import detect
import generate


class TestDetect(unittest.TestCase):
    def test_generate_then_detect_asymmetric_on_text(self) -> None:
        source = "facebook/opt-1.3b"
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            device_map="auto",
            torch_dtype="auto",
        )

        (
            generated_text,
            generated_tokens,
            pk,
            params,
            num_tokens,
            num_planted_errors,
        ) = generate.generate_text_asymmetric(
            'After the martyrdom of St. Boniface, Vergilius was made Bishop of Salzburg (766 or 767) and laboured successfully for the upbuilding of his diocese as well as for the spread of the Faith in neighbouring heathen countries, especially in Carinthia. He died at Salzburg, 27 November, 789. In 1233 he was canonized by Gregory IX. His doctrine that the earth is a sphere was derived from the teaching of ancient geographers, and his belief in the existence of the antipodes was probably influenced by the accounts which the ancient Irish voyagers gave of their journeys. This, at least, is the opinion of Rettberg ("Kirchengesch. Deutschlands", II, 236).',
            model,
            tokenizer,
            "multinomial",
            crypto.DEFAULT_MESSAGE_LENGTH,
            crypto.DEFAULT_SIGNATURE_SEGMENT_LENGTH,
            crypto.DEFAULT_BIT_SIZE,
            crypto.DEFAULT_MAX_PLANTED_ERRORS,
            None,
            None,
            None,
        )

        self.assertTrue(
            detect.search_for_asymmetric_watermark(
                pk,
                params,
                generated_text,
                crypto.DEFAULT_MESSAGE_LENGTH,
                crypto.DEFAULT_SIGNATURE_SEGMENT_LENGTH,
                crypto.DEFAULT_BIT_SIZE,
                crypto.DEFAULT_MAX_PLANTED_ERRORS,
            ),
            "Checking for a watermark in watermarked tokens should always return True",
        )

    def test_generate_then_detect_symmetric_on_bitstring(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-1.3b",
            device_map="auto",
            torch_dtype="auto",
        )

        (
            watermarked_text,
            watermarked_tokens,
            watermarked_bitstring,
        ) = generate.generate_text_symmetric(
            "The best flavor of cupcake might be chocolate",
            512,
            model,
            tokenizer,
            "multinomial",
            crypto.DEFAULT_SECURITY_PARAMETER,
        )
        self.assertTrue(
            detect.detect_symmetric_watermark(
                watermarked_bitstring, crypto.DEFAULT_SECURITY_PARAMETER
            ),
            "Checking for a watermark in watermarked tokens should always return True",
        )

        (
            plain_text,
            plain_tokens,
            plain_bitstring,
        ) = generate.generate_text_plain_with_bits(
            "The best flavor of cupcake might be chocolate",
            512,
            model,
            tokenizer,
            "multinomial",
        )

        self.assertFalse(
            detect.detect_symmetric_watermark(
                plain_bitstring, crypto.DEFAULT_SECURITY_PARAMETER
            ),
            "Checking for a watermark in plain tokens should always return False",
        )
