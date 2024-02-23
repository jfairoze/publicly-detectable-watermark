from typing import Callable
import numpy as np
from transformers import AutoTokenizer


def generate_tokens(
    model_name: str,
) -> list[str]:
    """Generate a list of tokens from a given model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.get_vocab()
    return list(tokens.values())


def generate_dirichlet_distribution(num_tokens: int) -> list[np.float64]:
    """Generate a random Dirichlet probability distribution over the tokens."""
    return list(np.random.dirichlet(np.ones(num_tokens), size=1)[0])


def generate_uniform_distribution(num_tokens: int) -> list[np.float64]:
    """Generate a uniform probability distribution over the tokens."""
    return [1 / np.float64(num_tokens) for _ in range(num_tokens)]


def simple_encoder(
    tokens: list[str],
) -> tuple[Callable[[str], str], Callable[[str], str], list[str], int]:
    """
    Takes in a list of tokens and returns the encoder and decoder functions, a padded encoding list, and the maximum bit length.
    """
    tokens = sorted(tokens)
    unpadded_mapping = {token: bin(counter)[2:] for counter, token in enumerate(tokens)}
    max_bit_length = len(max(unpadded_mapping.values(), key=len))

    encode_mapping = {
        token: bin(counter)[2:].zfill(max_bit_length)
        for counter, token in enumerate(tokens)
    }
    decode_mapping = {v: k for k, v in encode_mapping.items()}

    def encode(token: str) -> str:
        return encode_mapping[token]

    def decode(binstring: str) -> str:
        return decode_mapping[binstring]

    return encode, decode, list(encode_mapping.values()), max_bit_length


def get_probability_distribution_for_bit(
    p: list[np.float64],
    encoding: list[str],
    bit_index: int,
    max_bit_length: int,
    previous_bits: str,
) -> tuple[np.float64, np.float64]:
    """
    Get the probability distribution for the bit at bit_index given the previous bits.

    Returns the probability of 0 and the probability of 1.
    """

    assert bit_index < max_bit_length
    assert len(previous_bits) <= bit_index

    if previous_bits == "":
        pr0 = np.sum(
            [
                p[index] if codeword[bit_index] == "0" else 0
                for index, codeword in enumerate(encoding)
            ]
        )
        return pr0, np.subtract(1, pr0)

    # denominator is the probability the bits before bit_index match the
    # given previous_bits.
    denominator = np.sum(
        [
            p[index] if codeword[:bit_index] == previous_bits else 0
            for index, codeword in enumerate(encoding)
        ]
    )

    # pr0_numerator is the probability the bits before bit_index match the
    # given previous_bits _and_ the next bit is 0.
    pr0_numerator = np.sum(
        [
            p[index] if codeword[: (bit_index + 1)] == previous_bits + "0" else 0
            for index, codeword in enumerate(encoding)
        ]
    )

    pr0 = np.divide(pr0_numerator, denominator)
    pr1 = np.subtract(1, pr0)

    return pr0, pr1


def sample_bit(
    p: list[np.float64],
    encoding: list[str],
    bit_index: int,
    max_bit_length: int,
    previous_bits: str,
    sample_type: str,
) -> int:
    """Sample a bit at bit_index using the given sample_type."""
    pr = get_probability_distribution_for_bit(
        p, encoding, bit_index, max_bit_length, previous_bits
    )
    return sample_bit_by_sample_type(sample_type, pr)


def sample_bit_by_sample_type(
    sample_type: str, pr: tuple[np.float64, np.float64]
) -> int:
    """Sample a bit using the given sample_type and probability distribution."""
    try:
        if sample_type == "argmax":
            return int(np.argmax(pr))
        elif sample_type == "multinomial":
            return int(np.random.binomial(1, pr[1]))
    except ValueError as e:
        if np.isclose(pr[0], 0):
            pr0 = 0
            pr1 = 1
        else:
            pr0 = 1
            pr1 = 0

        return int(np.random.binomial(1, pr1))

    raise ValueError(f"sample_type {sample_type} not supported")


def sample_bits(
    p: list[np.float64], encoding: list[str], max_bit_length: int, sample_type: str
) -> str:
    """Sample max_bit_length bits from the given probability distribution. This function is only used for unit tests."""
    bits = ""
    previous_bits = ""
    for i in range(max_bit_length):
        bits += str(
            sample_bit(p, encoding, i, max_bit_length, previous_bits, sample_type)
        )
        previous_bits = bits
    return bits
