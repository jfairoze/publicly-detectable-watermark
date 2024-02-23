import argparse
import gc
import logging
import os
import time
import timeit
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import crypto
import detect
import generate

NUM_GENERATION_RETRIES = 3
MAX_PROMPT_LENGTH = 256
TRUNCATED_RESULT_CHAR_LIMIT = 256

# Output directories
DATA_DIRECTORY = "data"
BENCHMARK_DIRECTORY = f"{DATA_DIRECTORY}/benchmarks"
if not os.path.exists(BENCHMARK_DIRECTORY):
    os.makedirs(BENCHMARK_DIRECTORY)

# Column names
FUNCTION_COL = "Function"
PROMPT_COL = "Prompt"
NUM_TOKENS_COL = "Num Tokens"
MODEL_COL = "Model"
TOKENIZER_COL = "Tokenizer"
SAMPLE_TYPE_COL = "Sample Type"
SECURITY_PARAMETER_COL = "Sec Param"
SIGNATURE_LENGTH_COL = "Sig Len"
SIGNATURE_SEGMENT_LENGTH_COL = "Sig Seg Len"
BIT_SIZE_COL = "Bit Size"
MAX_PLANTED_ERRORS_COL = "Max Planted Errors"
NUM_PLANTED_ERRORS_COL = "Num Planted Errors"
TIME_COL = "Time (s)"
RESULT_COL = "Result"
TRUNCATED_RESULT_COL = "Truncated Result"
CREATED_AT_COL = "Created At"

logging.basicConfig(filename="logging.log", encoding="utf-8", level=logging.INFO)


def benchmark(args: argparse.Namespace) -> None:
    """Run benchmarks for the specified algorithms."""

    if not any(
        [
            args.plain,
            args.plain_bits,
            args.symmetric,
            args.asymmetric,
        ]
    ):
        # If no flags are set, run all benchmarks. If any are set, we only run the benchmarks for the flags that are set.
        args.plain = True
        args.plain_bits = True
        args.symmetric = True
        args.asymmetric = True

    benchmark_funcs = []
    if args.plain:
        benchmark_funcs.append(generate.generate_text_plain.__name__)
    if args.plain_bits:
        benchmark_funcs.append(generate.generate_text_plain_with_bits.__name__)
    if args.symmetric:
        benchmark_funcs.append(generate.generate_text_symmetric.__name__)
        benchmark_funcs.append(detect.detect_symmetric_watermark.__name__)
    if args.asymmetric:
        benchmark_funcs.append(generate.generate_text_asymmetric.__name__)
        benchmark_funcs.append(detect.search_for_asymmetric_watermark.__name__)

    inputs = get_inputs(args)

    for input in inputs:
        (
            prompt,
            num_tokens,
            model_name,
            tokenizer_name,
            sample_type,
            security_parameter,
            signature_length,
            signature_segment_length,
            bit_size,
            max_planted_errors,
        ) = input

        # Truncate the prompt to a maximum of MAX_PROMPT_LENGTH characters, rounding up to ensure words are not cut off.
        prompt_part1, prompt_rest = (
            prompt[: min(MAX_PROMPT_LENGTH, len(prompt))],
            prompt[min(MAX_PROMPT_LENGTH, len(prompt)) :],
        )
        prompt_part2 = prompt_rest.split(" ", 1)
        prompt = prompt_part1 + prompt_part2[0]

        # prompt = prompt[: min(MAX_PROMPT_LENGTH, len(prompt))]
        logging.info(f"prompt: {prompt}")

        # Read in benchmark csv and skip inputs that have already been benchmarked at least num_samples_per_input times.
        model_name_for_file = model_name.replace("/", "-")
        file_suffix = f"model={model_name_for_file}_sample-type={sample_type}_security-parameter={security_parameter}_signature-length={crypto.SIGNATURE_LENGTH}_signature-segment-length={signature_segment_length}_bit-size={bit_size}_max-planted-errors={max_planted_errors}"
        benchmark_file_name = f"benchmark_{file_suffix}.csv"
        benchmark_file_path = f"{BENCHMARK_DIRECTORY}/{benchmark_file_name}"

        existing_df = None
        num_samples_per_input = args.num_samples_per_input
        if os.path.exists(benchmark_file_path):
            logging.info(f"found existing benchmark file: {benchmark_file_path}")

            existing_df = pd.read_csv(benchmark_file_path)
            existing_num_samples_per_input = count_samples_per_input(
                existing_df, prompt, benchmark_funcs
            )
            logging.info(
                f"{existing_num_samples_per_input} existing samples for input: {input},\nbenchmark_funcs: {benchmark_funcs}",
            )
            num_samples_per_input = max(
                num_samples_per_input - existing_num_samples_per_input, 0
            )

        logging.info(f"{num_samples_per_input} samples to run")
        for _ in range(num_samples_per_input):
            if os.path.exists(benchmark_file_path):
                # Update existing_df on each iteration with latest results.
                logging.info(f"updating existing_df")
                existing_df = pd.read_csv(benchmark_file_path)

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype="auto",
            )
            logging.info(
                f"model: {model_name}, max_position_embeddings: {model.config.max_position_embeddings}, torch_dtype: {model.config.torch_dtype}"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                max_length=model.config.max_position_embeddings,
                truncation=True,
            )

            created_at = int(time.time())

            results = []

            if args.asymmetric:
                gen_asym, det_asym, num_tokens = benchmark_asymmetric(
                    prompt,
                    model,
                    tokenizer,
                    sample_type,
                    signature_length,
                    signature_segment_length,
                    bit_size,
                    max_planted_errors,
                    created_at,
                )
                results.append(gen_asym)
                results.append(det_asym)

            if args.symmetric:
                gen_sym, det_sym = benchmark_symmetric(
                    prompt,
                    num_tokens,
                    model,
                    tokenizer,
                    sample_type,
                    security_parameter,
                    created_at,
                )
                results.append(gen_sym)
                results.append(det_sym)

            if args.plain_bits:
                results.append(
                    benchmark_plain_with_bits(
                        prompt,
                        num_tokens,
                        model,
                        tokenizer,
                        sample_type,
                        created_at,
                    )
                )

            if args.plain:
                results.append(
                    benchmark_plain(
                        prompt,
                        num_tokens,
                        model,
                        tokenizer,
                        sample_type,
                        created_at,
                    )
                )

            # Create a table of the results
            input_cols = [
                PROMPT_COL,
                NUM_TOKENS_COL,
                MODEL_COL,
                TOKENIZER_COL,
                SAMPLE_TYPE_COL,
                SECURITY_PARAMETER_COL,
                SIGNATURE_LENGTH_COL,
                SIGNATURE_SEGMENT_LENGTH_COL,
                BIT_SIZE_COL,
                MAX_PLANTED_ERRORS_COL,
            ]
            df = pd.DataFrame(
                results,
                columns=[FUNCTION_COL]
                + input_cols
                + [
                    NUM_PLANTED_ERRORS_COL,
                    TIME_COL,
                    RESULT_COL,
                    CREATED_AT_COL,
                ],
            )
            df[TRUNCATED_RESULT_COL] = df[RESULT_COL].str[:TRUNCATED_RESULT_CHAR_LIMIT]

            # If there were existing results, append the new results and save the updated csv.
            if existing_df is not None:
                logging.info(f"appending to existing benchmark df")
                df = pd.concat([existing_df, df])
            logging.info(f"writing benchmark csv")
            df.to_csv(
                benchmark_file_path,
                index=False,
            )

            # Clean up before next iteration
            del model
            del tokenizer
            with torch.no_grad():
                gc.collect()
                torch.cuda.empty_cache()


def benchmark_plain(
    prompt: str,
    num_tokens: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample_type: str,
    created_at: int,
) -> list:
    """Runs benchmark for generate_text_plain."""

    logging.info("benchmarking plain...")
    func = generate.generate_text_plain

    start = timeit.default_timer()
    generated_text, generated_tokens = func(
        prompt, num_tokens, model, tokenizer, sample_type
    )
    end = timeit.default_timer()
    time = end - start

    return [
        func.__name__,
        format_text(prompt),
        num_tokens,
        model.config.name_or_path,
        tokenizer.name_or_path,
        sample_type,
        np.nan,  # No security parameter for plain
        np.nan,  # No signature length for plain
        np.nan,  # No signature segment length for plain
        np.nan,  # No bit size for plain
        np.nan,  # No ecc symbols for plain
        0,  # No planted errors for plain
        time,
        format_text(generated_text),
        created_at,
    ]


def benchmark_plain_with_bits(
    prompt: str,
    num_tokens: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample_type: str,
    created_at: int,
) -> list:
    """Runs benchmark for generate_text_plain_with_bits."""

    logging.info("benchmarking plain with bits...")
    func = generate.generate_text_plain_with_bits

    start = timeit.default_timer()
    generated_text, generated_tokens, bitstring = func(
        prompt, num_tokens, model, tokenizer, sample_type
    )
    end = timeit.default_timer()
    time = end - start

    return [
        func.__name__,
        format_text(prompt),
        num_tokens,
        model.config.name_or_path,
        tokenizer.name_or_path,
        sample_type,
        np.nan,  # No security parameter for plain with bits
        np.nan,  # No signature length for plain with bits
        np.nan,  # No signature segment length for plain with bits
        np.nan,  # No bit size for plain with bits
        np.nan,  # No ecc symbols for plain with bits
        0,  # No planted errors for plain with bits
        time,
        format_text(generated_text),
        created_at,
    ]


def benchmark_symmetric(
    prompt: str,
    num_tokens: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample_type: str,
    security_parameter: float,
    created_at: int,
) -> tuple[list, list]:
    """
    Runs benchmarks for generate_text_symmetric and detect_symmetric_watermark.

    Returns two records, one for the generation benchmarks and one for the detection benchmarks.
    """

    logging.info("benchmarking symmetric...")
    gen_func = generate.generate_text_symmetric

    gen_start = timeit.default_timer()
    generated_text, generated_tokens, bitstring = gen_func(
        prompt,
        num_tokens,
        model,
        tokenizer,
        sample_type,
        crypto.DEFAULT_SECURITY_PARAMETER,
    )
    gen_end = timeit.default_timer()
    gen_time = gen_end - gen_start

    det_func = detect.detect_symmetric_watermark
    det_start = timeit.default_timer()
    watermarked = det_func(bitstring, crypto.DEFAULT_SECURITY_PARAMETER)
    det_end = timeit.default_timer()
    det_time = det_end - det_start

    return [
        gen_func.__name__,
        format_text(prompt),
        num_tokens,
        model.config.name_or_path,
        tokenizer.name_or_path,
        sample_type,
        security_parameter,
        np.nan,  # No signature length for symmetric
        np.nan,  # No signature segment length for symmetric
        np.nan,  # No bit size for symmetric
        np.nan,  # No ecc symbols for symmetric
        0,  # No planted errors for symmetric
        gen_time,
        format_text(generated_text),
        created_at,
    ], [
        det_func.__name__,
        format_text(prompt),
        num_tokens,
        model.config.name_or_path,
        tokenizer.name_or_path,
        sample_type,
        security_parameter,
        np.nan,  # No signature length for symmetric
        np.nan,  # No signature segment length for symmetric
        np.nan,  # No bit size for symmetric
        np.nan,  # No ecc symbols for symmetric
        0,  # No planted errors for symmetric
        det_time,
        watermarked,
        created_at,
    ]


def benchmark_asymmetric(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample_type: str,
    signature_length: int,
    signature_segment_length: int,
    bit_size: int,
    max_planted_errors: int,
    created_at: int,
) -> tuple[list, list, int]:
    """
    Runs benchmarks for generate_text_asymmetric and search_for_asymmetric_watermark.

    Returns two records (one for the time it takes to generate a text sample and
    another for the time it takes to detect the watermark) and the number of tokens
    generated.
    """

    logging.info("benchmarking asymmetric...")
    gen_func = generate.generate_text_asymmetric
    succeeded_within_retries = False

    for i in range(NUM_GENERATION_RETRIES):
        logging.info(f"attempt {i}: generation starting")
        try:
            gen_start = timeit.default_timer()
            (
                generated_text,
                generated_tokens,
                pk,
                params,
                num_tokens,
                num_planted_errors,
            ) = gen_func(
                prompt,
                model,
                tokenizer,
                sample_type,
                signature_segment_length,
                bit_size,
                max_planted_errors,
                None,  # Don't need sk_path since we'll just generate a new one on each benchmark
                None,  # Don't need pk_path since it's returned directly above
                None,  # Don't need params_path since it's returned directly above
            )
            gen_end = timeit.default_timer()
            gen_time = gen_end - gen_start
        except Exception as e:
            logging.error(f"attempt {i}: generation failed: {e}")
            continue

        logging.info(f"attempt {i}: generation finished")
        succeeded_within_retries = True
        break

    # Raise an error if we didn't succeed within the number of retries.
    if not succeeded_within_retries:
        raise Exception(
            f"Failed to generate text after {NUM_GENERATION_RETRIES} retries"
        )

    logging.info("detection starting")
    det_func = detect.search_for_asymmetric_watermark
    det_start = timeit.default_timer()
    watermarked = det_func(
        pk,
        params,
        generated_text,
        signature_segment_length,
        bit_size,
        max_planted_errors,
    )
    det_end = timeit.default_timer()
    det_time = det_end - det_start
    logging.info(f"detection finished: {watermarked}")

    return (
        [
            gen_func.__name__,
            format_text(prompt),
            num_tokens,
            model.config.name_or_path,
            tokenizer.name_or_path,
            sample_type,
            np.nan,  # No security parameter for asymmetric
            signature_length,
            signature_segment_length,
            bit_size,
            max_planted_errors,
            num_planted_errors,
            gen_time,
            format_text(generated_text),
            created_at,
        ],
        [
            det_func.__name__,
            format_text(prompt),
            num_tokens,
            model.config.name_or_path,
            tokenizer.name_or_path,
            sample_type,
            np.nan,  # No security parameter for asymmetric
            signature_length,
            signature_segment_length,
            bit_size,
            max_planted_errors,
            num_planted_errors,
            det_time,
            watermarked,
            created_at,
        ],
        num_tokens,
    )


def count_samples_per_input(
    df: pd.DataFrame, prompt: str, benchmark_funcs: list[str]
) -> int:
    """
    Count the number of samples for a given input in the dataframe.
    This allows us to skip inputs that have already been benchmarked enough times based on the input arguments.

    A "sample" is defined as a set of benchmarks for all four algorithms (plain, plain with bits, symmetric, asymmetric) with the same `Created At` time.
    """

    num_samples_per_input = 0
    grouped = df[df[PROMPT_COL] == format_text(prompt)].groupby(CREATED_AT_COL)
    for name, group_df in grouped:
        if set(benchmark_funcs).issubset(group_df[FUNCTION_COL]):
            num_samples_per_input += 1

    return num_samples_per_input


def get_inputs(
    args: argparse.Namespace,
) -> List[tuple[str, int, str, str, str, int, int, int, int, int]]:
    # Skip prompts with these words.
    invalid_words = [
        "nvidia",
        "mercedes-benz",
        "cuda",
        "Aug. 28 at El Paso Montwood 4 p.m.\nSept. 5 at La Cueva 1 p.m.".lower(),  # Prompt is a list of dates
        "google",
        "microsoft",
        "facebook",
        "nintendo",
    ]

    # Get random prompts from the c4 dataset.
    dataset = load_dataset(
        "c4", "realnewslike", split="train", streaming=True, trust_remote_code=True
    )
    dataset = dataset.shuffle(buffer_size=10_000, seed=0)
    prompts = dataset.take(args.num_prompts * 10)
    prompts = [
        prompt["text"]
        for prompt in prompts
        if not any(word in prompt["text"].lower() for word in invalid_words)
    ]
    prompts = prompts[: args.num_prompts]

    inputs = [
        (
            prompt,
            args.num_tokens,
            args.model,
            args.model,
            args.sample_type,
            args.security_parameter,
            crypto.SIGNATURE_LENGTH,
            args.signature_segment_length,
            args.bit_size,
            args.max_planted_errors,
        )
        for prompt in prompts
    ]
    return inputs


def format_text(text: str) -> str:
    """Formatting to display new lines explicitly."""
    return text.replace("\n", "\\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark")
    parser.add_argument(
        "--plain",
        action=argparse.BooleanOptionalAction,
        help="flag to benchmark plain generation",
    )
    parser.add_argument(
        "--plain-bits",
        action=argparse.BooleanOptionalAction,
        help="flag to benchmark plain with bits generation",
    )
    parser.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        help="flag to benchmark symmetric generation and detection",
    )
    parser.add_argument(
        "--asymmetric",
        action=argparse.BooleanOptionalAction,
        help="flag to benchmark asymmetric generation and detection",
    )
    parser.add_argument(
        "--num-samples-per-input",
        default=1,
        type=int,
        help="the number of times to run each benchmark for each input",
    )
    parser.add_argument(
        "--num-prompts",
        default=1,
        type=int,
        help="the number of times to run each benchmark for each input",
    )
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-v0.1",
        type=str,
        help="the id of the Hugging Face model to use",
    )
    parser.add_argument(
        "--num-tokens",
        default=700,
        type=int,
        help="the number of tokens to generate for non-asymmetric schemes; if asymmetric is set, this value is overridden by the number of tokens needed for asymmetric generation",
    )
    parser.add_argument(
        "--security-parameter",
        default=crypto.DEFAULT_SECURITY_PARAMETER,
        type=float,
        help="the security parameter for symmetric generation and detection",
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
    parser.add_argument(
        "--sample-type",
        default="multinomial",
        type=str,
        help="the type of sampling to use at generation time, one of: 'argmax', 'multinomial', 'nucleus'",
        choices=("argmax", "multinomial", "nucleus"),
    )
    benchmark(parser.parse_args())
