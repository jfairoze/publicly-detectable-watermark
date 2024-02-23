import json
import logging
import os

import openai
import pandas as pd
from dotenv import load_dotenv

import generate
from benchmark import FUNCTION_COL, PROMPT_COL, RESULT_COL

logging.basicConfig(filename="logging.log", encoding="utf-8", level=logging.INFO)

# Load environment variables from .env file
load_dotenv()
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

MODEL_NAME = "gpt-4-1106-preview"

GEN_FUNCS = [
    generate.generate_text_plain.__name__,
    generate.generate_text_plain_with_bits.__name__,
    generate.generate_text_symmetric.__name__,
    generate.generate_text_asymmetric.__name__,
]

COMPLETION_ENUMS = ["Completion 1", "Completion 2", "Completion 3", "Completion 4"]

GEN_FUNC_TO_COMPLETION_ENUM = dict(zip(GEN_FUNCS, COMPLETION_ENUMS))
COMPLETION_ENUM_TO_GEN_FUNC = dict(zip(COMPLETION_ENUMS, GEN_FUNCS))

SCORE_COL = "Score"

SCORE_PROMPT_TEMPLATE = """
You are an unbiased judge. Your job is to score text completions for a given prompt on a scale from 0 to 100 where 0 is the worst and 100 is the best. When you give a rating, give a summary of the reasons why you gave that rating in bullet points.

The prompt is: {prompt}

The following text are the four distinct text completions generated from the above prompt. Rate them all individually:

Completion 1:
{plain_result}

Completion 2:
{plain_with_bits_result}

Completion 3:
{symmetric_result}

Completion 4:
{asymmetric_result}
"""

FORMAT_INSTRUCTIONS = """
Finally, summarize the scores in a json object format where the keys are completion numbers formatted as "Completion 1", "Completion 2", "Completion 3", "Completion 4", and the values are a nested json object with two key-value pairs inside: "score" with the score you gave, and "reasoning" with the reasoning for the score you gave. Here is an example of the format:

{
    "Completion 1": {
        "score": 0,
        "reasoning": ["reason 1", "reason 2", ...]
    },
    "Completion 2": {
        "score": 0,
        "reasoning": ["reason 1", "reason 2", ...]
    },
    "Completion 3": {
        "score": 0,
        "reasoning": ["reason 1", "reason 2", ...]
    },
    "Completion 4": {
        "score": 0,
        "reasoning": ["reason 1", "reason 2", ...]
    }
}

Note the values for score and reasoning are simply placeholders above. Please fill them in with your own scores and reasonings.
"""


def score(benchmark_filenames: list[str]) -> None:
    """Score the generations from the benchmark files."""

    for filename in benchmark_filenames:
        logging.info(f"scoring {filename}")

        new_filename = filename.replace("benchmark_", "scored_benchmark_")
        if os.path.exists(new_filename):
            logging.info(f"found existing score file: {new_filename}, skipping")
            continue

        # Read in the CSV and keep only generation rows
        df = pd.read_csv(filename)
        df = df[df[FUNCTION_COL].str.startswith("generate")]

        final_dfs = []

        grouped = df.groupby(PROMPT_COL)
        for prompt, prompt_df in grouped:
            logging.info(f"prompt {prompt}")

            results = dict(
                zip(prompt_df[FUNCTION_COL].tolist(), prompt_df[RESULT_COL].tolist())
            )

            result = score_prompt(prompt, results)
            gen_func_to_score = parse_response(result)
            prompt_df[SCORE_COL] = prompt_df[FUNCTION_COL].map(gen_func_to_score)
            logging.info("prompt_df", prompt_df)

            final_dfs.append(prompt_df)

        final_df = pd.concat(final_dfs, ignore_index=True)
        final_df.to_csv(new_filename, index=False)


def parse_response(response: openai.types.chat.ChatCompletion) -> dict[str, str]:
    """Parse the response from OpenAI and return the scores for each generation function in a dict."""

    content = response.choices[0].message.content
    if content:
        content_json = json.loads(content)
    else:
        raise ValueError("content from response is None")

    gen_func_to_score = {}
    for k, v in content_json.items():
        gen_func = COMPLETION_ENUM_TO_GEN_FUNC[k]
        score = v["score"]
        gen_func_to_score[gen_func] = score

    return gen_func_to_score


def score_prompt(
    prompt: str, results: dict[str, str]
) -> openai.types.chat.ChatCompletion:
    """Score the results for a single prompt."""

    # Generate the prompt to send to OpenAI
    plain_result = results[generate.generate_text_plain.__name__]
    plain_with_bits_result = results[generate.generate_text_plain_with_bits.__name__]
    symmetric_result = results[generate.generate_text_symmetric.__name__]
    asymmetric_result = results[generate.generate_text_asymmetric.__name__]

    message_content = (
        SCORE_PROMPT_TEMPLATE.format(
            prompt=prompt,
            plain_result=plain_result,
            plain_with_bits_result=plain_with_bits_result,
            symmetric_result=symmetric_result,
            asymmetric_result=asymmetric_result,
        )
        + FORMAT_INSTRUCTIONS
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message_content,
            }
        ],
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        temperature=0,
        seed=0,
    )

    return response


if __name__ == "__main__":
    benchmark_filenames = [
        "data/benchmarks/" + file
        for file in os.listdir("data/benchmarks")
        if file.startswith("benchmark_") and file.endswith(".csv")
    ]

    score(benchmark_filenames)
