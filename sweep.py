import os
import pickle
import subprocess
import time
from typing import Dict, List, Tuple


def execute_commands(
    commands: List[str], num_trials: int = 3
) -> Dict[str, Tuple[float, int]]:
    results = {}

    try:
        with open("results.pickle", "rb") as file:
            results = pickle.load(file)
    except FileNotFoundError:
        results = {}

    for command in commands:
        if command in results:
            continue

        print(f"Executing command: {command}")

        execution_times = []
        character_counts = []

        for _ in range(num_trials):
            start_time = time.time()
            subprocess.run(command, shell=True)
            end_time = time.time()

            output_file = "wat.txt"
            if not os.path.exists(output_file):
                continue
            else:
                with open(output_file, "r") as file:
                    file_content = file.read()
                    character_count = len(file_content)
                os.remove(output_file)

            execution_time = end_time - start_time
            execution_times.append(execution_time)
            character_counts.append(character_count)

        if len(execution_times) < num_trials and len(character_counts) < num_trials:
            results[command] = (-1, -1)
        else:
            average_execution_time = sum(execution_times) / len(execution_times)
            average_character_count = sum(character_counts) / len(character_counts)

            print(f"Average execution time: {average_execution_time}")
            print(f"Average character count: {average_character_count}")

            results[command] = (average_execution_time, average_character_count)

        # Save results to pickle
        with open("results.pickle", "wb") as file:
            pickle.dump(results, file)

    return results


def get_command(
    signature_segment_length: int, bit_size: int, max_planted_errors: int
) -> str:
    return f"python generate.py --model 'mistralai/Mistral-7B-Instruct-v0.2' --prompt '<s>[INST] Write a 500 word essay about the American Civil War [/INST]' --signature-segment-length {signature_segment_length} --bit-size {bit_size} --max-planted-errors {max_planted_errors} --load-in-4bit"


# Sweep over signature segment length, bit size, and max planted errors and track how long it takes and the number of characters used each time.
commands = []
for signature_segment_length in range(5, 56):
    for bit_size in range(1, 10):
        for max_planted_errors in range(0, 20):
            commands.append(
                get_command(signature_segment_length, bit_size, max_planted_errors)
            )

results = execute_commands(commands)
print(results)

# Example results:

# "python generate.py --model 'mistralai/Mistral-7B-Instruct-v0.2' --prompt '<s>[INST] Write a 500 word essay about the American Civil War [/INST]' --signature-segment-length 16 --bit-size 1 --max-planted-errors 4": (
#         609.1530494689941,
#         6943,
#     ),

# "python generate.py --model 'mistralai/Mistral-7B-Instruct-v0.2' --prompt '<s>[INST] Write a 500 word essay about the American Civil War [/INST]' --signature-segment-length 32 --bit-size 2 --max-planted-errors 2": (
#         328.2910327911377,
#         5839,
#     ),

#  "python generate.py --model 'mistralai/Mistral-7B-Instruct-v0.2' --prompt '<s>[INST] Write a 500 word essay about the American Civil War [/INST]' --signature-segment-length 32 --bit-size 4 --max-planted-errors 2": (
#         589.2110574245453,
#         3809,
#     ),
