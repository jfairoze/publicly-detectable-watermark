#!/bin/bash

# large ----------------------------

python benchmark.py --num-samples-per-input 5 --num-prompts 10 --model "mistralai/Mistral-7B-v0.1" --message-length 16 --signature-segment-length 32 --bit-size 2 --max-planted-errors 2

python benchmark.py --num-samples-per-input 5 --num-prompts 10 --model "mistralai/Mistral-7B-v0.1" --message-length 16 --signature-segment-length 32 --bit-size 2 --max-planted-errors 0

# small ----------------------------

python benchmark.py --num-samples-per-input 5 --num-prompts 10 --model "mistralai/Mistral-7B-v0.1" --message-length 8 --signature-segment-length 16 --bit-size 2 --max-planted-errors 2

python benchmark.py --num-samples-per-input 5 --num-prompts 10 --model "mistralai/Mistral-7B-v0.1" --message-length 8 --signature-segment-length 16 --bit-size 2 --max-planted-errors 0

# fast -----------------------------

python benchmark.py --num-samples-per-input 5 --num-prompts 10 --model "mistralai/Mistral-7B-v0.1" --message-length 16 --signature-segment-length 16 --bit-size 1 --max-planted-errors 2

python benchmark.py --num-samples-per-input 5 --num-prompts 10 --model "mistralai/Mistral-7B-v0.1" --message-length 16 --signature-segment-length 16 --bit-size 1 --max-planted-errors 0