#!/bin/bash

conda create -n wat-3.11 python=3.11

conda activate wat-3.11

pip install -r requirements.txt

if [ -z "$HF_TOKEN" ]
then
  echo "HF_TOKEN is not set. Please set it if you are using a gated model from Hugging Face."
  exit 1
fi

huggingface-cli login --token $HF_TOKEN