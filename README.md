# Publicly Detectable Watermarking for Language Models

This is the implementation for our paper [Publicly-Detectable Watermarking for Language Models](https://arxiv.org/abs/2310.18491).
All information that can be found in the paper was produced using this codebase.
For a more detailed explanation of the code, please refer to the paper.

## Colab Notebook

See [this](https://colab.research.google.com/drive/1xgUinqY0TXwVoGvB6ZwB6tR6KwLJrvnf?usp=sharing) example Colab notebook for a runnable, step-by-step guide on how to produce and detect a publicly-detectable watermark.

## Setup

Update and install build dependencies, then create a Conda environment and install Python dependencies:

```bash
sudo apt update
sudo apt install build-essential

git clone https://github.com/jfairoze/publicly-detectable-watermark
cd publicly-detectable-watermark

conda create -n wat-3.11 python=3.11
conda activate wat-3.11

pip install -r requirements.txt
```

## Generation

To generate text using a Hugging Face transformer model, run `generate.py`. Example for `asymmetric`:

```bash
python generate.py --prompt "Berkeley is known for" --model mistralai/Mistral-7B-v0.1 --gen-type asymmetric --sample-type multinomial
```

The following flags are supported:

- `--prompt` is a string for the input text.
- `--model` is a string representing the Hugging Face transformer model to use.
- `--seed` is the seed for PyTorch and Numpy pseudorandomness. It does not control cryptographic randomness and exists to allow for reproducible generation. The default seed is `0`.
- `--gen-type` is the generation algorithm to use. `plain`, `plain_with_bits`, `symmetric`, and `asymmetric` are currently supported. If unspecified, the default is `asymmetric`.
- `--sample-type` is the decoding algorithm to use when selecting the next token (or bit). `argmax`, `multinomial`, and `nucleus` are currently supported. If unspecified, the default is `multinomial`.
- `--num-tokens` is an integer number of tokens to generate. If unspecified, the default is `80`. This flag is only applicable to `plain`, `plain_with_bits`, and `symmetric` generation.

Additional flags for `--gen-type asymmetric` are below. Depending on the specific model and prompt, you may need to tune these parameters to get a successful result.

- `--sk` is the target path for the secret generation key file. If unspecified, the default is `sk.pickle`. If a file already exists at this path, it will be loaded and reused for generaiton.
- `--pk` is the target output path for the public detection key file. If unspecified, the default is `pk.pickle`.
- `--params` is the target path for the public parameters file. If unspecified, the default is `params.pickle`. If a file already exists at this path, it will be loaded and reused for generation.
- `--signature-segment-length` is an integer number of characters in each signature segment. If unspecified, the default is `16`.
- `--bit-size` is an integer number of signature bits encoded in each signature segment. If unspecified, the default is `2`.
- `--message-length` is an integer number of characters in the message. If unspecified, the default is `8`. It is recommended to set message-length to signature-segment-length // bit-size.
- `--max-planted-errors` is an upper bound on the number of errors to plant in the message. If unspecified, the default is `2`.

Additional flag for `--gen-type symmetric` is below. Depending on the specific model and prompt, you may need to tune this parameter to get a successful result.

- `--security-parameter` is the security parameter used for symmetric generation and detection. If unspecified, the default is `16`.

## Detection

To test for a watermark in a given text document, run `detect.py`. Example:

```bash
python detect.py --pk pk.pickle --params params.pickle wat.txt
```

The following flags are required to detect a watermark in an `asymmetric` generation:

- `--pk` is the input path for the public detection key.
- `--params` is the input path for the public parameters of the signature scheme.
- `--message-length` should be the same as the value used during generation.
- `--signature-segment-length` should be the same as the value used during generation.
- `--bit-size` should be the same as the value used during generation.
- `--max-planted-errors` should be the same as the value used during generation.

The following flags are required to detect a watermark in a `symmetric` generation:

- `--security-parameter` should be the same as the value used during generation.

## Benchmarking

To benchmark the different generation and detection functions, run `benchmark.py`. To run all benchmarks:

```bash
python benchmark.py --num-samples-per-input 3 --num-prompts 5
```

Or to run only the asymmetric benchmark with the default number of samples (1) and prompts (1):

```bash
python benchmark.py --asymmetric
```

The following flags are supported:

- `--plain`, `--plain-bits`, `--symmetric`, `--asymmetric` are boolean flags that control which benchmarks are run. If any flags are specified, they will be applied. If none are specified, the all benchmarks will be run.
- `--num-samples-per-input` is an integer value that controls how many times each benchmark is run for each function and input. If unspecified, the default is `1`.
- `--num-prompts` is an integer value that controls how many different prompts are generated. If unspecified, the default is `1`.
- `--model` is a string representing the Hugging Face transformer model to use. If unspecified, the default is `mistralai/Mistral-7B-v0.1`.
- `--num-tokens` is the number of tokens to generate for `plain`, `plain_with_bits`, and `symmetric` generation. If unspecified, the default is `700`. If asymmetric is set, this value is overridden by the number of tokens needed for asymmetric generation.
- `--security-parameter` is the security parameter used for symmetric generation and detection. If unspecified, the default is `16`.
- `--signature-segment-length` is an integer number of characters in each signature segment. If unspecified, the default is `16`.
- `--bit-size` is an integer number of signature bits encoded in each signature segment. If unspecified, the default is `2`.
- `--message-length` is an integer number of characters in the message. If unspecified, the default is `8`. It is recommended to set message-length to signature-segment-length // bit-size.
- `--max-planted-errors` is an upper bound on the number of errors to plant in the message. If unspecified, the default is `2`.
- `--sample-type` is the decoding algorithm to use when selecting the next token (or bit). `argmax`, `multinomial`, and `nucleus` are currently supported. If unspecified, the default is `multinomial`.

## Testing

Run all unit tests with:

```bash
python -m unittest
```
