import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from benchmark import (
    BIT_SIZE_COL,
    FUNCTION_COL,
    MAX_PLANTED_ERRORS_COL,
    MODEL_COL,
    NUM_PLANTED_ERRORS_COL,
    PROMPT_COL,
    SIGNATURE_SEGMENT_LENGTH_COL,
    TIME_COL,
)
from score import SCORE_COL

GENERATE_AND_DETECT_PALETTE = "muted"
ERR_KWS = {"linewidth": 1.2}
FIGSIZE = (12, 4)


def plot_generate_and_detect_runtimes(filenames: list[str]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    plot_generate_runtimes(filenames, ax1)
    plot_detect_runtimes(filenames, ax2)

    # Only keep the legend with all four labels and move it to the top right
    ax2.legend_ = ax1.legend_
    ax1.legend_.remove()
    sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))

    # Save the figure to a file without cutting off the legend
    fig.savefig("generate_and_detect_runtimes", bbox_inches="tight", dpi=300)

    plt.tight_layout()
    plt.show()


def plot_generate_runtimes(filenames: list[str], ax: Axes) -> None:
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        prompts = df[PROMPT_COL].unique()
        df[PROMPT_COL] = df[PROMPT_COL].map(
            {prompt: i for i, prompt in enumerate(prompts)},
        )
        df = df[df[FUNCTION_COL].str.startswith("generate")]

        # Remove generate_text prefix from each entry in the Function column of df
        df[FUNCTION_COL] = df[FUNCTION_COL].map(lambda s: s[14:])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(df.head())

    sns.barplot(
        data=df,
        x=PROMPT_COL,
        y=TIME_COL,
        hue=FUNCTION_COL,
        dodge=True,
        palette=GENERATE_AND_DETECT_PALETTE,
        ax=ax,
        errorbar=("pi", 95),
        err_kws=ERR_KWS,
        capsize=0.4,
        linewidth=0,
    )
    model = df[MODEL_COL].unique()[0]
    ax.set_title("Generation Runtimes for " + model)
    ax.grid(True)

    # Move the legend outside of the plot to the top right
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


def plot_detect_runtimes(filenames: list[str], ax: Axes) -> None:
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        prompts = df[PROMPT_COL].unique()
        df[PROMPT_COL] = df[PROMPT_COL].map(
            {prompt: i for i, prompt in enumerate(prompts)},
        )
        df = df[df[FUNCTION_COL].str.startswith("detect")]

        # For each value in the FUNCTION_COL column of df, where the value is something like "detect_asymmetric_watermark" or "detect_symmetric_watermark", remove the "detect_" prefix and "_watermark*" suffix using regex.
        df[FUNCTION_COL] = df[FUNCTION_COL].str.replace(
            r"^detect_(asymmetric|symmetric)_watermark.*",
            r"\1",
            regex=True,
        )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(df.head())

    sns.barplot(
        data=df,
        x=PROMPT_COL,
        y=TIME_COL,
        hue=FUNCTION_COL,
        dodge=True,
        palette=GENERATE_AND_DETECT_PALETTE,
        ax=ax,
        errorbar=("pi", 95),
        err_kws=ERR_KWS,
        capsize=0.2,
        width=0.4,
        linewidth=0,
    )
    ax.set_yscale("log")
    ax.set_ylabel(TIME_COL + " (log)")
    model = df[MODEL_COL].unique()[0]
    ax.set_title("Detection Runtimes for " + model)
    ax.grid(True)

    # Move the legend outside of the plot to the top right
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


def plot_generate_runtimes_asymmetric(filenames: list[str]) -> None:
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        prompts = df[PROMPT_COL].unique()
        df[PROMPT_COL] = df[PROMPT_COL].map(
            {prompt: i for i, prompt in enumerate(prompts)},
        )
        # Keep only the rows where the function is asymmetric generation
        df = df[df[FUNCTION_COL].str.startswith("generate")]
        df = df[df[FUNCTION_COL].str.contains("asymmetric")]

        # Remove generate_text prefix from each entry in the Function column of df
        df[FUNCTION_COL] = df[FUNCTION_COL].map(lambda s: s[14:])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(
        by=[
            PROMPT_COL,
            SIGNATURE_SEGMENT_LENGTH_COL,
            BIT_SIZE_COL,
            MAX_PLANTED_ERRORS_COL,
        ]
    )
    print(df.head())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(
        data=df,
        x=PROMPT_COL,
        y=TIME_COL,
        hue=df[
            [SIGNATURE_SEGMENT_LENGTH_COL, BIT_SIZE_COL, MAX_PLANTED_ERRORS_COL]
        ].apply(tuple, axis=1),
        dodge=True,
        palette="Paired",
        ax=ax,
        errorbar=("pi", 95),
        err_kws=ERR_KWS,
        capsize=0.4,
        linewidth=0,
    )
    model = df[MODEL_COL].unique()[0]
    ax.legend(title="ℓ, β, γ")
    ax.set_title("Asymmetric Generation Runtimes for " + model)
    ax.grid(True)

    # Move the legend outside of the plot to the top right
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # Save the figure to a file without cutting off the legend
    fig.savefig("generate_runtimes_asymmetric", bbox_inches="tight", dpi=300)

    plt.show()


def plot_generate_quality_scores(filenames: list[str]) -> None:
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        prompts = df[PROMPT_COL].unique()
        df[PROMPT_COL] = df[PROMPT_COL].map(
            {prompt: i for i, prompt in enumerate(prompts)},
        )
        # Keep only the rows where the function starts with generate
        df = df[df[FUNCTION_COL].str.startswith("generate")]

        # Remove generate_text prefix from each entry in the Function column of df
        df[FUNCTION_COL] = df[FUNCTION_COL].map(lambda s: s[14:])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Convert the FUNCTION_COL column to a categorical type with the custom order
    order = ["asymmetric", "symmetric", "plain_with_bits", "plain"]
    df[FUNCTION_COL] = pd.Categorical(df[FUNCTION_COL], categories=order, ordered=True)

    df = df.sort_values(
        by=[
            FUNCTION_COL,
            SIGNATURE_SEGMENT_LENGTH_COL,
            BIT_SIZE_COL,
            MAX_PLANTED_ERRORS_COL,
        ]
    )
    combined_col = FUNCTION_COL + ", (ℓ, β, γ)"
    df[combined_col] = (
        df[FUNCTION_COL].astype(str)
        + ",\n("
        + df[[SIGNATURE_SEGMENT_LENGTH_COL, BIT_SIZE_COL, MAX_PLANTED_ERRORS_COL]]
        .astype(str)
        .agg(", ".join, axis=1)
        + ")"
    )
    # Replace all instances of (nan, nan, nan) in combined_col with a single nan
    df[combined_col] = df[combined_col].str.replace(
        r",\n\(nan, nan, nan\)", "", regex=True
    )
    # Replace all instances of .0 in combined_col with empty string
    df[combined_col] = df[combined_col].str.replace(r"\.0", "", regex=True)
    print(df.head())

    # Generate a custom color palette to match generate_and_detect_runtimes graph
    blue_palette = sns.color_palette("Blues", n_colors=10)
    orange = sns.color_palette("Oranges", n_colors=1)[0]
    green = sns.color_palette("Greens", n_colors=1)[0]
    red = sns.color_palette("Reds", n_colors=1)[0]
    colors = blue_palette[2:8] + [orange, green, red]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    grouped = df.groupby(combined_col, sort=False)

    mean_scores = []
    for i, (name, group_df) in enumerate(grouped):
        sns.barplot(
            data=group_df,
            x=combined_col,
            y=SCORE_COL,
            dodge=True,
            color=colors[i],
            ax=ax,
            errorbar=("pi", 95),
            err_kws=ERR_KWS,
            capsize=0.2,
            linewidth=0,
            width=0.4,
        )

        print("Mean Score", group_df[SCORE_COL].mean())
        mean_scores.append(group_df[SCORE_COL].mean())
    print(sorted(mean_scores))

    model = df[MODEL_COL].unique()[0]
    ax.set_title("GPT-4 Generation Quality Scores for " + model)
    ax.grid(True)
    ax.set_ylim([0, 100])

    # Save the figure to a file without cutting off the legend
    fig.savefig("generate_quality_scores", bbox_inches="tight", dpi=300)

    plt.show()


if __name__ == "__main__":
    sns.set_style("darkgrid")

    # Change directories and filenames to the benchmark files you would like to plot.

    benchmark_filenames_mistral = [
        "data_mistral/benchmarks/" + filename
        for filename in os.listdir("data_mistral/benchmarks")
        if filename.startswith("benchmark_") and filename.endswith("csv")
    ]
    plot_generate_and_detect_runtimes(benchmark_filenames_mistral)

    scored_benchmark_filenames_mistral = [
        "data_mistral/benchmarks/" + filename
        for filename in os.listdir("data_mistral/benchmarks")
        if filename.startswith("scored_") and filename.endswith("csv")
    ]
    plot_generate_quality_scores(scored_benchmark_filenames_mistral)

    benchmark_filenames_opt = [
        "data_opt/benchmarks/" + filename
        for filename in os.listdir("data_opt/benchmarks")
        if filename.startswith("benchmark_") and filename.endswith("csv")
    ]
    plot_generate_runtimes_asymmetric(benchmark_filenames_opt)
