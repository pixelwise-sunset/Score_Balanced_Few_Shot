#!/usr/bin/env python3
"""Render the Chinese-only JBI main dev/test table figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TABLE_COLUMNS = [
    ("factual_consistency_wgold", "Factual"),
    ("writing_style", "Style"),
]

DISPLAY_NAMES = {
    "BERT": "BERT",
    "Bootstrap Random Few-shot 20x7": "Few-shot (Bootstrap)",
    "Few-shot w/ RAG": "Few-shot w/ RAG",
    "Score-balanced Softmax Few-shot 20x7": "Score-balanced Softmax",
    "Metric-wise Ensemble": "Metric-wise Ensemble",
}

ROW_ORDER = [
    "Few-shot (Bootstrap)",
    "BERT",
    "Few-shot w/ RAG",
    "Score-balanced Softmax",
    "Metric-wise Ensemble",
]

METHOD_TO_KEY = {
    "BERT": "bert",
    "Bootstrap Random Few-shot 20x7": "bootstrap_random20x7",
    "Few-shot w/ RAG": "rag_fewshot",
    "Score-balanced Softmax Few-shot 20x7": "score_balanced_softmax20x7",
    "Metric-wise Ensemble": "metricwise_ensemble",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/jbi_main_results_20260618/zh")
    parser.add_argument(
        "--output",
        default="results/jbi_main_results_20260618/zh/figures/chinese_dev_test_main_table.png",
    )
    parser.add_argument(
        "--pdf-output",
        default="results/jbi_main_results_20260618/zh/figures/chinese_dev_test_main_table.pdf",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["display_method"] = df["method"].map(DISPLAY_NAMES).fillna(df["method"])
    df["method_key"] = df["method"].map(METHOD_TO_KEY)
    order = {name: idx for idx, name in enumerate(ROW_ORDER)}
    df["_order"] = df["display_method"].map(order).fillna(999).astype(int)
    return df.sort_values(["_order", "display_method"]).reset_index(drop=True)


def format_value(value: float) -> str:
    return f"{float(value):.3f}"


def metric_name_for_column(column: str) -> str:
    return {
        "factual_consistency_wgold": "factual-consistency-wgold",
        "writing_style": "writing-style",
    }[column]


def draw_table(ax: plt.Axes, df: pd.DataFrame, title: str, source_map: dict[str, str]) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color="#2D3742", pad=14)

    headers = ["Method"] + [label for _, label in TABLE_COLUMNS]
    rows = []
    for _, row in df.iterrows():
        rows.append([row["display_method"]] + [format_value(row[column]) for column, _ in TABLE_COLUMNS])

    col_widths = [0.58, 0.21, 0.21]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.02, 1.0, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.6)

    header_color = "#203B55"
    edge_color = "#D5DEE8"
    zebra_color = "#F2F6FA"
    highlight_color = "#FFF0B8"
    text_color = "#3A4653"

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(11)
            cell.set_height(0.16)
        else:
            cell.set_facecolor(zebra_color if r % 2 == 0 else "white")
            cell.get_text().set_color(text_color)
            cell.set_height(0.13)
            if c == 0:
                cell.get_text().set_ha("left")
                cell.PAD = 0.035

    source_candidates = df[df["method_key"] != "metricwise_ensemble"].copy()
    for col_idx, (column, _) in enumerate(TABLE_COLUMNS, start=1):
        best = source_candidates[column].astype(float).max()
        selected_method = source_map.get(metric_name_for_column(column))
        for row_idx, value in enumerate(df[column].astype(float), start=1):
            method_key = df.loc[row_idx - 1, "method_key"]
            if method_key != "metricwise_ensemble" and abs(value - best) < 1e-12:
                table[(row_idx, col_idx)].get_text().set_fontweight("bold")
            if method_key != "metricwise_ensemble" and method_key == selected_method:
                table[(row_idx, col_idx)].set_facecolor(highlight_color)


def source_map_for_context(results_dir: Path, context: str) -> dict[str, str]:
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["source_maps"][context]


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output = Path(args.output)
    pdf_output = Path(args.pdf_output) if args.pdf_output else None

    required_tables = {
        ("with_gold", "dev"): results_dir / "tables/with_gold_dev.csv",
        ("with_gold", "test"): results_dir / "tables/with_gold_test.csv",
        ("without_gold", "dev"): results_dir / "tables/without_gold_dev.csv",
        ("without_gold", "test"): results_dir / "tables/without_gold_test.csv",
    }
    missing = [str(path) for path in required_tables.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required table(s): {missing}")

    tables = {key: load_table(path) for key, path in required_tables.items()}
    source_maps = {
        "with_gold": source_map_for_context(results_dir, "with_gold"),
        "without_gold": source_map_for_context(results_dir, "without_gold"),
    }

    fig = plt.figure(figsize=(16, 9.8), dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], hspace=0.36, wspace=0.1)
    axes = {
        ("with_gold", "dev"): fig.add_subplot(gs[0, 0]),
        ("with_gold", "test"): fig.add_subplot(gs[0, 1]),
        ("without_gold", "dev"): fig.add_subplot(gs[1, 0]),
        ("without_gold", "test"): fig.add_subplot(gs[1, 1]),
    }

    fig.suptitle("Chinese Dev-Selected Main Results", fontsize=22, fontweight="bold", color="#2D3742", y=0.97)
    titles = {
        ("with_gold", "dev"): "With Gold - Dev",
        ("with_gold", "test"): "With Gold - Test",
        ("without_gold", "dev"): "Without Gold - Dev",
        ("without_gold", "test"): "Without Gold - Test",
    }
    for key, ax in axes.items():
        context, _split = key
        draw_table(ax, tables[key], titles[key], source_maps[context])

    fig.text(
        0.02,
        0.025,
        "Yellow cells mark the dev-selected source used by Metric-wise Ensemble; the same source map is highlighted "
        "on test. Bold values mark the best score among non-ensemble methods within each displayed metric. "
        "Metric-wise Ensemble rows are not highlighted.",
        fontsize=10.5,
        color="#718096",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    if pdf_output:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")
    if pdf_output:
        print(f"Saved {pdf_output}")


if __name__ == "__main__":
    main()
