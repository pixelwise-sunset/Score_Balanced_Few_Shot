#!/usr/bin/env python
"""Build English dev/test tables using dev-selected metric-wise ensemble maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from eval_en import EVAL_COLS_UNIQUE, score_correlations


KEY_COLS = ["dataset", "encounter_id", "lang", "candidate", "candidate_author_id", "metric"]
OUT_COLS = KEY_COLS + ["rater_id", "value"]
METRICS = ["disagree_flag", "completeness", "factual-accuracy", "relevance", "writing-style", "overall"]
SCORE_COLUMNS = [
    ("all_en_all_mean", "ALL-en-ALL-mean"),
    ("disagree_flag_mean", "ALL-en-disagree_flag-mean"),
    ("completeness_mean", "ALL-en-completeness-mean"),
    ("factual_accuracy_mean", "ALL-en-factual-accuracy-mean"),
    ("relevance_mean", "ALL-en-relevance-mean"),
    ("writing_style_mean", "ALL-en-writing-style-mean"),
    ("overall_mean", "ALL-en-overall-mean"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selected_x_json", required=True)
    parser.add_argument("--dev_gold_csv", default="datasets/mediqa-eval-2026-valid_1rater_en.csv")
    parser.add_argument("--test_gold_csv", default="datasets/test_assets/test_gold_en.csv")

    parser.add_argument("--dev_fewshot_with_dir", required=True)
    parser.add_argument("--dev_fewshot_without_dir", required=True)
    parser.add_argument("--test_fewshot_with_dir", required=True)
    parser.add_argument("--test_fewshot_without_dir", required=True)

    parser.add_argument("--dev_bert_with", default="results/pubmedbert_dataset_specific_repro_20260601_152512_ALL")
    parser.add_argument("--dev_bert_without", default="results/pubmedbert_exp05_hybrid_no_image")
    parser.add_argument("--dev_rag_without", default="exp/few_shot/runs/main30b-shot20-no-gold-rag-v2-pubmedbert-query-mmr-oof-singleload")
    parser.add_argument("--test_bert_with", default="results/main_tables_30b_shot20_pubmedbert_test/bert_matched_with_gold")
    parser.add_argument("--test_bert_without", default="results/main_tables_30b_shot20_pubmedbert_test/bert_matched_without_gold")
    parser.add_argument("--test_rag_without", default="exp/few_shot/runs/main30b-test-no-gold-rag-v2-pubmedbert-query-mmr-singleload")
    return parser.parse_args()


def score_key(metric: str) -> str:
    return f"ALL-en-{metric}-mean"


def load_scores(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def source_entry(label: str, base_dir: str | Path) -> dict:
    base = Path(base_dir)
    return {
        "label": label,
        "prediction_csv": base / "prediction.csv",
        "score_json": base / "prediction.json",
    }


def choose_metric_sources(dev_sources: list[dict]) -> dict[str, str]:
    loaded = {source["label"]: load_scores(source["score_json"]) for source in dev_sources}
    metric_sources = {}
    for metric in METRICS:
        best_label = max(dev_sources, key=lambda source: float(loaded[source["label"]].get(score_key(metric), float("-inf"))))["label"]
        metric_sources[metric] = best_label
    return metric_sources


def load_prediction(label: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in OUT_COLS if col not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns: {missing}")
    df = df.loc[:, OUT_COLS].copy()
    if df.duplicated(KEY_COLS).any():
        raise SystemExit(f"{path} has duplicate prediction keys")
    df["source_method"] = label
    return df


def build_ensemble(
    sources: list[dict],
    metric_sources: dict[str, str],
    gold_csv: Path,
    out_dir: Path,
    rater_id: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {source["label"]: load_prediction(source["label"], source["prediction_csv"]) for source in sources}
    selected = []
    row_counts = {}
    for metric, label in metric_sources.items():
        metric_df = frames[label].loc[frames[label]["metric"] == metric, OUT_COLS].copy()
        metric_df["rater_id"] = rater_id
        selected.append(metric_df)
        row_counts[metric] = len(metric_df)
    output = pd.concat(selected, ignore_index=True).sort_values(KEY_COLS).loc[:, OUT_COLS]
    if output.duplicated(KEY_COLS).any():
        raise SystemExit(f"{out_dir} ensemble has duplicate keys")
    pred_csv = out_dir / "prediction.csv"
    score_json = out_dir / "prediction.json"
    source_map = out_dir / "source_map.json"
    output.to_csv(pred_csv, index=False)

    scores = score_correlations(pd.read_csv(gold_csv), output.drop_duplicates(subset=EVAL_COLS_UNIQUE))
    score_json.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    source_map.write_text(
        json.dumps(
            {
                "selection_split": "dev",
                "metric_sources": metric_sources,
                "sources": {source["label"]: str(source["prediction_csv"]) for source in sources},
                "row_counts": row_counts,
                "total_rows": len(output),
                "rater_id": rater_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"label": "Metric-wise Ensemble (dev-selected)", "prediction_csv": pred_csv, "score_json": score_json}


def write_table(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "score_json"] + [name for name, _ in SCORE_COLUMNS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            scores = load_scores(entry["score_json"])
            row = {"method": entry["label"], "score_json": str(entry["score_json"])}
            row.update({name: scores.get(key) for name, key in SCORE_COLUMNS})
            writer.writerow(row)


def read_selected_x(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return int(data["selected_x"])


def require_paths(entries: list[dict]) -> None:
    for entry in entries:
        for key in ["prediction_csv", "score_json"]:
            path = Path(entry[key])
            if not path.exists():
                raise SystemExit(f"Missing required file for {entry['label']}: {path}")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    selected_x = read_selected_x(Path(args.selected_x_json))
    dev_x_with = Path(args.dev_fewshot_with_dir) / f"x{selected_x}"
    dev_x_without = Path(args.dev_fewshot_without_dir) / f"x{selected_x}"
    test_x_with = Path(args.test_fewshot_with_dir) / f"x{selected_x}"
    test_x_without = Path(args.test_fewshot_without_dir) / f"x{selected_x}"

    dev_with_sources = [
        source_entry(f"Few-shot Bootstrap 20x{selected_x}", dev_x_with),
        source_entry("BERT", args.dev_bert_with),
    ]
    dev_without_sources = [
        source_entry(f"Few-shot Bootstrap 20x{selected_x}", dev_x_without),
        source_entry("BERT", args.dev_bert_without),
        source_entry("Few-shot w/ RAG", args.dev_rag_without),
    ]
    test_with_sources = [
        source_entry(f"Few-shot Bootstrap 20x{selected_x}", test_x_with),
        source_entry("BERT", args.test_bert_with),
    ]
    test_without_sources = [
        source_entry(f"Few-shot Bootstrap 20x{selected_x}", test_x_without),
        source_entry("BERT", args.test_bert_without),
        source_entry("Few-shot w/ RAG", args.test_rag_without),
    ]
    for entries in [dev_with_sources, dev_without_sources, test_with_sources, test_without_sources]:
        require_paths(entries)

    with_map = choose_metric_sources(dev_with_sources)
    without_map = choose_metric_sources(dev_without_sources)

    dev_with_ens = build_ensemble(
        dev_with_sources,
        with_map,
        Path(args.dev_gold_csv),
        out / "dev" / "ensemble_with_gold",
        "dev_selected_metricwise_with_gold",
    )
    dev_without_ens = build_ensemble(
        dev_without_sources,
        without_map,
        Path(args.dev_gold_csv),
        out / "dev" / "ensemble_without_gold",
        "dev_selected_metricwise_without_gold",
    )
    test_with_ens = build_ensemble(
        test_with_sources,
        with_map,
        Path(args.test_gold_csv),
        out / "test" / "ensemble_with_gold",
        "dev_selected_metricwise_with_gold",
    )
    test_without_ens = build_ensemble(
        test_without_sources,
        without_map,
        Path(args.test_gold_csv),
        out / "test" / "ensemble_without_gold",
        "dev_selected_metricwise_without_gold",
    )

    write_table(out / "dev" / "table_with_gold.csv", dev_with_sources + [dev_with_ens])
    write_table(out / "dev" / "table_without_gold.csv", dev_without_sources + [dev_without_ens])
    write_table(out / "test" / "table_with_gold.csv", test_with_sources + [test_with_ens])
    write_table(out / "test" / "table_without_gold.csv", test_without_sources + [test_without_ens])
    (out / "dev_selected_source_maps.json").write_text(
        json.dumps(
            {
                "selected_x": selected_x,
                "with_gold": with_map,
                "without_gold": without_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved dev/test tables to {out}")
    print(f"Selected x={selected_x}")
    print(f"With-gold source map: {with_map}")
    print(f"Without-gold source map: {without_map}")


if __name__ == "__main__":
    main()
