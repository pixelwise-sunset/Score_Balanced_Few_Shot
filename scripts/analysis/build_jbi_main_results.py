#!/usr/bin/env python3
"""Build the JBI English main-result directory.

The builder normalizes source predictions into one prediction key schema,
re-evaluates every copied prediction, selects metric-wise ensemble sources from
dev scores only, and applies the same source map to test predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_en import EVAL_COLS_UNIQUE, score_correlations


KEY_COLS = ["dataset", "encounter_id", "lang", "candidate", "candidate_author_id", "metric"]
OUT_COLS = KEY_COLS + ["rater_id", "value"]
METRICS = ["disagree_flag", "completeness", "factual-accuracy", "relevance", "writing-style", "overall"]
TABLE_COLUMNS = [
    ("all_en_all_mean", "ALL-en-ALL-mean"),
    ("disagree_flag", "ALL-en-disagree_flag-mean"),
    ("completeness", "ALL-en-completeness-mean"),
    ("factual_accuracy", "ALL-en-factual-accuracy-mean"),
    ("relevance", "ALL-en-relevance-mean"),
    ("writing_style", "ALL-en-writing-style-mean"),
    ("overall", "ALL-en-overall-mean"),
]
EXPECTED_ROWS = {"dev": 2898, "test": 3474}
EXPECTED_PER_METRIC = {"dev": 483, "test": 579}


@dataclass(frozen=True)
class SourceSpec:
    context: str
    split: str
    method_key: str
    label: str
    prediction_csv: Path
    score_json: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/jbi_main_results_20260618")
    parser.add_argument("--dev-gold-csv", default="datasets/mediqa-eval-2026-valid_1rater_en.csv")
    parser.add_argument("--test-gold-csv", default="datasets/test_assets/test_gold_en.csv")
    parser.add_argument(
        "--source-config",
        type=Path,
        help="Optional nested JSON overrides: context -> split -> method_key -> {prediction_csv, score_json, label}.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="write manifest/inventory and any complete normalizations")
    return parser.parse_args()


def default_specs() -> list[SourceSpec]:
    bert_root = Path("results/paper_style_bert_retrain_20260618")
    bootstrap_root = Path("results/fewshot_bootstrap20x7_original_20260617")
    score_dev_root = Path("results/en_shot_selection_probabilistic20x7/dev_oof/score_balanced_softmax20x7")
    score_test_root = Path("results/en_shot_selection_probabilistic20x7/runs/score_balanced_softmax20x7")
    rag_dev = Path("exp/few_shot/runs/main30b-shot20-no-gold-rag-v2-pubmedbert-query-mmr-oof-singleload")
    rag_test = Path("exp/few_shot/runs/main30b-test-no-gold-rag-v2-pubmedbert-query-mmr-singleload")
    boot_test_with = Path("exp/few_shot/runs/competition-qwen30b-bootstrap20x7-test-with-gold")
    boot_test_no = Path("exp/few_shot/runs/competition-qwen30b-bootstrap20x7-test-no-gold")

    specs = [
        SourceSpec("with_gold", "dev", "bert", "BERT", bert_root / "with_gold/prediction.csv", bert_root / "with_gold/prediction.json"),
        SourceSpec("with_gold", "test", "bert", "BERT", bert_root / "with_gold/prediction_test.csv", bert_root / "with_gold/prediction_test.json"),
        SourceSpec("without_gold", "dev", "bert", "BERT", bert_root / "without_gold/prediction.csv", bert_root / "without_gold/prediction.json"),
        SourceSpec("without_gold", "test", "bert", "BERT", bert_root / "without_gold/prediction_test.csv", bert_root / "without_gold/prediction_test.json"),
        SourceSpec("with_gold", "dev", "bootstrap_random20x7", "Bootstrap Random Few-shot 20x7", bootstrap_root / "with_gold_dev_oof_20x7/prediction.csv", bootstrap_root / "with_gold_dev_oof_20x7/prediction.json"),
        SourceSpec("with_gold", "test", "bootstrap_random20x7", "Bootstrap Random Few-shot 20x7", boot_test_with / "prediction.csv", boot_test_with / "prediction.json"),
        SourceSpec("without_gold", "dev", "bootstrap_random20x7", "Bootstrap Random Few-shot 20x7", bootstrap_root / "without_gold_dev_oof_20x7/prediction.csv", bootstrap_root / "without_gold_dev_oof_20x7/prediction.json"),
        SourceSpec("without_gold", "test", "bootstrap_random20x7", "Bootstrap Random Few-shot 20x7", boot_test_no / "prediction.csv", boot_test_no / "prediction.json"),
        SourceSpec("without_gold", "dev", "rag_fewshot", "Few-shot w/ RAG", rag_dev / "prediction.csv", rag_dev / "prediction.json"),
        SourceSpec("without_gold", "test", "rag_fewshot", "Few-shot w/ RAG", rag_test / "prediction.csv", rag_test / "prediction.json"),
        SourceSpec("with_gold", "dev", "score_balanced_softmax20x7", "Score-balanced Softmax Few-shot 20x7", score_dev_root / "with_gold/prediction.csv", score_dev_root / "with_gold/prediction.json"),
        SourceSpec("with_gold", "test", "score_balanced_softmax20x7", "Score-balanced Softmax Few-shot 20x7", score_test_root / "with_gold/prediction.csv", score_test_root / "with_gold/prediction.json"),
        SourceSpec("without_gold", "dev", "score_balanced_softmax20x7", "Score-balanced Softmax Few-shot 20x7", score_dev_root / "no_gold/prediction.csv", score_dev_root / "no_gold/prediction.json"),
        SourceSpec("without_gold", "test", "score_balanced_softmax20x7", "Score-balanced Softmax Few-shot 20x7", score_test_root / "no_gold/prediction.csv", score_test_root / "no_gold/prediction.json"),
    ]
    return specs


def apply_source_config(specs: list[SourceSpec], config_path: Path | None) -> list[SourceSpec]:
    if not config_path:
        return specs
    data = json.loads(config_path.read_text(encoding="utf-8"))
    updated = []
    for spec in specs:
        override = (
            data.get(spec.context, {})
            .get(spec.split, {})
            .get(spec.method_key)
        )
        if not override:
            updated.append(spec)
            continue
        updated.append(
            SourceSpec(
                context=spec.context,
                split=spec.split,
                method_key=spec.method_key,
                label=override.get("label", spec.label),
                prediction_csv=Path(override["prediction_csv"]),
                score_json=Path(override["score_json"]) if override.get("score_json") else spec.score_json,
            )
        )
    return updated


def gold_for_split(split: str, dev_gold: Path, test_gold: Path) -> Path:
    return dev_gold if split == "dev" else test_gold


def method_dir(out: Path, spec: SourceSpec) -> Path:
    return out / spec.context / spec.split / spec.method_key


def load_source_prediction(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in OUT_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df.loc[:, OUT_COLS].copy()
    conflicts = df.groupby(KEY_COLS, dropna=False)["value"].nunique().reset_index()
    bad = conflicts[conflicts["value"] > 1]
    if not bad.empty:
        raise ValueError(f"{path} has {len(bad)} duplicate keys with conflicting values")
    df = df.drop_duplicates(KEY_COLS, keep="first").sort_values(KEY_COLS).reset_index(drop=True)
    expected = EXPECTED_ROWS[split]
    if len(df) != expected:
        raise ValueError(f"{path} normalized to {len(df)} rows; expected {expected}")
    counts = df["metric"].value_counts().to_dict()
    for metric in METRICS:
        if counts.get(metric, 0) != EXPECTED_PER_METRIC[split]:
            raise ValueError(f"{path} has {counts.get(metric, 0)} rows for {metric}; expected {EXPECTED_PER_METRIC[split]}")
    if df.duplicated(KEY_COLS).any():
        raise ValueError(f"{path} still has duplicate keys after normalization")
    return df


def evaluate(gold_csv: Path, pred_df: pd.DataFrame, out_json: Path) -> dict:
    scores = score_correlations(pd.read_csv(gold_csv), pred_df.drop_duplicates(subset=EVAL_COLS_UNIQUE))
    out_json.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores


def normalize_source(spec: SourceSpec, out: Path, dev_gold: Path, test_gold: Path) -> dict:
    out_dir = method_dir(out, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = load_source_prediction(spec.prediction_csv, spec.split)
    pred_csv = out_dir / "prediction.csv"
    score_json = out_dir / "prediction.json"
    pred.to_csv(pred_csv, index=False)
    scores = evaluate(gold_for_split(spec.split, dev_gold, test_gold), pred, score_json)
    provenance = {
        "context": spec.context,
        "split": spec.split,
        "method_key": spec.method_key,
        "method_label": spec.label,
        "source_prediction_csv": str(spec.prediction_csv),
        "source_score_json": str(spec.score_json) if spec.score_json else None,
        "normalized_prediction_csv": str(pred_csv),
        "normalized_score_json": str(score_json),
        "rows": len(pred),
        "metric_counts": pred["metric"].value_counts().sort_index().to_dict(),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"spec": spec, "prediction_csv": pred_csv, "score_json": score_json, "scores": scores}


def score_key(metric: str) -> str:
    return f"ALL-en-{metric}-mean"


def choose_metric_sources(dev_entries: list[dict]) -> dict[str, str]:
    source_map = {}
    for metric in METRICS:
        source_map[metric] = max(
            dev_entries,
            key=lambda entry: float(entry["scores"].get(score_key(metric), float("-inf"))),
        )["spec"].method_key
    return source_map


def build_ensemble(
    context: str,
    split: str,
    entries: list[dict],
    source_map: dict[str, str],
    out: Path,
    gold_csv: Path,
) -> dict:
    frames = {
        entry["spec"].method_key: pd.read_csv(entry["prediction_csv"]).loc[:, OUT_COLS]
        for entry in entries
    }
    selected = []
    row_counts = {}
    for metric, method_key in source_map.items():
        metric_df = frames[method_key][frames[method_key]["metric"] == metric].copy()
        metric_df["rater_id"] = f"jbi_dev_selected_metricwise_{context}"
        selected.append(metric_df)
        row_counts[metric] = len(metric_df)
    pred = pd.concat(selected, ignore_index=True).sort_values(KEY_COLS).reset_index(drop=True)
    if len(pred) != EXPECTED_ROWS[split] or pred.duplicated(KEY_COLS).any():
        raise ValueError(f"{context}/{split} ensemble failed key validation")

    ens_spec = SourceSpec(context, split, "metricwise_ensemble", "Metric-wise Ensemble", Path(""))
    out_dir = method_dir(out, ens_spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "prediction.csv"
    score_json = out_dir / "prediction.json"
    pred.to_csv(pred_csv, index=False)
    scores = evaluate(gold_csv, pred, score_json)
    source_data = {
        "selection_split": "dev",
        "evaluation_split": split,
        "context": context,
        "metric_sources": source_map,
        "row_counts": row_counts,
        "sources": {entry["spec"].method_key: str(entry["prediction_csv"]) for entry in entries},
    }
    (out_dir / "source_map.json").write_text(json.dumps(source_data, indent=2), encoding="utf-8")
    (out_dir / "provenance.json").write_text(
        json.dumps({**source_data, "prediction_csv": str(pred_csv), "score_json": str(score_json)}, indent=2),
        encoding="utf-8",
    )
    return {"spec": ens_spec, "prediction_csv": pred_csv, "score_json": score_json, "scores": scores}


def write_table(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + [name for name, _ in TABLE_COLUMNS])
        writer.writeheader()
        for entry in entries:
            row = {"method": entry["spec"].label}
            for column, key in TABLE_COLUMNS:
                row[column] = entry["scores"].get(key)
            writer.writerow(row)


def draw_figure(out: Path, table_paths: list[Path]) -> None:
    rows = []
    for table_path in table_paths:
        if table_path.exists():
            df = pd.read_csv(table_path)
            for _, row in df.iterrows():
                rows.append({"table": table_path.stem, "method": row["method"], "score": row["all_en_all_mean"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(len(table_paths), 1, figsize=(10, 2.2 * len(table_paths)), dpi=180)
    if len(table_paths) == 1:
        axes = [axes]
    for ax, table_path in zip(axes, table_paths):
        sub = df[df["table"] == table_path.stem]
        ax.barh(sub["method"], sub["score"], color="#3B6EA8")
        ax.set_title(table_path.stem.replace("_", " "))
        ax.set_xlim(0, max(0.6, float(df["score"].max()) + 0.05))
        ax.grid(axis="x", color="#D9DEE7", linewidth=0.7)
        ax.set_xlabel("ALL-en-ALL-mean")
    fig.tight_layout()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "main_results.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    dev_gold = Path(args.dev_gold_csv)
    test_gold = Path(args.test_gold_csv)
    out.mkdir(parents=True, exist_ok=True)

    specs = apply_source_config(default_specs(), args.source_config)
    missing = []
    available = []
    for spec in specs:
        if not spec.prediction_csv.exists():
            missing.append({"method": spec.method_key, "context": spec.context, "split": spec.split, "path": str(spec.prediction_csv)})
        else:
            available.append(spec)

    if missing and not args.allow_missing:
        (out / "manifest.json").write_text(
            json.dumps({"status": "blocked_missing_sources", "missing_sources": missing}, indent=2),
            encoding="utf-8",
        )
        raise SystemExit(f"Missing {len(missing)} required source prediction CSVs; see {out / 'manifest.json'}")

    normalized: list[dict] = []
    errors = []
    for spec in available:
        try:
            normalized.append(normalize_source(spec, out, dev_gold, test_gold))
        except Exception as exc:
            errors.append({"method": spec.method_key, "context": spec.context, "split": spec.split, "path": str(spec.prediction_csv), "error": str(exc)})
            if not args.allow_missing:
                raise

    by_context_split: dict[tuple[str, str], list[dict]] = {}
    for entry in normalized:
        by_context_split.setdefault((entry["spec"].context, entry["spec"].split), []).append(entry)

    ensemble_entries = []
    source_maps = {}
    for context in ["with_gold", "without_gold"]:
        dev_entries = by_context_split.get((context, "dev"), [])
        test_entries = by_context_split.get((context, "test"), [])
        method_sets_match = {e["spec"].method_key for e in dev_entries} == {e["spec"].method_key for e in test_entries}
        expected_methods = {"bert", "bootstrap_random20x7", "score_balanced_softmax20x7"}
        if context == "without_gold":
            expected_methods.add("rag_fewshot")
        if {e["spec"].method_key for e in dev_entries} == expected_methods and method_sets_match:
            source_map = choose_metric_sources(dev_entries)
            source_maps[context] = source_map
            ensemble_entries.append(build_ensemble(context, "dev", dev_entries, source_map, out, dev_gold))
            ensemble_entries.append(build_ensemble(context, "test", test_entries, source_map, out, test_gold))

    all_entries = normalized + ensemble_entries
    table_paths = []
    for context in ["with_gold", "without_gold"]:
        for split in ["dev", "test"]:
            entries = [entry for entry in all_entries if entry["spec"].context == context and entry["spec"].split == split]
            if entries:
                table_path = out / "tables" / f"{context}_{split}.csv"
                write_table(table_path, entries)
                table_paths.append(table_path)
    draw_figure(out, table_paths)

    manifest = {
        "status": "complete" if not missing and not errors and len(source_maps) == 2 else "incomplete",
        "output_dir": str(out),
        "dev_gold_csv": str(dev_gold),
        "test_gold_csv": str(test_gold),
        "source_config": str(args.source_config) if args.source_config else None,
        "missing_sources": missing,
        "normalization_errors": errors,
        "source_maps": source_maps,
        "normalized_sources": [
            {
                "context": entry["spec"].context,
                "split": entry["spec"].split,
                "method": entry["spec"].method_key,
                "prediction_csv": str(entry["prediction_csv"]),
                "score_json": str(entry["score_json"]),
            }
            for entry in normalized
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved JBI main-results manifest to {out / 'manifest.json'}")
    print(f"Status: {manifest['status']}")
    if missing:
        print(f"Missing sources: {len(missing)}")
    if errors:
        print(f"Normalization errors: {len(errors)}")


if __name__ == "__main__":
    main()
