#!/usr/bin/env python
"""Train metric-specific PubMedBERT with dev OOF predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import huggingface_hub.errors as _hf_errors
    import huggingface_hub.hf_file_system as _hf_file_system

    if not hasattr(_hf_errors, "BucketNotFoundError"):
        _hf_errors.BucketNotFoundError = getattr(_hf_errors, "HfHubHTTPError", Exception)
    for _name in ["HfFileSystemResolvedBucketPath", "HfFileSystemResolvedRepositoryPath"]:
        if not hasattr(_hf_file_system, _name):
            setattr(_hf_file_system, _name, str)
except Exception:
    pass

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


METRICS = ["overall", "completeness", "factual-accuracy", "relevance", "disagree_flag", "writing-style"]
PREDICT_KEY_COLS = ["dataset", "encounter_id", "lang", "candidate_author_id", "metric"]
IDX2SCORE = {0: 0.0, 1: 0.5, 2: 1.0}


def clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def short_caption(row) -> str:
    return clean(row.get("image_caption", ""))[:300]


def format_cand_only(row) -> str:
    return clean(row.get("candidate", ""))


def format_query_cand(row) -> str:
    return f"[QUERY] {clean(row.get('query_text', ''))} [CANDIDATE] {clean(row.get('candidate', ''))}"


def format_context_first(row) -> str:
    return (
        f"[IMG] {short_caption(row)} "
        f"[QUERY] {clean(row.get('query_text', ''))} "
        f"[CANDIDATE] {clean(row.get('candidate', ''))}"
    )


def format_with_gold(row) -> str:
    return (
        f"[QUERY] {clean(row.get('query_text', ''))} "
        f"[CANDIDATE] {clean(row.get('candidate', ''))} "
        f"[GOLD] {clean(row.get('gold_texts', ''))}"
    )


def get_router_fn(iiyi_strategy, wound_strategy):
    def dynamic_router(row):
        ds = clean(row.get("dataset", "")).lower()
        if "iiyi" in ds:
            return iiyi_strategy(row)
        return wound_strategy(row)

    dynamic_router.__name__ = f"Router[iiyi={iiyi_strategy.__name__}, wound={wound_strategy.__name__}]"
    return dynamic_router


def get_strategy_config(strategy_type: str):
    if strategy_type == "matched_with_gold":
        return {metric: format_with_gold for metric in METRICS}
    if strategy_type == "matched_without_gold":
        return {metric: format_query_cand for metric in METRICS}
    if strategy_type == "historical_dataset_specific":
        return {
            "completeness": format_cand_only,
            "disagree_flag": get_router_fn(format_query_cand, format_context_first),
            "factual-accuracy": get_router_fn(format_context_first, format_query_cand),
            "relevance": format_context_first,
            "writing-style": get_router_fn(format_context_first, format_cand_only),
            "overall": format_cand_only,
        }
    if strategy_type == "historical_no_image":
        return {
            "writing-style": format_cand_only,
            "completeness": format_cand_only,
            "disagree_flag": format_cand_only,
            "factual-accuracy": format_query_cand,
            "relevance": format_query_cand,
            "overall": format_query_cand,
        }
    raise ValueError(f"Unknown strategy_type: {strategy_type}")


def map_label(value) -> int:
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    if numeric < 0.25:
        return 0
    if numeric < 0.75:
        return 1
    return 2


def add_training_columns(df: pd.DataFrame, metric: str, format_fn) -> pd.DataFrame:
    df = df[(df["lang"] == "en") & (df["metric"] == metric)].copy()
    df["labels"] = df["label"].apply(map_label)
    df["text"] = df.apply(format_fn, axis=1)
    return df


def augment_train_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    augmented = [df]
    is_woundcare_context = df["dataset"].str.contains("wound", case=False, na=False).any()
    if is_woundcare_context:
        if metric == "disagree_flag":
            augmented.extend([df[df["labels"] == 2]] * 3)
        elif metric in ["completeness", "factual-accuracy", "relevance", "writing-style"]:
            augmented.extend([df[df["labels"] == 0]] * 10)
            augmented.extend([df[df["labels"] == 1]] * 10)
    return pd.concat(augmented, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)


class TokenizedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, labels=None, max_len: int = 512):
        self.encodings = tokenizer(
            [str(text) for text in texts],
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        self.labels = None if labels is None else [int(label) for label in labels]

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(label_ids, predictions),
        "f1_macro": f1_score(label_ids, predictions, average="macro"),
    }


def predict_scores(model, tokenizer, texts, batch_size: int) -> list[float]:
    dataset = TokenizedTextDataset(texts, tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict"):
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = torch.argmax(model(**batch).logits, dim=-1).detach().cpu().numpy()
            scores.extend(IDX2SCORE[int(x)] for x in pred)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy_type", required=True)
    parser.add_argument("--train_csv", default="datasets/aligned_en_folded.csv")
    parser.add_argument("--official_valid_csv", default="datasets/mediqa-eval-2026-valid_1rater_en.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--predict_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_folds", type=int, default=0)
    parser.add_argument("--keep_model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_model:
        (output_dir / "models").mkdir(exist_ok=True)

    source_df = pd.read_csv(args.train_csv)
    template = pd.read_csv(args.official_valid_csv)
    template["value"] = -1.0
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    strategy_config = get_strategy_config(args.strategy_type)
    folds = sorted(source_df["fold"].dropna().astype(int).unique().tolist())
    if args.max_folds > 0:
        folds = folds[: args.max_folds]

    strategy_map = {}
    fold_scores = {}
    for metric in METRICS:
        format_fn = strategy_config[metric]
        strategy_map[metric] = getattr(format_fn, "__name__", str(format_fn))
        metric_df = add_training_columns(source_df, metric, format_fn)
        fold_scores[metric] = {}
        print(f"===== {args.strategy_type}: {metric} using {strategy_map[metric]} =====", flush=True)

        for fold in folds:
            print(f"[start] metric={metric} fold={fold}", flush=True)
            train_df = augment_train_df(metric_df[metric_df["fold"].astype(int) != fold].copy(), metric)
            val_df = metric_df[metric_df["fold"].astype(int) == fold].copy().reset_index(drop=True)

            train_dataset = TokenizedTextDataset(train_df["text"].tolist(), tokenizer, train_df["labels"].tolist())
            val_dataset = TokenizedTextDataset(val_df["text"].tolist(), tokenizer, val_df["labels"].tolist())
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=3,
                problem_type="single_label_classification",
            )
            fold_dir = output_dir / "models" / metric / f"fold_{fold}"
            training_args = TrainingArguments(
                output_dir=str(fold_dir),
                learning_rate=args.lr,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=args.epochs,
                weight_decay=0.01,
                save_strategy="no",
                logging_steps=10,
                logging_first_step=True,
                report_to="none",
                fp16=torch.cuda.is_available(),
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics,
            )
            trainer.train()
            eval_result = trainer.evaluate()
            fold_scores[metric][str(fold)] = eval_result

            scores = predict_scores(model, tokenizer, val_df["text"].tolist(), args.predict_batch_size)
            value_by_key = {
                tuple(row[col] for col in PREDICT_KEY_COLS): score
                for (_, row), score in zip(val_df.iterrows(), scores)
            }
            metric_mask = template["metric"] == metric
            for idx, row in template.loc[metric_mask].iterrows():
                key = tuple(row[col] for col in PREDICT_KEY_COLS)
                if key in value_by_key:
                    template.at[idx, "value"] = value_by_key[key]
                    template.at[idx, "rater_id"] = args.strategy_type

            if args.keep_model:
                trainer.save_model(str(fold_dir / "final"))
            del model, trainer
            torch.cuda.empty_cache()

    missing = int((template["value"] < 0).sum())
    if missing:
        raise RuntimeError(f"Missing {missing} OOF prediction rows")

    prediction_csv = output_dir / "dev_prediction.csv"
    template.to_csv(prediction_csv, index=False)
    with (output_dir / "dev_strategy_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "strategy_type": args.strategy_type,
                "model": args.model_name,
                "hyperparameters": {
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                },
                "strategy_map": strategy_map,
                "train_csv": args.train_csv,
                "official_valid_csv": args.official_valid_csv,
                "fold_scores": fold_scores,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved dev OOF prediction CSV to {prediction_csv}")
    subprocess.run(
        [
            sys.executable,
            "eval_en.py",
            args.official_valid_csv,
            str(prediction_csv),
            str(output_dir / "dev_prediction.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
