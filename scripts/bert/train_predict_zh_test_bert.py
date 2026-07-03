#!/usr/bin/env python
"""Train Chinese metric-specific BERT models on full dev and predict Chinese test."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


METRICS = ["factual-consistency-wgold", "writing-style"]
PREDICT_KEY_COLS = ["dataset", "encounter_id", "lang", "candidate_author_id", "metric"]
IDX2SCORE = {0: 0.0, 1: 0.5, 2: 1.0}


def clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def parse_gold_text(value) -> str:
    raw = clean(value).strip()
    if not raw:
        return ""
    if raw.startswith("[") or raw.startswith("{"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return "；".join(str(item).strip() for item in parsed if str(item).strip())
            return str(parsed)
        except Exception:
            return raw
    return raw


def format_candidate(row) -> str:
    return clean(row.get("candidate", ""))


def format_query_candidate(row) -> str:
    return f"问题：{clean(row.get('query_text', ''))} [SEP] 候选回答：{clean(row.get('candidate', ''))}"


def format_candidate_gold(row) -> str:
    gold_text = parse_gold_text(row.get("gold_texts", ""))
    if len(gold_text) > 500:
        gold_text = gold_text[:500]
    return f"候选回答：{clean(row.get('candidate', ''))} [SEP] 参考回答：{gold_text}"


def get_strategy_config(strategy_type: str):
    if strategy_type == "matched_with_gold":
        return {
            "factual-consistency-wgold": format_candidate_gold,
            "writing-style": format_candidate,
        }
    if strategy_type == "matched_without_gold":
        return {
            "factual-consistency-wgold": format_query_candidate,
            "writing-style": format_candidate,
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


def load_train_df(path: str, metric: str, format_fn) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["lang"] == "zh") & (df["metric"] == metric)].copy()
    df["labels"] = df["label"].apply(map_label)
    df["text"] = df.apply(format_fn, axis=1)

    class_counts = df["labels"].value_counts()
    augmented = [df]
    if len(class_counts) > 1:
        max_count = int(class_counts.max())
        for label_value, count in class_counts.items():
            count = int(count)
            if count > 0 and count * 3 < max_count:
                multiplier = min(int(max_count / count), 10)
                if multiplier > 1:
                    augmented.extend([df[df["labels"] == label_value]] * (multiplier - 1))
    df = pd.concat(augmented, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return df[["text", "labels"]]


def load_test_metric_df(path: str, metric: str, format_fn) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["lang"] == "zh") & (df["metric"] == metric)].copy()
    df["text"] = df.apply(format_fn, axis=1)
    return df


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
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = torch.argmax(model(**batch).logits, dim=-1).detach().cpu().numpy()
            scores.extend(IDX2SCORE[int(x)] for x in pred)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy_type", required=True, choices=["matched_with_gold", "matched_without_gold"])
    parser.add_argument("--train_csv", default="datasets/aligned_zh_folded.csv")
    parser.add_argument("--test_aligned_csv", required=True)
    parser.add_argument("--test_template_csv", required=True)
    parser.add_argument("--test_gold_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="GanjinZero/coder_all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--predict_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--keep_model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    strategy_config = get_strategy_config(args.strategy_type)
    template = pd.read_csv(args.test_template_csv)
    template["value"] = -1.0

    strategy_map = {}
    for metric in METRICS:
        format_fn = strategy_config[metric]
        strategy_map[metric] = getattr(format_fn, "__name__", str(format_fn))
        print(f"===== {args.strategy_type}: {metric} using {strategy_map[metric]} =====")

        train_df = load_train_df(args.train_csv, metric, format_fn)
        train_dataset = TokenizedTextDataset(train_df["text"].tolist(), tokenizer, train_df["labels"].tolist())

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=3,
            problem_type="single_label_classification",
            trust_remote_code=True,
        )
        metric_model_dir = output_dir / "models" / metric
        training_args = TrainingArguments(
            output_dir=str(metric_model_dir),
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
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
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        trainer.train()

        test_df = load_test_metric_df(args.test_aligned_csv, metric, format_fn)
        scores = predict_scores(model, tokenizer, test_df["text"].tolist(), args.predict_batch_size)
        value_by_key = {
            tuple(row[col] for col in PREDICT_KEY_COLS): score
            for (_, row), score in zip(test_df.iterrows(), scores)
        }
        metric_mask = template["metric"] == metric
        template.loc[metric_mask, "value"] = [
            value_by_key.get(tuple(row[col] for col in PREDICT_KEY_COLS), 0.0)
            for _, row in template.loc[metric_mask].iterrows()
        ]
        template.loc[metric_mask, "rater_id"] = args.strategy_type

        if args.keep_model:
            trainer.save_model(str(metric_model_dir / "final"))
        del model, trainer
        torch.cuda.empty_cache()

    prediction_csv = output_dir / "prediction.csv"
    template.to_csv(prediction_csv, index=False)
    with (output_dir / "strategy_config.json").open("w", encoding="utf-8") as f:
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
                "test_aligned_csv": args.test_aligned_csv,
                "test_template_csv": args.test_template_csv,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved prediction CSV to {prediction_csv}")

    subprocess.run(
        [
            sys.executable,
            "eval_zh.py",
            args.test_gold_csv,
            str(prediction_csv),
            str(output_dir / "prediction.json"),
        ],
        check=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
