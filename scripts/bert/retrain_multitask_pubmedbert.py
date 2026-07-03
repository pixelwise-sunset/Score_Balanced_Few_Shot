#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


METRIC_ORDER = [
    "completeness",
    "factual-accuracy",
    "relevance",
    "disagree_flag",
    "writing-style",
    "overall",
]

EVAL_COLS_UNIQUE = [
    "dataset",
    "encounter_id",
    "lang",
    "candidate",
    "candidate_author_id",
    "metric",
]

PREDICTION_MAP_KEY = [
    "dataset",
    "encounter_id",
    "lang",
    "candidate_author_id",
    "metric",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_to_class(value: float) -> int:
    try:
        value = float(value)
    except Exception:
        value = 0.0
    if value < 0.25:
        return 0
    if value < 0.75:
        return 1
    return 2


def parse_gold_texts(raw_value, is_train: bool, use_gold: bool) -> str:
    if not use_gold:
        return ""
    try:
        gold_list = json.loads(raw_value)
    except Exception:
        gold_list = []
    if not isinstance(gold_list, list) or not gold_list:
        return ""
    gold_list = [str(item) for item in gold_list if str(item).strip()]
    if not gold_list:
        return ""
    if is_train:
        random.shuffle(gold_list)
        selected = []
        current_len = 0
        for text in gold_list:
            if current_len + len(text) >= 1000:
                break
            selected.append(text)
            current_len += len(text)
        return " ; ".join(selected)
    return " ; ".join(gold_list)


@dataclass
class Sample:
    row: pd.Series
    labels: np.ndarray


class MediqaMultiTaskDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int,
        is_train: bool,
        use_gold: bool,
        use_caption: bool,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.use_gold = use_gold
        self.use_caption = use_caption
        self.samples = []

        group_cols = ["dataset", "encounter_id", "candidate_author_id"]
        for _, group in df.groupby(group_cols, sort=False):
            labels = np.zeros(len(METRIC_ORDER), dtype=np.float32)
            for _, row in group.iterrows():
                metric = str(row["metric"])
                if metric not in METRIC_ORDER:
                    continue
                value = float(row["label"])
                if metric == "disagree_flag":
                    value = 1.0 - value
                labels[METRIC_ORDER.index(metric)] = value
            self.samples.append(Sample(row=group.iloc[0], labels=labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        row = sample.row
        candidate = str(row.get("candidate", "") if pd.notna(row.get("candidate", "")) else "")
        query = str(row.get("query_text", "") if pd.notna(row.get("query_text", "")) else "")
        caption_text = ""
        if self.use_caption:
            caption = str(row.get("image_caption", "") if pd.notna(row.get("image_caption", "")) else "")
            caption = caption.replace("**", "").replace("\n", " ").strip()
            if caption:
                caption_text = f" [IMG_INFO]: {caption}"

        text_a = f"Query: {query} | Candidate: {candidate}{caption_text}"
        text_b = parse_gold_texts(
            row.get("gold_texts", "[]"),
            is_train=self.is_train,
            use_gold=self.use_gold,
        )
        encoding = self.tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels_class = np.array([label_to_class(v) for v in sample.labels], dtype=np.int64)
        metadata = {
            "dataset": str(row.get("dataset", "")),
            "encounter_id": str(row.get("encounter_id", "")),
            "candidate_author_id": str(row.get("candidate_author_id", "")),
            "lang": str(row.get("lang", "")),
        }
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels_class, dtype=torch.long),
            "metadata": json.dumps(metadata),
        }


class MultiTaskBertClassifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 3))
                for _ in METRIC_ORDER
            ]
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = torch.stack([head(pooled) for head in self.heads], dim=1)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, 3), labels.reshape(-1))
        return {"loss": loss, "logits": logits}


def collate_batch(batch):
    metadata = [item["metadata"] for item in batch]
    tensors = {
        key: torch.stack([item[key] for item in batch])
        for key in ["input_ids", "attention_mask", "labels"]
    }
    tensors["metadata"] = metadata
    return tensors


def move_batch_to_device(batch, device):
    metadata = batch.pop("metadata")
    moved = {key: value.to(device) for key, value in batch.items()}
    moved["metadata"] = metadata
    return moved


def prediction_scores(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    class_values = torch.tensor([0.0, 0.5, 1.0], device=logits.device)
    return (probs * class_values).sum(dim=-1)


def train_one_fold(fold: int, df: pd.DataFrame, tokenizer, args, device):
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    train_ds = MediqaMultiTaskDataset(
        train_df,
        tokenizer=tokenizer,
        max_len=args.max_len,
        is_train=True,
        use_gold=args.use_gold,
        use_caption=args.use_caption,
    )
    val_ds = MediqaMultiTaskDataset(
        val_df,
        tokenizer=tokenizer,
        max_len=args.max_len,
        is_train=False,
        use_gold=args.use_gold,
        use_caption=args.use_caption,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    model = MultiTaskBertClassifier(args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    best_loss = float("inf")
    best_pred_by_key = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"fold {fold} epoch {epoch + 1}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            metadata = batch.pop("metadata")
            _ = metadata
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                outputs = model(**batch)
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach().cpu())
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        val_loss, pred_by_key = evaluate_fold_model(model, val_loader)
        print(
            f"[fold {fold}] epoch {epoch + 1}/{args.epochs} "
            f"train_loss={running_loss / max(1, len(train_loader)):.4f} val_loss={val_loss:.4f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_pred_by_key = pred_by_key
            if args.keep_model:
                fold_dir = os.path.join(args.output_dir, "checkpoints", f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(fold_dir, "model_state.pt"))

    rows = []
    for _, row in val_df.iterrows():
        key_base = (
            str(row["dataset"]),
            str(row["encounter_id"]),
            str(row["candidate_author_id"]),
            str(row["lang"]),
        )
        metric = str(row["metric"])
        score_vec = best_pred_by_key[key_base]
        value = float(score_vec[METRIC_ORDER.index(metric)])
        if metric == "disagree_flag":
            value = 1.0 - value
        out_row = row.copy()
        out_row["pred_score"] = max(0.0, min(1.0, value))
        rows.append(out_row)

    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows), best_loss


def evaluate_fold_model(model, val_loader):
    model.eval()
    total_loss = 0.0
    pred_by_key = {}
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, next(model.parameters()).device)
            metadata = batch.pop("metadata")
            outputs = model(**batch)
            total_loss += float(outputs["loss"].detach().cpu())
            scores = prediction_scores(outputs["logits"]).detach().cpu().numpy()
            for meta_json, score_vec in zip(metadata, scores):
                meta = json.loads(meta_json)
                key = (
                    str(meta["dataset"]),
                    str(meta["encounter_id"]),
                    str(meta["candidate_author_id"]),
                    str(meta["lang"]),
                )
                pred_by_key[key] = score_vec
    return total_loss / max(1, len(val_loader)), pred_by_key


def predict_dataset(model, data_loader):
    model.eval()
    device = next(model.parameters()).device
    pred_by_key = {}
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="predict", leave=False):
            batch = move_batch_to_device(batch, device)
            metadata = batch.pop("metadata")
            batch.pop("labels", None)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            scores = prediction_scores(outputs["logits"]).detach().cpu().numpy()
            for meta_json, score_vec in zip(metadata, scores):
                meta = json.loads(meta_json)
                key = (
                    str(meta["dataset"]),
                    str(meta["encounter_id"]),
                    str(meta["candidate_author_id"]),
                    str(meta["lang"]),
                )
                pred_by_key[key] = score_vec
    return pred_by_key


def build_prediction_csv(oof_df: pd.DataFrame, official_path: str, output_path: str, rater_id: str) -> None:
    official = pd.read_csv(official_path)
    score_map = {}
    for _, row in oof_df.iterrows():
        key = tuple(str(row[col]) for col in PREDICTION_MAP_KEY)
        score_map[key] = float(row["pred_score"])

    values = []
    hit_count = 0
    for _, row in official.iterrows():
        key = tuple(str(row[col]) for col in PREDICTION_MAP_KEY)
        if key in score_map:
            values.append(score_map[key])
            hit_count += 1
        else:
            values.append(0.0)
    official["value"] = values
    official["rater_id"] = rater_id
    official.to_csv(output_path, index=False)
    print(f"[write] {output_path} matched {hit_count}/{len(official)} rows", flush=True)


def run_oof(args):
    os.makedirs(args.output_dir, exist_ok=True)
    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[setup] device={device} use_gold={args.use_gold} use_caption={args.use_caption}", flush=True)
    print(f"[setup] model={args.model_name}", flush=True)

    df = pd.read_csv(args.train_csv).fillna("")
    df = df[df["lang"] == "en"].copy()
    df = df[df["metric"].isin(METRIC_ORDER)].copy()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    oof_parts = []
    fold_losses = {}
    folds = sorted(df["fold"].dropna().astype(int).unique().tolist())
    if args.max_folds > 0:
        folds = folds[: args.max_folds]
    for fold in folds:
        print(f"[start] fold={fold}", flush=True)
        fold_df, fold_loss = train_one_fold(fold, df, tokenizer, args, device)
        oof_parts.append(fold_df)
        fold_losses[str(fold)] = fold_loss
        fold_df.to_csv(os.path.join(args.output_dir, f"oof_fold_{fold}.csv"), index=False)

    oof_df = pd.concat(oof_parts, ignore_index=True)
    oof_df.to_csv(os.path.join(args.output_dir, "oof_raw.csv"), index=False)
    build_prediction_csv(
        oof_df=oof_df,
        official_path=args.official_valid_path,
        output_path=os.path.join(args.output_dir, "prediction.csv"),
        rater_id=args.exp_name,
    )
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"fold_losses": fold_losses}, f, indent=2)


def run_test(args):
    os.makedirs(args.output_dir, exist_ok=True)
    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[setup] test device={device} use_gold={args.use_gold} use_caption={args.use_caption}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    test_df = pd.read_csv(args.test_aligned_path).fillna("")
    test_df = test_df[test_df["lang"] == "en"].copy()
    test_df = test_df[test_df["metric"].isin(METRIC_ORDER)].copy()
    test_ds = MediqaMultiTaskDataset(
        test_df,
        tokenizer=tokenizer,
        max_len=args.max_len,
        is_train=False,
        use_gold=args.use_gold,
        use_caption=args.use_caption,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.predict_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    fold_predictions = []
    for fold in range(args.num_folds):
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoints", f"fold_{fold}", "model_state.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        print(f"[test] loading fold {fold}: {checkpoint_path}", flush=True)
        model = MultiTaskBertClassifier(args.model_name).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        fold_predictions.append(predict_dataset(model, test_loader))
        del model
        torch.cuda.empty_cache()

    all_keys = sorted(set().union(*[set(pred.keys()) for pred in fold_predictions]))
    avg_by_key = {}
    for key in all_keys:
        vectors = [pred[key] for pred in fold_predictions if key in pred]
        avg_by_key[key] = np.mean(np.stack(vectors, axis=0), axis=0)

    template = pd.read_csv(args.test_template_path)
    values = []
    hit_count = 0
    for _, row in template.iterrows():
        key = (
            str(row["dataset"]),
            str(row["encounter_id"]),
            str(row["candidate_author_id"]),
            str(row["lang"]),
        )
        metric = str(row["metric"])
        if key in avg_by_key and metric in METRIC_ORDER:
            value = float(avg_by_key[key][METRIC_ORDER.index(metric)])
            if metric == "disagree_flag":
                value = 1.0 - value
            values.append(max(0.0, min(1.0, value)))
            hit_count += 1
        else:
            values.append(-1.0)
    template["value"] = values
    template["rater_id"] = args.exp_name
    output_path = args.test_output_path or os.path.join(args.output_dir, "prediction_test.csv")
    template.to_csv(output_path, index=False)
    with open(os.path.join(args.output_dir, "test_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"hit_count": hit_count, "rows": len(template)}, f, indent=2)
    print(f"[write] {output_path} matched {hit_count}/{len(template)} rows", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["oof", "test"], default="oof")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--train_csv", default="datasets/aligned_en_folded.csv")
    parser.add_argument("--official_valid_path", default="datasets/mediqa-eval-2026-valid_1rater_en.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--use_gold", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_caption", type=int, choices=[0, 1], default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--keep_model", action="store_true")
    parser.add_argument("--max_folds", type=int, default=0)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--test_aligned_path", default="datasets/test_assets/test_aligned_en.csv")
    parser.add_argument("--test_template_path", default="datasets/test_assets/test_template_en.csv")
    parser.add_argument("--test_output_path")
    parser.add_argument("--predict_batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    args.use_gold = bool(args.use_gold)
    args.use_caption = bool(args.use_caption)
    if args.mode == "oof":
        run_oof(args)
    else:
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir is required for --mode test")
        run_test(args)


if __name__ == "__main__":
    main()
