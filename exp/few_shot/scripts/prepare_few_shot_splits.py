import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="datasets/aligned_en_folded.csv")
    parser.add_argument("--output_dir", default="exp/few_shot/datasets")
    parser.add_argument("--valid_fold", type=int, default=4)
    parser.add_argument("--en_only", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    if args.en_only and "lang" in df.columns:
        df = df[df["lang"] == "en"].copy()

    if "fold" not in df.columns:
        raise ValueError(f"{args.input_path} must contain a 'fold' column.")

    os.makedirs(args.output_dir, exist_ok=True)
    train_df = df[df["fold"] != args.valid_fold].copy()
    val_df = df[df["fold"] == args.valid_fold].copy()

    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Saved {len(train_df)} rows to {train_path}")
    print(f"Saved {len(val_df)} rows to {val_path}")


if __name__ == "__main__":
    main()
