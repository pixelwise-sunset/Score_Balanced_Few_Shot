import argparse
import csv
import json
from pathlib import Path


METRIC_KEYS = [
    ("ALL", "ALL-en-ALL-mean"),
    ("disagree_flag", "ALL-en-disagree_flag-mean"),
    ("completeness", "ALL-en-completeness-mean"),
    ("factual-accuracy", "ALL-en-factual-accuracy-mean"),
    ("relevance", "ALL-en-relevance-mean"),
    ("writing-style", "ALL-en-writing-style-mean"),
    ("overall", "ALL-en-overall-mean"),
]


def read_score(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="results/en_shot_selection_probabilistic20x7")
    args = parser.parse_args()
    output_root = Path(args.output_root)

    method_labels = {
        "score_balanced_softmax20x7": "Score-balanced softmax 20x7",
        "mmr_sample20x7": "MMR sample 20x7",
    }
    context_dirs = [("with_gold", "with_gold"), ("without_gold", "no_gold")]

    rows = []
    per_rows = []
    for method_dir in sorted((output_root / "runs").glob("*")):
        if not method_dir.is_dir():
            continue
        method = method_labels.get(method_dir.name, method_dir.name)
        for context, subdir in context_dirs:
            ctx_dir = method_dir / subdir
            score_path = ctx_dir / "prediction.json"
            score = read_score(score_path)
            if score:
                row = {"context": context, "method": method, "score_json": str(score_path)}
                row.update({name: score.get(key) for name, key in METRIC_KEYS})
                rows.append(row)

            per_summary = ctx_dir / "per_bootstrap" / "per_bootstrap_summary.csv"
            if per_summary.exists():
                with per_summary.open(newline="", encoding="utf-8") as f:
                    per_rows.extend(csv.DictReader(f))

    summary_path = output_root / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["context", "method", "score_json"] + [name for name, _ in METRIC_KEYS]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    per_path = output_root / "per_bootstrap_summary.csv"
    if per_rows:
        with per_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_rows)

    print(f"Saved {len(rows)} rows to {summary_path}")
    if per_rows:
        print(f"Saved {len(per_rows)} rows to {per_path}")
    for row in rows:
        print(f"{row['context']:12s} {row['method']:32s} ALL={row.get('ALL')}")


if __name__ == "__main__":
    main()
