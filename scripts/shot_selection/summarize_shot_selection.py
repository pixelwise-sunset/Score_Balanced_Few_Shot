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
    parser.add_argument("--output_root", default="results/en_shot_selection_lowcost")
    parser.add_argument("--output_csv", default="")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_csv = Path(args.output_csv) if args.output_csv else output_root / "summary.csv"

    rows = []
    baselines = [
        (
            "with_gold",
            "Bootstrap 20x7",
            Path("exp/few_shot/runs/competition-qwen30b-bootstrap20x7-test-with-gold/prediction.json"),
        ),
        (
            "without_gold",
            "Bootstrap 20x7",
            Path("exp/few_shot/runs/competition-qwen30b-bootstrap20x7-test-no-gold/prediction.json"),
        ),
        (
            "with_gold",
            "MMR 20x1",
            Path("exp/few_shot/runs/main30b-test-with-gold-pubmedbert-mmr-singleload/prediction.json"),
        ),
        (
            "without_gold",
            "MMR 20x1",
            Path("exp/few_shot/runs/main30b-test-no-gold-pubmedbert-mmr-singleload/prediction.json"),
        ),
    ]

    for context, method, path in baselines:
        score = read_score(path)
        if score:
            row = {"context": context, "method": method, "score_json": str(path)}
            row.update({name: score.get(key) for name, key in METRIC_KEYS})
            rows.append(row)

    method_labels = {
        "random20x1": "Random 20x1",
        "similarity_topk20x1": "Similarity top-k 20x1",
        "similarity_weighted20x1": "Similarity weighted 20x1",
        "score_balanced20x1": "Score-balanced 20x1",
        "agreement_score_balanced20x1": "Agreement+score-balanced 20x1",
    }
    for method_dir in sorted((output_root / "runs").glob("*")):
        if not method_dir.is_dir():
            continue
        method = method_labels.get(method_dir.name, method_dir.name)
        for context, subdir in [("with_gold", "with_gold"), ("without_gold", "no_gold")]:
            path = method_dir / subdir / "prediction.json"
            score = read_score(path)
            if not score:
                continue
            row = {"context": context, "method": method, "score_json": str(path)}
            row.update({name: score.get(key) for name, key in METRIC_KEYS})
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["context", "method", "score_json"] + [name for name, _ in METRIC_KEYS]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")
    for row in rows:
        print(f"{row['context']:12s} {row['method']:32s} ALL={row.get('ALL')}")


if __name__ == "__main__":
    main()
