from utils.eval.eval_json import en_small_sample_merge, get_correlations
import pandas as pd
import json

def get_csv_scores(pred_df:pd.DataFrame, 
                   true_df:pd.DataFrame, 
                   metrics:list[str], 
                   save_path:str, 
                   en_only:bool = True):

    if en_only:
        pred_df = pred_df[pred_df['lang'] == 'en']

    merged_df = en_small_sample_merge(true_df = true_df, pred_df = pred_df)

    merged_df.to_csv("my_test/merged_df.csv", index = False)
    scores = {}
    total_score = 0
    for metric in metrics:

        per_metric_df = merged_df[merged_df['metric'] == metric]
        # print(per_metric_df)
        kendalltau, pearson, spearman, _, _, _ = get_correlations(x = per_metric_df['value_x'], y = per_metric_df['value_y'])

        mean_corr = (kendalltau + pearson + spearman) / 3
        scores[metric] = mean_corr
        total_score += mean_corr

    scores["ALL_en_ALL_mean"] = total_score / len(metrics)

    with open(save_path, 'w') as f:
        json.dump(scores, f, indent = 2)

if __name__ == "__main__":
    true_df = pd.read_csv("datasets/mediqa-eval-2026-valid.csv")
    pred_df = pd.read_csv("exp/bert_run/runs/INFER/pred_writing-style.csv")

    metrics= ['writing-style']

    save_path = "exp/bert_run/runs/INFER/score.json"

    get_csv_scores(pred_df=pred_df, true_df=true_df, metrics=metrics, save_path=save_path)