import pandas as pd
from sklearn.model_selection import GroupKFold
import os

# # 配置
# INPUT_FILE = "/data/liyuan/datasets/competition/train_aligned.csv"
# OUTPUT_FILE = "/data/liyuan/datasets/competition/train_5folds.csv"

# def create_folds():
#     print(f"[INFO] Reading {INPUT_FILE}...")
#     df = pd.read_csv(INPUT_FILE)
    
#     # 初始化 fold 列
#     df["fold"] = -1
    
#     # 提取唯一的 encounter_id
#     unique_ids = df["encounter_id"].unique()
#     print(f"Unique Encounters: {len(unique_ids)}")
    
#     # 5折切分
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(unique_ids)):
#         # 获取当前折的验证集 ID
#         val_ids = unique_ids[val_idx]
#         # 将这些 ID 对应的数据标记为当前 fold
#         df.loc[df["encounter_id"].isin(val_ids), "fold"] = fold
        
#     # 检查分布
#     print("\nFold Distribution:")
#     print(df.groupby("fold").size())
    
#     df.to_csv(OUTPUT_FILE, index=False)
#     print(f"\n[SUCCESS] Saved folds to {OUTPUT_FILE}")

# Assume your DataFrame is called `df`
# KEYS = ['dataset', 'lang', 'encounter_id', 'candidate_author_id']

def create_folds(df:pd.DataFrame, keys):
    KEYS = keys
    # Step 1: Create a unique group identifier
    df['group_key'] = df[KEYS].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    print(df['group_key'])
    # Step 2: Use GroupKFold to split based on this group key
    gkf = GroupKFold(n_splits=5)
    df['fold'] = -1  # Initialize fold column

    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df['group_key'])):
        df.loc[val_idx, 'fold'] = fold

    # Optional: drop the helper column if not needed
    df = df.drop(columns=['group_key'])

    return df

def main(df:pd.DataFrame, keys, output_file) -> pd.DataFrame:
    folded = create_folds(df = df , keys = keys)
    folded.to_csv(output_file, index = False)
    return folded