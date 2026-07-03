import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import json
import os

# ================= 路径配置 =================
# 1. 你的对齐数据 (用于提供清洗好的文本给模型算分)
INPUT_ALIGNED = "/data/liyuan/datasets/competition/train_aligned.csv" 
# 2. 官方原始 CSV (绝对模板，用于通过格式检查)
OFFICIAL_VALID = "/data/liyuan/datasets/competition/mediqa-eval-2026-valid.csv" 
# 3. 最终输出路径
OUTPUT_FILE = "/data/liyuan/datasets/competition/baseline_submission.csv"

# 模型设置
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

class SimpleScorer:
    def __init__(self):
        print(f"[INFO] Loading model: {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()

    def get_embeddings(self, text_list):
        # 标准的 HuggingFace Embedding 提取流程
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return F.normalize(embeddings, p=2, dim=1)

def main():
    # --- 第一步：计算分数 (在 Aligned 数据上进行) ---
    print("[INFO] Step 1: Computing scores from aligned data...")
    if not os.path.exists(INPUT_ALIGNED):
        print(f"[ERROR] Aligned file not found: {INPUT_ALIGNED}")
        return
        
    df_aligned = pd.read_csv(INPUT_ALIGNED)
    
    # 自动识别文本列名
    cand_col = 'candidate_text' if 'candidate_text' in df_aligned.columns else 'candidate'
    
    # 提取 Gold Text (取 JSON 里的第一个)
    def extract_gold(x):
        try: return json.loads(x)[0] if json.loads(x) else ""
        except: return ""
    
    df_aligned['gold_single'] = df_aligned['gold_texts'].apply(extract_gold)
    
    # 填充空值，防止报错
    df_aligned[cand_col] = df_aligned[cand_col].fillna("")
    df_aligned['gold_single'] = df_aligned['gold_single'].fillna("")

    scorer = SimpleScorer()
    
    candidates = df_aligned[cand_col].tolist()
    golds = df_aligned['gold_single'].tolist()
    sim_scores = []
    
    print(f"Computing embeddings for {len(df_aligned)} rows...")
    for i in tqdm(range(0, len(df_aligned), BATCH_SIZE)):
        batch_c = candidates[i:i+BATCH_SIZE]
        batch_g = golds[i:i+BATCH_SIZE]
        emb_c = scorer.get_embeddings(batch_c)
        emb_g = scorer.get_embeddings(batch_g)
        sim = torch.sum(emb_c * emb_g, dim=1).cpu().numpy()
        sim_scores.extend(sim)
        
    df_aligned['raw_sim'] = sim_scores

    # --- 第二步：建立分数映射表 ---
    # 使用 (Dataset + Encounter + Author + Metric + Lang) 作为唯一键
    print("[INFO] Step 2: Creating score lookup table...")
    score_map = {}
    
    for _, row in df_aligned.iterrows():
        # 业务逻辑：Disagree flag 反转
        val = 1.0 - row['raw_sim'] if row['metric'] == 'disagree_flag' else row['raw_sim']
        val = float(max(0.0, min(1.0, val))) # 限制在 0-1 之间
        
        # 构造 Key (这一步必须准确)
        key = (
            str(row['dataset']), 
            str(row['encounter_id']), 
            str(row['candidate_author_id']), 
            str(row['metric']), 
            str(row['lang'])
        )
        score_map[key] = val

    # --- 第三步：模板填充 (在 官方原始 数据上进行) ---
    print(f"[INFO] Step 3: Loading OFFICIAL template: {OFFICIAL_VALID}")
    df_official = pd.read_csv(OFFICIAL_VALID)
    
    print("[INFO] Filling scores into template...")
    new_values = []
    matched_count = 0
    missing_count = 0
    
    # 逐行遍历官方文件，只根据 ID 填入分数
    for _, row in df_official.iterrows():
        key = (
            str(row['dataset']), 
            str(row['encounter_id']), 
            str(row['candidate_author_id']), 
            str(row['metric']), 
            str(row['lang'])
        )
        
        if key in score_map:
            new_values.append(score_map[key])
            matched_count += 1
        else:
            # 如果没算出来 (极少见)，填 -1
            new_values.append(-1)
            missing_count += 1

    # 【关键操作】只覆盖 value 列，其他所有列保持原样！
    df_official['value'] = new_values
    
    # --- 第四步：保存 ---
    # 直接保存 df_official，保证所有格式、换行符、列顺序与官方完全一致
    df_official.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*30)
    print(f"[SUCCESS] Submission saved to: {OUTPUT_FILE}")
    print(f"Total Rows: {len(df_official)}")
    print(f"Matched Predictions: {matched_count}")
    print("="*30)
    
    print("\nRun Eval:")
    print(f"python mediqa_eval_script.py {OFFICIAL_VALID} {OUTPUT_FILE}")

if __name__ == "__main__":
    main()