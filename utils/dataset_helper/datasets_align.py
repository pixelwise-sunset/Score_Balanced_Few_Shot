import pandas as pd
import json
import os
from tqdm import tqdm

# ================= 1. 路径配置 =================
# PATH_CSV = "/data/liyuan/datasets/competition/mediqa-eval-2026-valid.csv"
PATH_JSON_DERMA = "datasets/original_datasets/iiyi/valid_ht.json"
PATH_IMG_DERMA_ROOT = "/data/liyuan/datasets/competition/derma/data/iiyi/images_final/images_valid"
PATH_JSON_WOUND = "datasets/original_datasets/woundcare/valid.json"
PATH_IMG_WOUND_ROOT = "/data/liyuan/datasets/competition/woundcare/dataset-challenge-mediqa-2025-wv/images_final/images_valid"
# OUTPUT_FILE = "/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/datasets/my_aligned.csv"
# =================================================

def clean_text(text):
    """清洗文本：移除换行符等"""
    if text is None:
        return ""
    text = str(text)
    return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()

def load_json_db(path):
    print(f"[INFO] Loading JSON from: {path}")
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['encounter_id']: item for item in data}

def get_gold_data(json_item, lang):
    """
    提取标准答案信息。
    返回:
    1. primary_gold: 列表中的第一个回答 (字符串)
    2. all_golds_json: 所有回答的列表 (JSON 字符串)
    """
    responses = json_item.get('responses', [])
    
    # 如果没有回答，返回空
    if not responses:
        return "", "[]"
    
    # 1. 获取所有清洗过的回答
    all_gold_list = []
    for r in responses:
        content = r.get(f"content_{lang}")
        if content:
            # 清洗每一个回答
            cleaned_content = clean_text(content)
            if cleaned_content: # 确保不存空字符串
                all_gold_list.append(cleaned_content)
    # 3. 将列表序列化为 JSON 字符串，以便存入 CSV 的一列中
    # ensure_ascii=False 保证中文正常显示
    all_golds_json = json.dumps(all_gold_list, ensure_ascii=False)
    
    return all_golds_json

def find_image_path(json_item, root_dir):
    image_ids = json_item.get('image_ids', [])
    if not image_ids:
        return None
    img_filename = image_ids[0]
    full_path = root_dir + '/' + img_filename

    return full_path

def main(df, output_file) -> pd.DataFrame:
    # 1. 加载字典
    db_derma = load_json_db(PATH_JSON_DERMA)
    db_wound = load_json_db(PATH_JSON_WOUND)
    
    # 2. 读取 CSV
    # if not os.path.exists(PATH_CSV):
    #     print(f"[ERROR] CSV not found: {PATH_CSV}")
    #     return
    df_raw = df
    print(f"[INFO] Processing {len(df_raw)} rows...")

    aligned_data = []
    stats = {"success": 0, "missing_img": 0, "missing_json": 0}

    # 3. 遍历处理
    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        dataset = row['dataset']
        enc_id = row['encounter_id']
        lang = row['lang']
        candidate_author_id = row['candidate_author_id']
        rater = row['rater_id']
        
        # 路由逻辑
        if dataset == 'iiyi':
            db = db_derma
            img_root = PATH_IMG_DERMA_ROOT
        elif dataset == 'woundcare':
            db = db_wound
            img_root = PATH_IMG_WOUND_ROOT
        else:
            continue
            
        if enc_id not in db:
            stats["missing_json"] += 1
            continue
            
        item_data = db[enc_id]
        
        # --- 文本处理 ---
        # Query
        q_title = item_data.get(f"query_title_{lang}", "")
        q_content = item_data.get(f"query_content_{lang}", "")
        query_text = clean_text(f"{q_title or ''} {q_content or ''}")
        
        # Gold Responses (核心修改点)
        all_golds_json = get_gold_data(item_data, lang)
        
        # Candidate
        candidate_text = clean_text(row['candidate'])
        
        # Image
        img_path = find_image_path(item_data, img_root)
        if img_path is None:
            stats["missing_img"] += 1
        
        new_row = {
            "dataset": dataset,
            "encounter_id": enc_id,
            "lang": lang,
            "candidate": candidate_text,
            "candidate_author_id": candidate_author_id,
            "metric": row['metric'],
            "label": row['value'],
            "query_text": query_text,
            "image_path": img_path,
            "gold_texts": all_golds_json,
            "rater_id": rater
        }
        aligned_data.append(new_row)
        stats["success"] += 1

    df_aligned = pd.DataFrame(aligned_data)
    df_aligned.to_csv(output_file, index=False, encoding='utf-8-sig')
    # print("\n" + "="*30)
    # print(f"[SUCCESS] Dataset built: {output_file}")
    # print(f"Total Rows: {len(df_aligned)}")
    # print(f"Missing Images: {stats['missing_img']}")
    # print("="*30)
    
    # # 预览一下 JSON 列是否正常
    # print("Sample of gold_texts:")
    # print(df_aligned['gold_texts'].iloc[0])
    return df_aligned

if __name__ == "__main__":
    main()