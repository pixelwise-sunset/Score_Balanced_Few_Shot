from exp.few_shot.scripts.make_shot import *
import draccus
from draccus import field
from dataclasses import dataclass
import pandas as pd
import json
from utils.dataset_helper.get_data import *
from typing import List
import ast

@dataclass
class config:
    split_from:str = ""
    infer_path:str = ""
    output_path:str = ""
    sample_n:int = None #how many samples to do inference
    metrics:str = "[]"
    shot_num:int = 7
    en_only:bool = False


@draccus.wrap()
def main(cfg:config):
    #define the metrics
    metrics = ast.literal_eval(cfg.metrics)
    shot_df = split_shot(split_from=cfg.split_from, 
                         shot_num = cfg.shot_num, 
                         metrics = metrics, 
                         save_path=None)
    
    if len(metrics) == 1:
        shot_df = pd.read_csv(cfg.split_from)
        shot_df = shot_df[shot_df['metric'] == metrics[0]]
        shot_df = (shot_df.groupby('label', group_keys=False).sample(n=cfg.shot_num, random_state=114514))

    if cfg.sample_n:
        infer_df = pd.read_csv(cfg.infer_path).sample(cfg.sample_n, axis = 0)
    else:
        infer_df = pd.read_csv(cfg.infer_path)
    # _, infer_df = retrieve_data(df = aligned_df,lang = ['en'], sample_n=0, system = ['1','2','3']) 
    llm_input = []

    if cfg.en_only:
        shot_df = shot_df[shot_df['lang'] == 'en']
        infer_df = infer_df[infer_df['lang'] == 'en']
    #group the inference dataframe so that every iteration is a sample 
    infer_group = grouping(df = infer_df, group_type=['metric'])
    
    
    group_keys = []
    for key_val,infer in infer_group:
        
        shot = prepare_gold_texts_shot(infer_df = infer, 
                             shot_df=shot_df,
                             metrics=metrics)
        
        key = make_identity(infer)

        shot["key"] = key
        llm_input.append(shot)
        group_keys.append(key_val)

    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2, ensure_ascii=False)

    print(f'length of json:{len(llm_input)}')

if __name__ == "__main__":
    main()