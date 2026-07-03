from utils.dataset_helper.get_data import get_train_valid, grouping,retrieve_data, PER_SYSTEM_KEY,group_sample_from_df
from exp.few_shot.scripts.make_shot import prepare_shots, make_identity
import draccus
from dataclasses import dataclass
import pandas as pd
import ast
import json
from tqdm import tqdm

@dataclass
class bss_config:
    bs_sample_from:str = "" #data to pick shots from
    infer_path:str = "" #data to do inference
    output_path:str = ""
    sample_n:int = None #the how many samples to do inference
    exclude_image:bool = False
    metrics:str = "[]"
    shot_num:int = 7
    bootstrap_num:int = 10
    include_explain:bool = False
    continuous_output:bool = False
    woundcare_only:bool = False

@draccus.wrap()
def main(cfg:bss_config):
    metrics = ast.literal_eval(cfg.metrics)

    infer_df = pd.read_csv(cfg.infer_path)
    shot_df = pd.read_csv(cfg.bs_sample_from)
    #configs
    if cfg.woundcare_only:
        shot_df = shot_df[shot_df['dataset'] == 'woundcare']


    if cfg.sample_n:
        infer_df = pd.read_csv(cfg.infer_path).sample(cfg.sample_n, axis = 0)
    else:
        infer_df = pd.read_csv(cfg.infer_path)
    # _, infer_df = retrieve_data(df = aligned_df,lang = ['en'], sample_n=0, system = ['1','2','3']) 
    llm_input = []
    #group the inference dataframe so that every iteration is a sample 
    infer_group = grouping(df = infer_df, group_type=['metric'])

    for i in tqdm(range(cfg.bootstrap_num)):
        shot_samp = group_sample_from_df(df = shot_df,sample_n=cfg.shot_num)
        group_keys = []
        for key_val,infer in infer_group:
            shot = prepare_shots(infer_df = infer, 
                                shot_df=shot_samp,
                                metrics=metrics, 
                                exclude_image=cfg.exclude_image, 
                                include_explain=cfg.include_explain, 
                                continuous_output=cfg.continuous_output)
            
            key = make_identity(infer)

            shot["key"] = key
            llm_input.append(shot)
            group_keys.append(key_val)
            

    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2, ensure_ascii=False)
        print(f"json length{len(llm_input)}")

if __name__ == "__main__":
    main()



