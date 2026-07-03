import pandas as pd
import numpy as np
import random


FOLDED_DATA = "datasets/mediqa-eval-2026-valid-folded.csv"
ALIGNED_DATA = "datasets/mediqa-eval-2026-valid-aligned.csv"
ORIG_DATA = "datasets/mediqa-eval-2026-valid.csv"

SYSTEM_ID = ['SYSTEM001','SYSTEM002','SYSTEM003']
#NM and SG are the two raters for en iiyi. A1 is the rater for zh iiyi ,en woundcare and zh woundcare 
RATERS = ['NM','SG','A1']

KEYS = ['dataset', 'lang', 'encounter_id', 'candidate_author_id', 'metric']
PER_OBS_KEY = ['dataset', 'lang', 'encounter_id']
PER_SYSTEM_KEY = ['dataset', 'lang', 'encounter_id', 'candidate_author_id', 'candidate']
#for en
METRICS = ['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']

def remove_rater_duplicates(df):
    df = df.drop_duplicates(subset=KEYS, keep = 'first')
    return df

def group_sample_from_df(df: pd.DataFrame, sample_n: int) -> pd.DataFrame:
    grouped_df = grouping(df, group_type=['metric'])
    keys = list(grouped_df.groups.keys())
    chosen_keys = random.sample(keys, sample_n)
    sampled_df = pd.concat([grouped_df.get_group(k) for k in chosen_keys])
    return sampled_df.reset_index(drop=True)


# def grouping(df:pd.DataFrame, group_type:str = ['metric']) -> pd.DataFrame:
#     # key_columns = [col for col in df.columns if col not in group_type]
#     group_type += ['label', 'query_text', 'gold_texts']
#     key_columns = [col for col in df.columns if col not in group_type]
#     groups = df.groupby(key_columns)
#     # print(f"output {group_type} based on key: {key_columns}")
#     return groups

def grouping(df: pd.DataFrame, group_type=None):
    if group_type is None:
        group_type = ['metric']

    exclude_cols = group_type + ['label', 'query_text', 'gold_texts', 'candidate']
    key_columns = [col for col in df.columns if col not in exclude_cols]

    return df.groupby(key_columns)

def grouping_with_gold(df: pd.DataFrame, group_type=None):
    if group_type is None:
        group_type = ['metric']

    exclude_cols = group_type + ['label', 'query_text', 'candidate']
    key_columns = [col for col in df.columns if col not in exclude_cols]

    return df.groupby(key_columns)

def get_train_valid(valid_id) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(FOLDED_DATA)

    train_df = df[df['fold'] != valid_id]
    valid_df = df[df['fold'] == valid_id]

    return train_df, valid_df

def retrieve_data(df:pd.DataFrame,raters:list[str] = None,lang:str = None, system:list[str] = ["1", "2", "3"], sample_n:int = None, metrics:list[str] = ['writing-style', 'overall'], seed = 114514) -> pd.DataFrame:
    #get unique encounter_id
    if raters:
        df = df[df['rater_id'].isin(raters)]
    #specify the language
    if lang:
        df = df[df['lang'].isin(lang)]

    uniq_id = df['encounter_id'].unique()

    #sample from unique_id
    np.random.seed(seed)
    if sample_n:
        sampled_ids = np.random.choice(uniq_id, size=sample_n, replace=False)
        #get the remaining for generating a validation set
        remaining_ids = np.setdiff1d(uniq_id, sampled_ids)
    else:
        if sample_n == 0:
            sampled_ids = np.array([]) 
            remaining_ids = uniq_id
        else: 
            sampled_ids = uniq_id
            remaining_ids = []

    system = ["SYSTEM00" + s for s in system]

    # data_label_pair = df[df['encounter_id'] == sampled_ids and df['encounter_id'] == system]
    train_mask = (df['encounter_id'].isin(sampled_ids)) & (df['candidate_author_id'].isin(system)) & (df['metric'].isin(metrics))
    val_mask = (df['encounter_id'].isin(remaining_ids)) & (df['candidate_author_id'].isin(system)) & (df['metric'].isin(metrics))

    data_label_pair = df[train_mask].copy()
    val_data_label_pair = df[val_mask].copy()
    
    return data_label_pair.sort_values(by=["encounter_id", "candidate_author_id"]), val_data_label_pair.sort_values(by=["encounter_id", "candidate_author_id"])

def get_cov_mat(df:pd.DataFrame = pd.read_csv(ORIG_DATA)):
    df = df.sort_values(by=['encounter_id', 'candidate_author_id'])
    #returns two matrix:1.cov mat for en, 2.cov mat for zh
    en_df = df[df['lang'] == 'en']
    zh_df = df[df['lang'] == 'zh']
    #group
    #vectors for english
    disagree_flag=en_df[en_df['metric'] == 'disagree_flag']['value']
    completeness = en_df[en_df['metric'] == 'completeness']['value']
    factual_accuracy = en_df[en_df['metric'] == 'factual-accuracy']['value']
    relevance = en_df[en_df['metric'] == 'relevance']['value']
    writing_style = en_df[en_df['metric'] == 'writing-style']['value']
    overall = en_df[en_df['metric'] == 'overall']['value']

    #vectors for zh
    factual_consistency_wgold = zh_df[zh_df['metric'] == 'factual-consistency-wgold']['value']
    writing_style = zh_df[zh_df['metric'] == 'writing-style']['value']

    en_pure_label = pd.concat(
    [
        disagree_flag.reset_index(drop=True).rename('disagree_flag'),
        completeness.reset_index(drop=True).rename('completeness'),
        factual_accuracy.reset_index(drop=True).rename('factual_accuracy'),
        relevance.reset_index(drop=True).rename('relevance'),
        writing_style.reset_index(drop=True).rename('writing_style'),
        overall.reset_index(drop=True).rename('overall'),
    ],
    axis=1
    )

    zh_pure_label = pd.concat(
        [
            factual_consistency_wgold.reset_index(drop=True).rename('factual_consistency_wgold'),
            writing_style.reset_index(drop=True).rename('writing_style'),
        ],
        axis=1
    )

    #compute the covariance
    en_cov = en_pure_label.corr()
    zh_cov = zh_pure_label.corr()

    return en_cov, zh_cov
    # return en_pure_label, zh_pure_label


#the main is for testing the functions

if __name__ == "__main__":
    #test data splitting
    t,v = get_train_valid(2)
    t.to_csv('my_test/train.csv')
    v.to_csv('my_test/val.csv')
    print(t['fold'].unique())
    print(v['fold'].unique())
    #test cov function
    en,zh = get_cov_mat()

    en.to_csv("my_test/en_cov.csv")
    zh.to_csv("my_test/zh_cov.csv")

    aligned_df = pd.read_csv(ALIGNED_DATA)

    #test data pair generation
    data, val = retrieve_data(df = aligned_df, lang = ['en'], sample_n=50, system = ['1','2','3'])
    data.to_csv('exp/reinforced_few_shot/datasets/infer_sample.csv', index = False)
    val.to_csv('exp/reinforced_few_shot/datasets/infer_complement.csv', index = False)

    



    
