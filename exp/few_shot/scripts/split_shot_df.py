from utils.dataset_helper.get_data import get_train_valid, METRICS,grouping, retrieve_data
import pandas as pd
import itertools

KEYS = ['dataset', 'lang', 'encounter_id', 'candidate_author_id']



orig_df = pd.read_csv("datasets/mediqa-eval-2026-valid-folded.csv")
# train,val = get_train_valid(valid_id=[4])
# #exclude lang == zh

# train = train[train['lang'] != 'zh']
# val = val[val['lang'] != 'zh']

# train.to_csv("exp/few_shot/datasets/train.csv", index=False)
# val.to_csv("exp/few_shot/datasets/val.csv", index=False)

def select_shot(dataframe:pd.DataFrame):
    #this function finds all possible combinations of the scores of 'writing-style' and 'overall'
    aligned_df = dataframe

    _, df = retrieve_data(
        df=aligned_df,
        lang=['en'],
        sample_n=0,
        system=['1', '2', '3']
    )

    grouped_df = grouping(df=df, group_type=['metric'])

    score_vals = [0.0, 0.5, 1.0]

    # initialize all 9 combinations with None
    shots = {
        (ws, ov): None for ws, ov in itertools.product(score_vals, score_vals)
    }

    # group by sample identity (adjust keys if needed)
    sample_groups = grouped_df

    for _, g in sample_groups:
        ws_score = g.loc[g['metric'] == 'writing-style', 'label']
        ov_score = g.loc[g['metric'] == 'overall', 'label']

        # skip incomplete samples
        if ws_score.empty or ov_score.empty:
            continue

        ws = float(ws_score.iloc[0])
        ov = float(ov_score.iloc[0])

        key = (ws, ov)

        # only take the first occurrence
        if key in shots and shots[key] is None:
            shots[key] = g

        # early stop if all combos are filled
        if all(v is not None for v in shots.values()):
            break

    orig_df = df
    return shots, orig_df

def select_metrics(orig_df:pd.DataFrame,shot_df:pd.DataFrame, metrics:list[str])->pd.DataFrame:
    matched_df = pd.merge(orig_df, shot_df[KEYS], on=KEYS, how='inner')
    matched_df = matched_df[matched_df['metric'].isin(metrics)]
    return matched_df


def split_shot(split_from:str, shot_num:int, metrics:list[str], save_path:str) -> pd.DataFrame:
    train = pd.read_csv(split_from)

    train_iiyi = train[train['dataset'] == 'iiyi']
    train_woundcare  = train[train['dataset'] == 'woundcare']

    iiyi_shots, _ = select_shot(dataframe = train_iiyi) #contains overall and writing style


    woundcare_shots,_ = select_shot(dataframe=train_woundcare) #contains overall and writing style
    #separate the keys
    iiyi_shots_unique = {('iiyi', *k): v for k, v in iiyi_shots.items()}
    woundcare_shots_unique = {('woundcare', *k): v for k, v in woundcare_shots.items()}

    shots = {**iiyi_shots_unique, **woundcare_shots_unique}
    
    full_shots_df = pd.DataFrame()
    #save zero shot
    if save_path:
        pd.DataFrame(columns=train.columns).to_csv(f"{save_path}/shot0.csv", index=False)
    # Use enumerate to get the index (i) and unpack the items (key, shot)
    s = 0
    for i, (key, shot) in enumerate(shots.items(), 1):

        # Check if shot is not None and not an empty DataFrame
        if shot is not None and not shot.empty:
            # Concatenate
            shot = select_metrics(orig_df=orig_df, shot_df = shot, metrics=metrics)
            full_shots_df = pd.concat([full_shots_df, shot], ignore_index=True)
            s += 1
        
        #if the shot number is equal to the defined shot number, stop the loop and save the full_shots_df
        if s == shot_num:
            shots_df_nodup = full_shots_df.drop_duplicates(subset=KEYS + ['metric'])
            #save the dataframe if defined a path
            if save_path:
                shots_df_nodup.to_csv(f"{save_path}/shot{s}.csv", index=False)
                
            print(f"saved shot_df with shot number: {shot_num} and metrics: {metrics}")
            return shots_df_nodup



if __name__ == "__main__":
    split_shot(shot_num=14, metrics=METRICS, save_path="exp/few_shot/datasets/shot_selection")

        


        



    