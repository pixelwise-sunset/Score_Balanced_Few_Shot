import sys
import json

import pandas as pd
from scipy import stats
import numpy as np


EVAL_COLS_UNIQUE = ['dataset',
    'encounter_id',
    'candidate',
    'lang',
    'candidate_author_id',
    'metric']

EN_EVAL_COLS_UNIQUE = ['dataset',
    'encounter_id',
    'candidate_author_id',
    'metric']

LANG2METRICS = {
    'en': ['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall'],
    'zh': ['factual-consistency-wgold','writing-style']
}
DATASETS = ['iiyi','woundcare']


def get_correlations(x,y) :
    kendalltau, k_pval = stats.kendalltau(x, y)
    pearson, p_pval = stats.pearsonr(x, y)
    spearman, s_pval = stats.spearmanr(x, y)
    
    return kendalltau, pearson, spearman, k_pval, p_pval, s_pval

def organize_and_correlate(df_human, df_auto):
    # print(df_human)
    # print(df_auto)
    # merge_keys = EVAL_COLS_UNIQUE
    # missing_in_auto = df_human[~df_human[merge_keys].apply(tuple,1).isin(df_auto[merge_keys].apply(tuple,1))]
    # print("Rows in human not in auto:")
    # print(missing_in_auto)
    df_comb = pd.merge( df_human, df_auto, on=EVAL_COLS_UNIQUE)
    return get_correlations( df_comb['value_x'], df_comb['value_y'] )

def score_correlations( df_human, df_auto ) :
    results = {}

    for lang in ['en', 'zh'] :

        meanmetrics = []

        for metric in LANG2METRICS[lang] :

            df_human_temp = df_human[ (df_human['lang']==lang) & (df_human['metric']==metric) ]
            df_auto_temp = df_auto[ (df_auto['lang']==lang)  & (df_auto['metric']==metric) ]
            
            kendalltau, pearson, spearman, k_pval, p_pval, s_pval = organize_and_correlate( df_human_temp, df_auto_temp )

            results[ '{}-{}-{}-{}'.format('ALL',lang,metric,'kendalltau') ] = kendalltau
            results[ '{}-{}-{}-{}'.format('ALL',lang,metric,'pearson') ] = pearson
            results[ '{}-{}-{}-{}'.format('ALL',lang,metric,'spearman') ] = spearman
            results[ '{}-{}-{}-{}'.format('ALL',lang,metric,'mean') ] = np.mean( [kendalltau, pearson, spearman] )
            meanmetrics.append( results[ '{}-{}-{}-{}'.format('ALL',lang,metric,'mean') ] )

            for dataset in DATASETS :
                df_human_temp = df_human[ (df_human['lang']==lang) & (df_human['metric']==metric) & (df_human['dataset']==dataset) ]
                df_auto_temp = df_auto[ (df_auto['lang']==lang)  & (df_auto['metric']==metric) & (df_auto['dataset']==dataset) ]
                
                kendalltau, pearson, spearman, k_pval, p_pval, s_pval = organize_and_correlate( df_human_temp, df_auto_temp )

                results[ '{}-{}-{}-{}'.format(dataset,lang,metric,'kendalltau') ] = kendalltau
                results[ '{}-{}-{}-{}'.format(dataset,lang,metric,'pearson') ] = pearson
                results[ '{}-{}-{}-{}'.format(dataset,lang,metric,'spearman') ] = spearman
                results[ '{}-{}-{}-{}'.format(dataset,lang,metric,'mean') ] = np.mean( [kendalltau, pearson, spearman] )

        results[ '{}-{}-{}-{}'.format('ALL',lang,'ALL','mean') ] = np.mean( meanmetrics )

    return results

def main(df_human:pd.DataFrame, df_auto:pd.DataFrame, save_path):
    df_human = df_human
    df_auto = df_auto.drop_duplicates(subset=EVAL_COLS_UNIQUE)

    print( 'Rows in human-ratings: {}'.format( len(df_human) ) )
    print( 'Rows in automatic-system-ratings: {}'.format( len( df_auto) ) )

    #check if all gold's dataset-lang-encounter_id-candidate_author_id-metric is in system
    #gross check if numbers are as expected
    if len( df_human.drop_duplicates(subset=EVAL_COLS_UNIQUE)) != len( df_auto ) :
        print('Number of automatic metric ratings do not match the number of candidate-metric pairs')
        sys.exit(0)

    df_comb = pd.merge( df_human, df_auto, on=EVAL_COLS_UNIQUE )
    if len( df_comb ) != len( df_human ) :
        print('Merging candidate-metric pairs, there was a mismatch.')
        sys.exit(0)
    
    scores = score_correlations( df_human, df_auto )

    with open(save_path, 'w' ) as f :
        json.dump( scores, f, indent=4 )

if __name__ == "__main__":

    if len(sys.argv)<3 :
        print('python: mediqa_eval_script.py <human-eval-ratings> <auto-scorer-ratings>')
        sys.exit(0)

    fn_human = sys.argv[1]
    fn_auto = sys.argv[2]

    scores_path = sys.argv[3] if len(sys.argv) >=4 else 'scores.json'

    df_human = pd.read_csv(fn_human)
    df_auto = pd.read_csv(fn_auto).drop_duplicates(subset=EVAL_COLS_UNIQUE)

    print( 'Rows in human-ratings: {}'.format( len(df_human) ) )
    print( 'Rows in automatic-system-ratings: {}'.format( len( df_auto) ) )

    #check if all gold's dataset-lang-encounter_id-candidate_author_id-metric is in system
    #gross check if numbers are as expected
    if len( df_human.drop_duplicates(subset=EVAL_COLS_UNIQUE)) != len( df_auto ) :
        print('Number of automatic metric ratings do not match the number of candidate-metric pairs')
        sys.exit(0)

    df_comb = pd.merge( df_human, df_auto, on=EVAL_COLS_UNIQUE )
    if len( df_comb ) != len( df_human ) :
        print('Merging candidate-metric pairs, there was a mismatch.')
        sys.exit(0)
    
    scores = score_correlations( df_human, df_auto )

    with open( scores_path, 'w' ) as f :
        json.dump( scores, f, indent=4 )