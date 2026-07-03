import sys
import json

import pandas as pd
from scipy import stats
import numpy as np
import os


EVAL_COLS_UNIQUE = ['dataset',
    'encounter_id',
    'lang',
    'candidate',
    'candidate_author_id',
    'metric']

LANG2METRICS = {
    'en': ['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall'],
    'zh': ['factual-consistency-wgold','writing-style']
}
DATASETS = ['iiyi','woundcare']


def get_correlations(x,y) :
    # [Modified] 添加安全检查：如果数据点少于2个，直接返回 NaN，防止报错
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    # [Modified] 还可以选加一个方差检查，防止所有预测值都一样导致警告（可选）
    if np.std(x) == 0 or np.std(y) == 0:
         return 0.0, 0.0, 0.0, 1.0, 1.0, 1.0

    kendalltau, k_pval = stats.kendalltau(x, y)
    pearson, p_pval = stats.pearsonr(x, y)
    spearman, s_pval = stats.spearmanr(x, y)
    return kendalltau, pearson, spearman, k_pval, p_pval, s_pval

def organize_and_correlate( df_human, df_auto ) :
    df_comb = pd.merge( df_human, df_auto, on=EVAL_COLS_UNIQUE )
    return get_correlations( df_comb['value_x'], df_comb['value_y'] )

def average_multi_rater_gold( df_human ) :
    """Average human ratings when multiple raters share one prediction key."""
    missing = [col for col in EVAL_COLS_UNIQUE + ['value'] if col not in df_human.columns]
    if missing:
        raise ValueError(f'human ratings missing required columns: {missing}')
    df_human = df_human.copy()
    df_human['value'] = pd.to_numeric(df_human['value'], errors='coerce')
    return df_human.groupby(EVAL_COLS_UNIQUE, as_index=False, dropna=False)['value'].mean()

def score_correlations( df_human, df_auto ) :
    df_human = average_multi_rater_gold( df_human )
    results = {}

    for lang in ['en','zh'] :

        meanmetrics = []

        for metric in LANG2METRICS[lang] :

            df_human_temp = df_human[ (df_human['lang']==lang) & (df_human['metric']==metric) ]
            df_auto_temp = df_auto[ (df_auto['lang']==lang)  & (df_auto['metric']==metric) ]
            
            kendalltau, pearson, spearman, k_pval, p_pval, s_pval = organize_and_correlate( df_human_temp, df_auto_temp )

            results[ '{}-{}-{}-{}'.format('ALL',lang,metric,'kendalltau') ] = kendalltau
            results[ '{}-{}-{}-{}'.format('ALL',lang,metric,'pearson') ] = pearson
            results[ '{}-{}-{}-{}'.format('ALL',lang,metric,'spearman') ] = spearman
            # [Modified] 使用 nanmean 忽略空值
            results[ '{}-{}-{}-{}'.format('ALL',lang,metric,'mean') ] = np.nanmean( [kendalltau, pearson, spearman] )
            meanmetrics.append( results[ '{}-{}-{}-{}'.format('ALL',lang,metric,'mean') ] )

            for dataset in DATASETS :
                df_human_temp = df_human[ (df_human['lang']==lang) & (df_human['metric']==metric) & (df_human['dataset']==dataset) ]
                df_auto_temp = df_auto[ (df_auto['lang']==lang)  & (df_auto['metric']==metric) & (df_auto['dataset']==dataset) ]
                
                kendalltau, pearson, spearman, k_pval, p_pval, s_pval = organize_and_correlate( df_human_temp, df_auto_temp )

                results[ '{}-{}-{}-{}'.format(dataset,lang,metric,'kendalltau') ] = kendalltau
                results[ '{}-{}-{}-{}'.format(dataset,lang,metric,'pearson') ] = pearson
                results[ '{}-{}-{}-{}'.format(dataset,lang,metric,'spearman') ] = spearman
                # [Modified] 使用 nanmean
                results[ '{}-{}-{}-{}'.format(dataset,lang,metric,'mean') ] = np.nanmean( [kendalltau, pearson, spearman] )

        # [Modified] 使用 nanmean 计算总分，这样如果 zh 全是 NaN，不会影响 en 的分数显示
        results[ '{}-{}-{}-{}'.format('ALL',lang,'ALL','mean') ] = np.nanmean( meanmetrics )

    return results


if __name__ == "__main__":

    if len(sys.argv)<3 :
        print('python: mediqa_eval_script.py <human-eval-ratings> <auto-scorer-ratings>')
        sys.exit(0)

    fn_human = sys.argv[1]
    fn_auto = sys.argv[2]

    out_path = 'results'
    # 确保输出目录存在
    if not os.path.exists(out_path): os.makedirs(out_path)
    
    scores_path = sys.argv[3] if len(sys.argv) >=4 else os.path.join(out_path, os.path.basename(os.path.dirname(fn_auto)),f'{os.path.splitext(os.path.basename(fn_auto))[0]}.json')
    
    df_human = pd.read_csv(fn_human)
    df_auto = pd.read_csv(fn_auto).drop_duplicates(subset=EVAL_COLS_UNIQUE)

    print( 'Rows in human-ratings: {}'.format( len(df_human) ) )
    df_human_avg = average_multi_rater_gold( df_human )
    if len(df_human_avg) != len(df_human):
        print( 'Rows in human-ratings after multi-rater averaging: {}'.format( len(df_human_avg) ) )
    print( 'Rows in automatic-system-ratings: {}'.format( len( df_auto) ) )

    #check if all gold's dataset-lang-encounter_id-candidate_author_id-metric is in system
    #gross check if numbers are as expected
    
    # [Modified] 这里建议把 exit 改成 warning，因为你有意只跑英文，行数肯定对不上
    if len( df_human_avg ) != len( df_auto ) :
        print('[WARNING] Number of ratings mismatch. Proceeding anyway since you might be evaluating a subset (e.g. English only).')
        # sys.exit(0) # 注释掉强行退出，允许只跑部分数据

    df_comb = pd.merge( df_human_avg, df_auto, on=EVAL_COLS_UNIQUE )
    
    # [Modified] 同样的，只要有重叠数据就可以跑，不必完全相等
    if len( df_comb ) == 0:
         print('[ERROR] No overlapping data found between human and auto ratings!')
         sys.exit(0)
    
    scores = score_correlations( df_human, df_auto )

    with open( scores_path, 'w' ) as f :
        json.dump( scores, f, indent=4 )
    
    print( 'Saved scores to {}'.format( scores_path ) )
