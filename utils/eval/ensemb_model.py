import pandas as pd
from utils.dataset_helper.get_data import *

def ensemb_model_outputs(bert_out:pd.DataFrame, gemma_out:pd.DataFrame, bert_selection:list[str], gemma_selection:list[str]):
    bert_out_selected = bert_out[bert_out['metric'].isin(bert_selection)]
    gemma_out_selected = gemma_out[gemma_out['metric'].isin(gemma_selection)]

    return pd.concat([bert_out_selected, gemma_out_selected])