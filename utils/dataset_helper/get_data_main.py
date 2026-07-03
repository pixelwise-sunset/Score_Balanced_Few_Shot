#if you want to get a grouped dataset where each group contains all the metrics
import pandas as pd
from utils.dataset_helper.get_data import grouping

if __name__ == "__main__":

    path_to_csv = "path/to/csv"

    df = pd.read_csv(path_to_csv)
    grouped_df = grouping(df, group_type = ['metric'])

    for key_vals, group in grouped_df:
        print(grouped_df)
