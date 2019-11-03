import pandas as pd

def csv_to_df(path_to_csv):
    print("Reading in heasrt data")
    return pd.read_csv(path_to_csv)
