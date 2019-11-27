import pandas as pd

def csv_to_df(path_to_csv, delimiter=','):
    print("Reading data csv file...")
    return pd.read_csv(path_to_csv, sep=delimiter)
