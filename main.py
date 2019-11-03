import pandas as pd
import read_data
import os

def main():
    # Read heart disease data into dataframe
    heart = read_data.csv_to_df(os.path.join('data', 'heart.csv'))

if __name__ == "__main__":
    main()