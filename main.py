import read_data
import os
import preprocessing.data_preprocessing as dp
from feature_selection import filter_selection

def main():
    # Read heart disease data into dataframe
    heart = read_data.csv_to_df(os.path.join('data', 'heart.csv'))

    print('---------- Data Preprocessing ----------')
    # Remove any null values from the data
    dp.remove_null_values(heart)

    # Update values in target field to only include 0 or 1
    dp.replace_target_values(heart)

    # Split the data into train and test data
    data_train, data_test  = dp.split_data(heart, 'target', 0.25)
    

    print("---------- Filter Feature Selection ----------")
    # Pearson correlation
    filter_subset = filter_selection.pearson_correlation(data_train, 0.4)
    print(filter_subset)


if __name__ == "__main__":
    main()