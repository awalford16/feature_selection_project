import read_data
import os
import preprocessing.data_preprocessing as dp
from feature_selection import filter_selection

def main():
    # Read heart disease data into dataframe
    heart = read_data.csv_to_df(os.path.join('data', 'heart.csv'))
    cardio = read_data.csv_to_df(os.path.join('data', 'cardio_train.csv'), ';')

    print('---------- Data Preprocessing ----------')
    # Remove any null values from the data
    dp.remove_null_values(heart)
    dp.remove_null_values(cardio)

    # Split the data into train and test data
    heart_train, heart_test  = dp.split_data(heart, 'target', 0.25)
    cardio_train, cardio_test = dp.split_data(cardio, 'cardio', 0.3)

    print("---------- Filter Feature Selection ----------")
    # Pearson correlation
    heart_subset = filter_selection.pearson_correlation(heart_train, 'target', 0.4)
    cardio_subset = filter_selection.pearson_correlation(cardio_train, 'cardio', 0.05)
    print(cardio_subset)

    # Variance Threshold

    # Mutual Information


    print('---------- Cross-Validation ----------')


if __name__ == "__main__":
    main()