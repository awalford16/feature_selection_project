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
    hx_train, hx_test, hy_train, hy_test  = dp.split_data(heart, 'target', 0.25)
    cx_train, cx_test, cy_train, cy_test = dp.split_data(cardio, 'cardio', 0.3)


    print("---------- Filter Feature Selection ----------")
    # Pearson correlation
    # Pass in target to compare correlation
    h_correlation = filter_selection.pearson_correlation(hx_train, hy_train, 'target', 0.4)
    c_correlation = filter_selection.pearson_correlation(cx_train, cy_train, 'cardio', 0.05)
    print(c_correlation)

    # Variance Threshold
    # Only pass in training data without target
    h_variance = filter_selection.variance_selection(cx_train, 0.5)
    print(len(h_variance[0]))

    # Mutual Information


    print('---------- Cross-Validation ----------')


if __name__ == "__main__":
    main()