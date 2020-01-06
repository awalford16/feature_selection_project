import read_data
import os
import preprocessing.data_preprocessing as dp
from feature_selection.filter_selection import FilterSelection

def main():
    # Read heart disease data into dataframe
    heart = read_data.csv_to_df(os.path.join('data', 'heart.csv'))
    cardio = read_data.csv_to_df(os.path.join('data', 'cardio_train.csv'), ';')

    print('---------- Data Preprocessing ----------')
    # Remove any null values from the data
    dp.remove_null_values(heart)
    dp.remove_null_values(cardio)

    # Cast columns to category type
    heart = dp.col_to_cat(heart, ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    cardio = dp.col_to_cat(cardio, ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])

    # Split the data into train and test data
    hx_train, hx_test, hy_train, hy_test  = dp.split_data(heart, 'target', 0.25)
    cx_train, cx_test, cy_train, cy_test = dp.split_data(cardio, 'cardio', 0.2)


    print("---------- Filter Feature Selection ----------")
    # Create FS object
    fs = FilterSelection(0.3, 0.5, 0.1)
    
    # Pearson correlation
    # Pass in target to compare correlation
    h_corr = fs.corr_selection(hx_train, hy_train, 'target')
    c_corr= fs.corr_selection(cx_train, cy_train, 'cardio')
    print(h_corr.head())
    print(c_corr.head())

    # Variance Threshold
    # Only pass in training data without target
    h_variance = fs.variance_selection(cx_train)

    # Mutual Information
    h_entropy = fs.entropy_selection(hx_train, hy_train)


    print('---------- Cross-Validation ----------')


if __name__ == "__main__":
    main()