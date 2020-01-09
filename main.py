import read_data
import os
import preprocessing.data_preprocessing as dp
from feature_selection.filter_selection import FilterSelection
from models.forest import Forest

def main():
    # Read heart disease data into dataframe
    heart = read_data.csv_to_df(os.path.join('data', 'heart.csv'))
    cardio = read_data.csv_to_df(os.path.join('data', 'cardio_train.csv'), ';')

    print('---------- Data Preprocessing ----------')
    # Remove any null values from the data
    dp.remove_null_values(heart)
    dp.remove_null_values(cardio)

    # Cast stated columns to category type
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
    print('Correlation Feature Selection...')
    h_corr = fs.corr_selection(hx_train, hy_train, 'target')
    c_corr= fs.corr_selection(cx_train, cy_train, 'cardio')

    print(f'Heart Feature Count: {h_corr.shape[1]}')
    print(f'Cardio Feature Count: {c_corr.shape[1]}')

    # Variance Threshold
    # Reduce datasets based on the range of data
    print('\nVariance Feature Selection...')
    h_variance = fs.variance_selection(hx_train)
    c_variance = fs.variance_selection(cx_train)

    print(f'Heart Feature Count: {h_variance.shape[1]}')
    print(f'Cardio Feature Count: {c_variance.shape[1]}')

    # Mutual Information
    print('\nInformation Gain Feature Selection...')
    h_entropy = fs.entropy_selection(hx_train, hy_train)
    c_entropy = fs.entropy_selection(cx_train, cy_train)

    print(f'Heart Feature Count: {h_entropy.shape[1]}')
    print(f'Cardio Feature Count: {c_entropy.shape[1]}')

    # Store all feature subsets in an list
    subsets = [h_corr, c_corr, h_variance, c_variance, h_entropy, c_entropy]


    print('---------- Training with Correlation Data ----------')
    rf = Forest(500, 10)

    # Train the model with correlation feature subset
    rf.train(h_corr, hy_train)
    corr_pred = rf.test(hx_test[h_corr.columns])
    acc, prec, rec = rf.score(corr_pred, hy_test)
    print(f'Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}')

if __name__ == "__main__":
    main()