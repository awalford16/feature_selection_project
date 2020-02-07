import read_data
import os

from preprocessing.data_preprocessing import Data
from feature_selection.filter_selection import FilterSelection
from feature_selection.wrapper_selection import WrapperSelection
from models.forest import Forest

def display_features(model, d1, d2):
    print(f'{model} Dataset 1 Features: {d1.columns}')
    print(f'{model} Dataset 2 Features: {d2.columns}')

def main():
    # Read heart disease data into dataframe
    d1 = read_data.csv_to_df(os.path.join('data', 'heart.csv'))
    d2 = read_data.csv_to_df(os.path.join('data', 'cardio_train.csv'), ';')

    print('---------- Data Preprocessing ----------')
    print('Reading in data from Dataset 1 and Dataset 2...')
    d1_data = Data(d1, 'target')
    d2_data = Data(d2, 'cardio')

    # Remove any null values from the data
    d1_data.remove_null_values()
    d2_data.remove_null_values()

    d1_data.normalise()
    d2_data.normalise()

    # Discretize continuous data into categories
    #d1_data.discretize(['trestbps', 'chol', 'thalach', 'oldpeak'], 5)
    #d2_data.discretize(['age', 'height', 'weight', 'ap_hi', 'ap_lo'], 5)

    # Split the data into train and test data
    d1_x_train, d1_x_test, d1_y_train, d1_y_test  = d1_data.split_data(0.25)
    d2_x_train, d2_x_test, d2_y_train, d2_y_test = d2_data.split_data(0.30)


    print("---------- Filter Feature Selection ----------")
    # Create FS object
    fs = FilterSelection(7)

    # Chi Square
    # Reduce datasets based on the range of data
    print('\nChi-Square Feature Selection...')
    d1_chi = fs.chi2(d1_x_train, d1_y_train)
    d2_chi = fs.chi2(d2_x_train, d2_y_train)
    display_features('Chi2', d1_chi, d2_chi)

    # Mutual Information
    print('\nMutual Information Feature Selection...')
    d1_mi = fs.mi(d1_x_train, d1_y_train)
    d2_mi = fs.mi(d2_x_train, d2_y_train)
    display_features('MI', d1_mi, d2_mi)

    # Mutual Information
    print('\nmRMR Feature Selection...')
    # Minimum Redundancy Maximum Relevance
    d1_mrmr = fs.mrmr(d1_x_train)
    d2_mrmr = fs.mrmr(d2_x_train)
    display_features('mRMR', d1_mrmr, d2_mrmr)

    # Mutual Information
    print('\nReliefF Feature Selection...')
    d1_rel = fs.rf(d1_x_train, d1_y_train)
    d2_rel = fs.rf(d2_x_train, d2_y_train)
    display_features('ReliefF', d1_rel, d2_rel)


    print('---------- Training with Correlation Data ----------')
    rf = Forest(500, 10)

    print('Using Hybrid Feature Selection...')
    # Option to implement Hybrid feature selection
    ws = WrapperSelection(5, rf.model)
    features = ws.forward_select(d1_chi, d1_y_train)

    print(features.columns)

    # # Train the model with correlation feature subset
    # rf.train(d1_chi, d1_y_train)
    # chi_pred = rf.test(d1_x_test[d1_chi.columns])
    # acc, prec, rec = rf.score(chi_pred, d1_y_test)
    # print(f'Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}')

if __name__ == "__main__":
    main()