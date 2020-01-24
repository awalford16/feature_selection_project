import read_data
import os
from preprocessing.data_preprocessing import Data
from feature_selection.filter_selection import FilterSelection
from models.forest import Forest

def main():
    # Read heart disease data into dataframe
    heart = read_data.csv_to_df(os.path.join('data', 'heart.csv'))
    cardio = read_data.csv_to_df(os.path.join('data', 'cardio_train.csv'), ';')

    print('---------- Data Preprocessing ----------')
    h_data = Data(heart, 'target')
    c_data = Data(cardio, 'cardio')

    # Remove any null values from the data
    h_data.remove_null_values()
    c_data.remove_null_values()

    h_data.normalise()
    c_data.normalise()

    # Discretize continuous data into categories
    #h_data.discretize(['trestbps', 'chol', 'thalach', 'oldpeak'], 5)
    #c_data.discretize(['age', 'height', 'weight', 'ap_hi', 'ap_lo'], 5)

    # Split the data into train and test data
    hx_train, hx_test, hy_train, hy_test  = h_data.split_data(0.25)
    cx_train, cx_test, cy_train, cy_test = c_data.split_data(0.2)


    print("---------- Filter Feature Selection ----------")
    # Create FS object
    fs = FilterSelection(7)

    # Chi Square
    # Reduce datasets based on the range of data
    print('\nChi-Square Feature Selection...')
    h_chi = fs.chi2(hx_train, hy_train)
    c_chi = fs.chi2(cx_train, cy_train)

    print(f'Chi2 Heart Features: {h_chi.columns}')
    print(f'Chi2 Cardio Features: {c_chi.columns}')

    # Mutual Information
    print('\nMutual Information Feature Selection...')
    h_mi = fs.mi(hx_train, hy_train)
    c_mi = fs.mi(cx_train, cy_train)

    print(f'MI Heart Features: {h_mi.columns}')
    print(f'MI Cardio Features: {c_mi.columns}')

    # Minimum Redundancy Maximum Relevance
    h_mrmr = fs.mrmr(hx_train)
    c_mrmr = fs.mrmr(cx_train)

    print(f'MRMR Heart Features: {h_mrmr.columns}')
    print(f'MRMR Cardio Features: {c_mrmr.columns}')



    print('---------- Training with Correlation Data ----------')
    rf = Forest(500, 10)

    # Train the model with correlation feature subset
    rf.train(h_chi, hy_train)
    chi_pred = rf.test(hx_test[h_chi.columns])
    acc, prec, rec = rf.score(chi_pred, hy_test)
    print(f'Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}')

if __name__ == "__main__":
    main()