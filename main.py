from preprocessing import read_data
import os
import pandas as pd

from plotting import Plot
from program.fs_process import FSProcess
from program.model_process import ModelProcess
from preprocessing.data_preprocessing import Data
from feature_selection.wrapper_selection import WrapperSelection
from models.forest import Forest


def main():
    # Read heart disease data into dataframe
    d1 = read_data.csv_to_df(os.path.join('data', 'heart.csv'))
    d2 = read_data.csv_to_df(os.path.join('data', 'cardio_train.csv'), ';')

    print('---------- Data Preprocessing ----------')
    print('Reading in data from Dataset 1 and Dataset 2...')
    d1_data = Data(d1, 'target')
    d2_data = Data(d2, 'cardio')

    # Create data plots
    plt = Plot()
    plt.stack_plot(d1_data.data, 'sex', 'target')
    plt.stack_plot(d2_data.data, 'gender', 'cardio')


    # Remove any null values from the data
    d1_data.remove_null_values()
    d2_data.remove_null_values()

    d1_data.normalise()
    d2_data.normalise()

    plt.box_plot(d1_data.data, ['trestbps', 'chol', 'thalach', 'oldpeak'], 'target')
    plt.iso_forest(d2_data.data, 'ap_hi')

    # Discretize continuous data into categories
    #d1_data.discretize(['trestbps', 'chol', 'thalach', 'oldpeak'], 5)
    #d2_data.discretize(['age', 'height', 'weight', 'ap_hi', 'ap_lo'], 5)

    # Split the data into train and test data
    d1_x_train, d1_x_test, d1_y_train, d1_y_test  = d1_data.split_data(0.25)
    d2_x_train, d2_x_test, d2_y_train, d2_y_test = d2_data.split_data(0.30)


    print("---------- Filter Feature Selection ----------")

    fs_options = ['Chi Square', 'Mutual Information', 'Max Relevance Minimum Redundancy', 'ReliefF']

    # Create FS process variable
    fs = FSProcess(d1_x_train, d1_y_train, d2_x_train, d2_y_train)

    # Pre-assign variables
    d1 = d2 = None

    # Validate d1 and d2 are both dataframes
    while not isinstance(d1, pd.DataFrame) or not isinstance(d2, pd.DataFrame):
        for i in range(len(fs_options)):
            print(f'{i+1}. {fs_options[i]}')

        fs_method = int(input('Select Feature Selection Option: '))

        d1, d2 = fs.exec_fs(fs_method)


    print('---------- Model Training ----------')
    # # Choice to apply hybrid feature selection
    # hybrid = None
    # while not hybrid == 1 or not hybrid == 2:
    #     print('1. Train Model with Current Features\n2. Train Model with Hybrid Feature Selection')
    #     hybrid = int(input('How to train model: '))

    # Choice on classifier
    model = None
    while model != 1 and model != 2:
        print('1. Random Forest\n2. Artificial Neural Network')
        model = int(input('Select Classification Model: '))

    # Train model appropriately
    d1_model = ModelProcess(model, d1, d1_y_train, d1_x_test[d1.columns], d1_y_test)
    print('\n\nDataset 1:')
    d1_model.run()

    d2_model = ModelProcess(model, d2, d2_y_train, d2_x_test[d2.columns], d2_y_test)
    print('\n\nDataset 2:')
    d2_model.run()


if __name__ == "__main__":
    main()