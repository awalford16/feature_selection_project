import os
import pandas as pd

from plotting import Plot
from models.classification import Classification
from data_process.data import DataInit
from feature_selection.feature_select import FeatureSelection
from feature_selection.wrapper_selection import WrapperSelection
from comparison import Comparison


def main():
    # Store data file names
    data_files = [
        'heart.csv',
        'cardio.csv'
    ]

    # List of columns that are numerical
    num_cols = [
        ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
        ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    ]

    # Store list of data target names
    target_names = [
        'target',
        'cardio'
    ]    

    data_selector = DataInit(data_files, num_cols, target_names)

    print('\n---------- Datasets ----------')
    x_train = None
    while not isinstance(x_train, pd.DataFrame):
        # Print out possible data files
        for i in range(len(data_files)):
            print(f'{i + 1}. {data_files[i]}')

        choice = int(input(f'\nSelect a dataset: '))
        x_train, x_test, y_train, y_test = data_selector.select_dataset(choice)


    print("\n---------- Filter Feature Selection ----------")
    # Create FS object
    fs = FeatureSelection(x_train, y_train, 9)
    fs.exec_fs()


    print('\n---------- Model Training ----------')
    # Choice on classifier
    model = None
    while model != 1 and model != 2:
        print('1. Random Forest\n2. Artificial Neural Network')
        model = int(input('Select Classification Model: '))


    print('\n---------- Performance Scores ----------')
    # Init comparison class to compare results
    c = Comparison()

    # Loop through each feature subset
    for subset in fs.feature_sets.keys():
        
        print(f"\n\n{subset} Feature Selection")
        # Use different subset sizes with a minimum of 5 features
        for subset_size in range(5, fs.k + 1):
            print(f"Testing with {subset_size} features...")
            # Get relative columns
            columns = fs.feature_sets[subset][:subset_size]

            # Train and test model using relative columns
            classifier = Classification(model, x_train[columns], y_train, x_test[columns], y_test)
            classifier.run()
            
            c.compare(subset, columns, classifier)

        c.display_method_result(subset)

    c.display_results()



if __name__ == "__main__":
    main()