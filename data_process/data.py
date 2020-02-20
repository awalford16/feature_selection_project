from data_process.data_preprocessing import PreProcessing
from data_process import read_data
import os

class DataInit:
    def __init__(self, dataset_files, num_cols, targets):
        self.datasets = dataset_files
        self.numerical_cols = num_cols
        self.targets = targets

    def select_dataset(self, choice):
        if choice > len(self.datasets) or choice < 0:
            return None, None, None, None

        sep = None
        # Change delimeter for dataset 2
        if choice == 2:
            sep = ';'

        # Read in dataset
        df = read_data.csv_to_df(os.path.join('data', self.datasets[choice - 1]), sep)

        processor = PreProcessing(df, self.targets[choice - 1])
        self.process_data(processor)

        # Return processed data to user
        return processor.x_train, processor.x_test, processor.y_train, processor.y_test


    # Exectute data pre-processing steps
    def process_data(self, processor):
        # Remove any null values
        processor.remove_null_values()

        # Remove outliers
        processor.remove_outliers(self.numerical_cols)

        # Normalise the data
        processor.normalise()

        # Split the data 25%
        processor.split_data(0.25)