from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

class Data:
    def __init__(self, data, target):
        self.data = data
        self.target_name = target

    # Split the data into train and test portions
    def split_data(self, split):
        feature_data = self.data.loc[:, self.data.columns != self.target_name]
        target = self.data[self.target_name]

        if split > 1 or split < 0:
            raise Exception('Value for split cannot be greater than 1 or less than 0.')

        split_percent = int(split * 100)
        print('Splitting data into {}% train and {}% test'.format(100 - split_percent, split_percent))
        
        train_x, test_x, train_y, test_y = train_test_split(feature_data, target, test_size=split)
        return train_x, test_x, train_y, test_y 


    # Remove null values found in columns and rows from dataset
    def remove_null_values(self):
        print("Found {} null values within dataset.".format(self.data.isnull().sum().sum()))
        self.data.dropna(axis=0)


    # Normalise data to have consistent ranges
    def normalise(self):
        data_min = self.data.min()
        data_max = self.data.max()

        self.data = ((self.data - data_min) / (data_max - data_min))


    # Discretize continuous data to nominal data
    def discretize(self, cols, bins):
        print('Discretising Continuous Data...')
        for col in cols:
            # Create bins to place data
            disc = KBinsDiscretizer(n_bins=bins, encode='onehot')

            # Split data into bins
            self.data[col] = disc.fit_transform(self.data[col])
        

    # Cast columns to category types
    def col_to_cat(self, cols):
        for col in cols:
            self.data[col] = self.data[col].astype('category')
        