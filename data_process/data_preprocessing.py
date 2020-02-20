from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from scipy import stats

class PreProcessing:
    def __init__(self, data, target):
        self.data = data
        self.target_name = target
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    # Split the data into train and test portions
    def split_data(self, split):
        feature_data = self.data.loc[:, self.data.columns != self.target_name]
        target = self.data[self.target_name]

        if split > 1 or split < 0:
            raise Exception('Value for split cannot be greater than 1 or less than 0.')

        split_percent = int(split * 100)
        print('Splitting data into {}% train and {}% test'.format(100 - split_percent, split_percent))
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(feature_data, target, test_size=split)


    # Remove null values found in columns and rows from dataset
    def remove_null_values(self):
        print("Found {} null values within dataset.".format(self.data.isnull().sum().sum()))
        self.data.dropna(axis=0)

    # Get outliers
    def get_outliers(self, cols):
        df2 = pd.DataFrame(self.data, columns=cols)
        # th = 3
        
        # z_score = np.abs(stats.zscore(df2))
        # return np.where(z_score > th)[0]
        q1 = df2.quantile(0.25)
        q3 = df2.quantile(0.75)
        iqr = q3 - q1

        outliers = ((df2 < (q1 - 1.5 * iqr)) | (df2 > (q3 + 1.5 * iqr)))
        df2 = df2[~outliers.any(axis=1)]
        return df2.index.values.tolist()


    # Remove outliers from the dataset
    def remove_outliers(self, cols):
        # get index values of rows that contain outliers
        indices = self.get_outliers(cols)
        self.data = self.data.loc[indices, :]
        print(self.data.shape)


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
        