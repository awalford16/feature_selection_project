from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data, target_name, split):
    # split the data into test and train
    feature_data = data.loc[:, data.columns != target_name]
    target = data[target_name]

    if split > 1 or split < 0:
        raise Exception('Value for split cannot be greater than 1 or less than 0.')

    split_percent = int(split * 100)
    print('Splitting data into {}% train and {}% test'.format(100 - split_percent, split_percent))
    return train_test_split(feature_data, target, test_size=split)

def remove_null_values(data):
    # Remove null values found in columns and rows from dataset
    print("Found {} null values within dataset.".format(data.isnull().sum().sum()))
    return data.dropna(axis=0)