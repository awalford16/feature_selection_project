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
    
    train = data[int(len(data) * split):]
    test = data[:int(len(data) * split)]

    print(len(train))
    print(len(test))

    return train, test
    
    #return train_test_split(feature_data, target, test_size=split)


def remove_null_values(data):
    # Remove null values found in columns and rows from dataset
    print("Found {} null values within dataset.".format(data.isnull().sum().sum()))
    return data.dropna(axis=0)


# Heart disease target contains 5 different values. Need to replace with either 0 or 1
def replace_target_values(data):
    for i, target in enumerate(data.target):
        # if the target is 1, 2, 3 or 4 replace with 1
        if target > 0:
            data.replace({'target': i}, 1)

    return data
