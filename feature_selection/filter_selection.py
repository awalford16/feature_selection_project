import sklearn.feature_selection as fs
import pandas as pd
import numpy as np
from plotting import Plot

class FeatureSelection:
    def __init__(self, pearson_threshold = 0.4, variance_threshold=0.4, entropy_threshold=0.2):
        self.pc_th = pearson_threshold
        self.var_th = variance_threshold
        self.e_th = entropy_threshold

    # Pearson correlation code available from: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
    def pearson_correlation(self, data, target_data, target_name):
        # Merge data and target data into same dataframe
        data[target_name] = target_data.values

        # get correlations between data features
        corr = data.corr()
        names = list(data.columns)
        
        # Create plot of pearson correlation
        plt = Plot()
        plt.plot_pc(corr, names)

        # identify correlations with output variable
        target = corr[target_name][:-1]

        # Return columns with values greater than required correlation
        return target[(abs(target) > self.pc_th)]

    # Variance filter selection
    def variance_selection(self, data):
        selector = fs.VarianceThreshold(threshold=self.var_th)
        return selector.fit_transform(data)

    # Filter feature selection based on information gain
    def entropy_selection(self, data, target_data):
        ig = fs.mutual_info_classif(data, target_data)
        drop_columns = []

        # Loop through each of the IG values (Ignore last value (target variable))
        for i in range(len(ig)-1):            
            # If the value is > threshold, keep in feature set
            if ig[i] < self.e_th:
                drop_columns.append(data.columns[i])

        return data.drop(columns=drop_columns)


