import sklearn.feature_selection as fs
import pandas as pd
import numpy as np
import operator
from feature_selection.multivariate import MultivariateSelection
from feature_selection.univariate import UnivariateSelection

# Facade pattern class to encapsulate different feature selection functionality
class FilterSelection():
    # Initialise minimum feature count
    def __init__(self, min_features):
        self.us = UnivariateSelection(min_features)
        self.ms = MultivariateSelection(min_features)

    def chi2(self, data, target_data):
        feature_scores = self.us.chi_square_selection(data, target_data)
        return self.drop_columns(data, feature_scores, operator.lt, 0.1)

    def mi(self, data, target_data):
        feature_scores = self.us.mi_selection(data, target_data)
        return self.drop_columns(data, feature_scores, operator.gt, 0.08)

    def rf(self, data, target_data):
        feature_scores = self.ms.relief_selection(data.to_numpy(), target_data.to_numpy())
        return self.drop_columns(data, feature_scores, operator.gt, 500)

    def mrmr(self, data):
        return self.ms.mrmr_selection(data)


    # Function to drop columns
    def drop_columns(self, data, scores, compare, threshold):
        drop_columns = []

        # Loop through each of the IG values (Ignore last value (target variable))
        for i in range(len(scores)):
            # If the value doesnt meet the threshold, append to list of columns to drop
            if compare(scores[i], threshold):
                print(f'{scores[i]}')
                drop_columns.append(data.columns[i])

        return data.drop(columns=drop_columns)


    
