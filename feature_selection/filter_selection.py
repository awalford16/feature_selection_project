import sklearn.feature_selection as fs
import pandas as pd
import numpy as np
import operator
from feature_selection.multivariate import MultivariateSelection
from feature_selection.univariate import UnivariateSelection
from plotting import Plot

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

        # Plot relieff scores
        relief_plot = Plot()
        scores = pd.Series(feature_scores, index=data.columns)
        relief_plot.plot_fs(scores, "ReliefF", 'Dataset 1 ReliefF Scores')
        del relief_plot

        return self.drop_columns(data, feature_scores, operator.gt, 800)

    def mrmr(self, data):
        return self.ms.mrmr_selection(data)


    # Drop columns from dataset which dont surpass statistical threshold
    def drop_columns(self, data, scores, compare, threshold):
        drop_columns = []

        # Loop through each of the IG values (Ignore last value (target variable))
        for i in range(len(scores)):
            # If the value doesnt meet the threshold, append to list of columns to drop
            if not compare(scores[i], threshold):
                drop_columns.append(data.columns[i])

        return data.drop(columns=drop_columns)


    
