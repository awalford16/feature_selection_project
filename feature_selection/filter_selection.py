import sklearn.feature_selection as fs
import pandas as pd
import numpy as np
from feature_selection.multivariate import MultivariateSelection
from feature_selection.univariate import UnivariateSelection

# Facade pattern class to encapsulate different feature selection functionality
class FilterSelection():
    # Initialise minimum feature count
    def __init__(self, min_features):
        self.us = UnivariateSelection(min_features)
        self.ms = MultivariateSelection(min_features)

    def chi2(self, data, target_data):
        return self.us.chi_square_selection(data, target_data)

    def mi(self, data, target_data):
        return self.us.mi_selection(data, target_data)

    def rf(self, data, target_data):
        top_features = self.ms.relief_selection(data.to_numpy(), target_data.to_numpy())
        return data[data.columns[top_features]]

    def mrmr(self, data):
        return self.ms.mrmr_selection(data)

    


    
