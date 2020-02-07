import sklearn.feature_selection as fs
import pandas as pd
import numpy as np
from feature_selection.multivariate import MultivariateSelection
from feature_selection.univariate import UnivariateSelection

# Facade pattern class to encapsulate different feature selection functionality
class FilterSelection(UnivariateSelection, MultivariateSelection):
    # Initialise minimum feature count
    def __init__(self, min_features):
        self.k = min_features

    def chi2(self, data, target_data):
        return UnivariateSelection.chi_square_selection(self, data, target_data)

    def mi(self, data, target_data):
        return UnivariateSelection.mi_selection(self, data, target_data)

    def rf(self, data, target_data):
        top_features = MultivariateSelection.relief_selection(self, data.to_numpy(), target_data.to_numpy())
        return data[data.columns[top_features]]

    def mrmr(self, data):
        return MultivariateSelection.mrmr_selection(self, data)

    


    
