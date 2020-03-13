from feature_selection.filter_selection import FilterSelection
import pandas as pd

class FeatureSelection:
    def __init__(self, x, y, k):
        self.feature_sets = {}
        self.k = k
        self.x = x
        self.y = y

    def display_features(self, model, features):
        print(f'{model} Features: {features.values}')

    # Create function based switch method for selecting fs method
    def exec_fs(self):
        fs = FilterSelection(self.k)

        # Array of executable methods
        fs_methods = {
            'chi-square': fs.chi2(self.x, self.y), 
            'mutual-info': fs.mi(self.x, self.y),
            'mrmr': fs.mrmr(self.x),
            'relieff': fs.rf(self.x, self.y)
        }

        for method in fs_methods:
            print(f'\nExecuting {method} selection')
            subset = (fs_methods[method]).columns
            self.feature_sets[method] = subset
            self.display_features(method, subset)
