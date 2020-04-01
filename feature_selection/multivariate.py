import pymrmr
from ReliefF import ReliefF as rf
import numpy as np
from plotting import Plot
import pandas as pd

class MultivariateSelection():
    def __init__(self, k):
        self.k = k

    # Feature selection through max relevance min redundancy
    def mrmr_selection(self, data):
        cols = pymrmr.mRMR(data, 'MID', self.k)
        return data[cols]

    # Feature selection using ReliefF algorithm    
    def relief_selection(self, data, target_data):
        w = rf(n_features_to_keep=self.k)
        w.fit(data, target_data)

        # absolute so all scores are positive
        return w.feature_scores

