import pymrmr
from ReliefF import ReliefF

class MultivariateSelection():
    # Feature selection through max relevance min redundancy
    def mrmr_selection(self, data):
        cols = pymrmr.mRMR(data, 'MIQ', self.k)
        return data[cols]

    def relief_f_selection(self, data, target_data):
        w = ReliefF(n_neighbors=data.shape[1], n_features_to_keep=self.k)
        return w.fit_transform(data, target_data)
