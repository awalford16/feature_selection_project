import pymrmr
from skrebate import ReliefF
from ReliefF import ReliefF as rf

class MultivariateSelection():
    # Feature selection through max relevance min redundancy
    def mrmr_selection(self, data):
        cols = pymrmr.mRMR(data, 'MIQ', self.k)
        return data[cols]

    def relief_f_selection(self, data, target_data):
        # Determine weights between features
        w = ReliefF(n_neighbors=data.shape[1], n_features_to_select=self.k)
        
        # Return array of indices resembling top features
        return (w.fit(data, target_data)).top_features_[:self.k]

    def relief_selection(self, data, target_data):
        w = rf(n_features_to_keep=self.k)
        w.fit(data, target_data)
        return w.top_features[:self.k]

