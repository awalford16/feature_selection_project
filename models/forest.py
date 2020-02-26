from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

class Forest:
    def __init__(self, trees, leaf_samples):
        self.model = RandomForestClassifier(n_estimators=trees, min_samples_leaf=leaf_samples)
