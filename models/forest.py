from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

class Forest:
    def __init__(self, trees):
        self.model = RandomForestClassifier(n_estimators=trees)
        self.hyper_params = [5, 10, 20, 50]

    def set_hyper_params(self, val):
        print(f'{val} samples per leaf.')
        self.model.set_params(min_samples_leaf=val)