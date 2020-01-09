from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
from models.classification import Classification

class Forest(Classification):
    def __init__(self, trees, leaf_samples):
        self.model = RandomForestClassifier(n_estimators=trees, min_samples_leaf=leaf_samples)

    # Train the forest using training data
    def train(self, x, y):
        self.model.fit(x,y)

    # Test the forest with testing data
    def test(self, x):
        return self.model.predict(x)
