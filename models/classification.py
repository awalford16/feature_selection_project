from sklearn import metrics
from models.ann import NeuralNet
from models.forest import Forest

# Create abstract class for classification models
class Classification:
    # Appropriately assign values
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.xtrain = x_train
        self.ytrain = y_train
        self.xtest = x_test
        self.ytest = y_test

        if model == 1:
            self.model = Forest(500, 10).model
        elif model == 2:
            self.model = NeuralNet(500).model
    
    # Select classification model


    # Train classification model
    def train_model(self):
        self.model.fit(self.xtrain, self.ytrain)

    # Predict future data
    def test_model(self):
        return self.model.predict(self.xtest)

    # Calculate precision and recall for 
    def score(self, y_test, y_pred):
        acc = metrics.accuracy_score(y_test, y_pred)
        sens = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)

        return acc, sens, recall

    # Model learning process
    def run(self):
        # TODO: Perform cross validation
        self.train_model()

        pred = self.test_model()

        acc, spec, sens = self.score(self.ytest, pred)
        self.print_results(acc, spec, sens)
    
    # Print model results
    def print_results(self, acc, spec, sens):
        print(f'Accuracy: {acc}\nSpecificity: {spec}\nSensitivity: {sens}')
