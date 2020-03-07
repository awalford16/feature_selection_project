from sklearn import metrics
from models.ann import NeuralNet
from models.forest import Forest
import matplotlib as plt

# Create abstract class for classification models
class Classification:
    # Appropriately assign values
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.xtrain = x_train
        self.ytrain = y_train
        self.xtest = x_test
        self.ytest = y_test

        if model == 1:
            self.model = Forest(500)
        elif model == 2:
            self.model = NeuralNet()
    

    # Train classification model
    def train_model(self):
        self.model.model.fit(self.xtrain, self.ytrain)


    # Predict future data
    def test_model(self):
        return self.model.model.predict(self.xtest)


    # Model learning process
    def run(self):
        for val in self.model.hyper_params:
            self.model.set_hyper_params(val)
            self.train_model()

            pred = self.test_model()

            acc, spec, sens = self.score(pred)
            self.print_results(acc, spec, sens)   

        
    # Calculate precision and recall for 
    def score(self, pred):
        acc = metrics.accuracy_score(self.ytest, pred)

        # Get true positives/negatives and false positives/negatives
        tn, fp, fn, tp = metrics.confusion_matrix(self.ytest, pred).ravel()

        # Perform sensitivity and specificity calculations   
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)

        return acc, spec, sens          


    # Print model results
    def print_results(self, acc, spec, sens):
        print(f'Accuracy: {acc:.2f}\nSpecificity: {spec:.2f}\nSensitivity: {sens:.2f}')
