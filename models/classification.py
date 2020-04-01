from sklearn import metrics
from sklearn.model_selection import cross_val_score
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
        
        # Performance scores
        self.acc = 0
        self.sens = 0
        self.spec = 0

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
        # Initialise best hyper-parameter values
        best_params = self.model.hyper_params[0]
        best_score = 0

        for val in self.model.hyper_params:
            self.model.set_hyper_params(val)
            
            # Predict accuracy score with cross-validation
            val_scores = cross_val_score(self.model.model, self.xtrain, self.ytrain, cv=5)

            # Determine best hyper-parameters
            if val_scores.mean() > best_score:
                best_score = val_scores.mean()
                best_params = val

        print(f"Best hyper-parameter value: {best_params}")
        self.model.set_hyper_params(best_params)
        self.train_model()

        pred = self.test_model()
        self.score(pred)  

        
    # Calculate precision and recall for 
    def score(self, pred):
        self.acc = metrics.accuracy_score(self.ytest, pred)

        # Get true positives/negatives and false positives/negatives
        tn, fp, fn, tp = metrics.confusion_matrix(self.ytest, pred).ravel()

        # Perform sensitivity and specificity calculations   
        self.sens = tp / (tp + fn)
        self.spec = tn / (tn + fp)        


    # Print model results
    def print_results(self):
        print(f'Accuracy: {self.acc:.2f}\nSpecificity: {self.spec:.2f}\nSensitivity: {self.sens:.2f}')
