from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import warnings
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

        self.model_type = model

        self.model = self.create_model(50)

        # Ignore warnings from Scikitlearn
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    

    # Train classification model
    def train_model(self):
        self.model.model.fit(self.xtrain, self.ytrain)


    # Predict future data
    def test_model(self):
        return self.model.model.predict(self.xtest)

    
    # Create a classification model
    def create_model(self, params):
        if self.model_type == 1:
            return Forest(params)
        if self.model_type == 2:
            return NeuralNet(params)

        return None


    # Model learning process
    def run(self):
        # GridSearch identifies the best hyper-parameters for a given model
        grid_search = GridSearchCV(self.model.model, self.model.hyper_params, cv=5)
        best_model = grid_search.fit(self.xtrain, self.ytrain)

        print(f"Best Hyper-Parameter Value: {best_model.best_estimator_.get_params()['min_samples_leaf']}.")
        self.model.model = best_model.best_estimator_

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
