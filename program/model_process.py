from feature_selection.wrapper_selection import WrapperSelection
from models.forest import Forest

class ModelProcess:
    # Appropriately assign values
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.xtrain = x_train
        self.ytrain = y_train
        self.xtest = x_test
        self.ytest = y_test

        if model == 1:
            self.model = Forest(500, 10)
        # else set to ANN
    
    def run(self):
        # TODO: Perform cross validation
        self.model.train(self.xtrain, self.ytrain)

        pred = self.model.test(self.xtest)

        acc, spec, sens = self.model.score(self.ytest, pred)
        self.print_results(acc, spec, sens)
    
    # Print model results
    def print_results(self, acc, spec, sens):
        print(f'Accuracy: {acc}\nSpecificity: {spec}\nSensitivity: {sens}')


