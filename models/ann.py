from sklearn.neural_network import MLPClassifier
from sklearn import metrics 

class NeuralNet:
    def __init__(self, nodes):
        self.model = MLPClassifier(solver='sgd', hidden_layer_sizes=nodes)
        self.hyper_params = [50, 100, 500, 1000]

    def set_hyper_params(self, val):
        #print(f'{val} hidden neurons.')
        self.model.set_params(hidden_layer_sizes=val)

