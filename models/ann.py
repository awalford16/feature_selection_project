from sklearn.neural_network import MLPClassifier
from sklearn import metrics 

class NeuralNet:
    def __init__(self, h_layers):
        self.model = MLPClassifier(solver='sgd', hidden_layer_sizes=h_layers)

