from sklearn import metrics
import abc

# Create abstract class for classification models
class Classification(abc.ABC):
    # Train the classification model
    @abc.abstractmethod
    def train(self, x, y):
        pass

    # Test the classification model
    @abc.abstractmethod
    def test(self, x):
        pass
    
    # Calculate precision and recall for 
    def score(self, y_test, y_pred):
        acc = metrics.accuracy_score(y_test, y_pred)
        sens = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)

        return acc, sens, recall