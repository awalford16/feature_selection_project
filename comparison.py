class Comparison:
    def __init__(self):
        self.best_method = None
        self.best_subset = []
        self.acc = 0
        self.sens = 0
        self.spec = 0

    def compare(self, fs, subset, model):
        current_best = self.acc + self.sens
        model_score = model.acc + model.sens

        if (model_score > current_best):
            self.best_method = fs
            self.best_subset = subset
            self.acc = model.acc
            self.sens = model.sens
            self.spec = model.spec
        elif (model_score == current_best and (len(subset) < len(self.best_subset))):
            self.best_method = fs
            self.best_subset = subset
            self.acc = model.acc
            self.sens = model.sens
            self.spec = model.spec

    def display_results(self):
        print(f'\n\nBest Feature Selection Method: {self.best_method}')
        print(f'Best Feature Subset: {self.best_subset}')
        print(f'Accuracy: {(self.acc * 100):.2f}%')
        print(f'Sensitivity: {(self.sens * 100):.2f}%')
        print(f'Speficity: {(self.spec * 100):.2f}%')