class Comparison:
    def __init__(self):
        self.best_method = None
        self.best_subset = []
        self.acc = 0
        self.sens = 0
        self.spec = 0

        self.fs_method_best = {}



    def compare(self, fs, subset, model):
        is_best_method = self.compare_method(fs, len(subset), model)

        # If it is not the best scores produced from that method then will not be best overall
        if not is_best_method:
            return None

        current_best = self.acc + self.sens
        model_score = model.acc + model.sens

        if (model_score > current_best):
            self.best_method = fs
            self.best_subset = subset.values
            self.acc = model.acc
            self.sens = model.sens
            self.spec = model.spec
        
        # If scores are equal, select smaller subset
        elif (model_score == current_best and (len(subset) < len(self.best_subset))):
            self.best_method = fs
            self.best_subset = subset.values
            self.acc = model.acc
            self.sens = model.sens
            self.spec = model.spec


    def compare_method(self, fs, subset, model):
        if fs not in self.fs_method_best:
            self.fs_method_best[fs] = {
                'subset': subset,
                'acc': model.acc,
                'sens': model.sens,
                'spec': model.spec
            }
            return True
        
        current_best = self.fs_method_best[fs]['acc'] + self.fs_method_best[fs]['sens']
        model_score = model.acc + model.sens

        if model_score > current_best:
            self.fs_method_best[fs] = {
                'subset': subset,
                'acc': model.acc,
                'sens': model.sens,
                'spec': model.spec
            }
            return True

        return False


    def display_method_result(self, fs):
        results = self.fs_method_best[fs]
        print(f'\n\n{fs} best results with {results["subset"]} features:')
        print(f'Accuracy: {(results.get("acc") * 100):.2f}%')
        print(f'Sensitivity: {(results.get("sens") * 100):.2f}%')
        print(f'Specificity: {(results.get("spec") * 100):.2f}%')

    # Display best overall results
    def display_results(self):
        print(f'\n\nBest Feature Selection Method: {self.best_method}')
        print(f'Best Feature Subset: {self.best_subset}')
        print(f'Accuracy: {(self.acc * 100):.2f}%')
        print(f'Sensitivity: {(self.sens * 100):.2f}%')
        print(f'Speficity: {(self.spec * 100):.2f}%')