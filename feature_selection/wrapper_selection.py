from mlxtend.feature_selection import SequentialFeatureSelector as sfs

class WrapperSelection:
    def __init__(self, min_features, model):
        self.k = min_features
        self.model = model

    # perform wrapper forward selection
    def forward_select(self, data, target_data):
        # Set arguments for function and pass as array
        args = [True, data, target_data]

        return self.exec_wrapper(*args)

    # Perform wrapper backwards selection
    def backward_select(self, data, target_data):
        args = [False, data, target_data]
        
        return self.exec_wrapper(*args)


    def exec_wrapper(self, forward, data, target_data):
        select = sfs(self.model, k_features=self.k, forward=forward, floating=False, scoring='accuracy', cv=0)
        features = select.fit(data, target_data)

        # Convert tuple output to array
        feature_i = []
        feature_i.extend(features.k_feature_idx_)
        
        #Return all rows for selected features
        return data.iloc[:, feature_i]


# @article{raschkas_2018_mlxtend,
#   author       = {Sebastian Raschka},
#   title        = {MLxtend: Providing machine learning and data science 
#                   utilities and extensions to Python’s  
#                   scientific computing stack},
#   journal      = {The Journal of Open Source Software},
#   volume       = {3},
#   number       = {24},
#   month        = apr,
#   year         = 2018,
#   publisher    = {The Open Journal},
#   doi          = {10.21105/joss.00638},
#   url          = {http://joss.theoj.org/papers/10.21105/joss.00638}
# }