from sklearn import feature_selection as fs
import pandas as pd
from plotting import Plot

class UnivariateSelection:
    # Chi Square Feature Selection
    def chi_square_selection(self, data, target_data):
        # Calculate chi scores for each feature
        chi = fs.chi2(data, target_data)

        # Store p values in array
        p_values = pd.Series(chi[1], index = data.columns)

        # Plot p-values to visualise which are more valuable columns
        chi_plt = Plot()
        chi_plt.plot_chi(p_values)
        del chi_plt

        # Drop features below threshold (95% confidence, set alpha = 0.05)
        return self.selector(fs.chi2, data, target_data)


    # Mutual Information Feature Selection
    def mi_selection(self, data, target_data):
        return self.selector(fs.mutual_info_classif, data, target_data)
        
    
    def selector(self, method, data, target_data):
        selector = fs.SelectKBest(method, k = self.k)
        selector.fit(data, target_data)

        # Get name of columns to keep
        cols = selector.get_support(indices=True)

        print(cols)

        # Overwrite exisitng dataframe with selected columns
        return data[data.columns[cols]]