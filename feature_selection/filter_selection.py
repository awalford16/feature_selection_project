import sklearn.feature_selection as fs
import pandas as pd
import numpy as np
from plotting import Plot

class FilterSelection:
    # Initialise threshold values
    def __init__(self, correlation_threshold = 0.4, variance_threshold=0.4, entropy_threshold=0.2):
        self.c_th = correlation_threshold
        self.var_th = variance_threshold
        self.e_th = entropy_threshold

    # Use data type of each feature to determine which correlation selection method
    def corr_selection(self, data, target_data, target_name):
        # Initialise arrays for different column data types
        cat_arr = pd.DataFrame()
        num_arr = pd.DataFrame()

        # Loop through eah feature
        for col in data.columns:
            # Detect columb data type
            if data[col].dtype.name == 'category':
                # Store column in category dataframe
                cat_arr[col] = data[col]
                continue
            
            # Store column data in non categorical dataframe
            num_arr[col] = data[col]

        # Use pearson correlation for numerical data
        pc_cols = self.pearson_correlation(num_arr, target_data, target_name)

        # Use chi-square for categorical data
        chi_cols = self.chi_square_selection(cat_arr, target_data)

        # Concatinate results into one dataframe and return result
        return pd.concat([pc_cols, chi_cols], axis=1)


    # Pearson correlation code available from: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
    def pearson_correlation(self, data, target_data, target_name):
        # Merge data and target data into same dataframe (Convert target data to float for correlation)
        data[target_name] = target_data.astype('float')

        # get correlations between data features
        corr = data.corr()
        names = list(data.columns)
        
        # Create plot of pearson correlation
        crr_plt = Plot()
        crr_plt.plot_pc(corr, names)
        del crr_plt

        # identify correlations with output variable
        target = corr[target_name][:-1]

        # Drop columns from dataframe that do not meet threshold
        return self.drop_columns(data.loc[:, data.columns != target_name], target, self.c_th)


    # Chi square analyses correlation with category data
    def chi_square_selection(self, data, target_data):
        # Calculate chi scores for each feature
        chi = fs.chi2(data, target_data)

        # Store p values in array
        p_values = pd.Series(chi[1], index = data.columns)
        print(p_values)

        # Plot p-values to visualise which are more valuable columns
        chi_plt = Plot()
        chi_plt.plot_chi(p_values)
        del chi_plt

        # Drop features below threshold (95% confidence, set alpha = 0.05)
        return self.drop_columns(data, p_values, 0.05)
        

    # Variance filter selection
    def variance_selection(self, data):
        selector = fs.VarianceThreshold(threshold=self.var_th)
        return selector.fit_transform(data)


    # Filter feature selection based on information gain
    def entropy_selection(self, data, target_data):
        ig = fs.mutual_info_classif(data, target_data)
        return self.drop_columns(data, ig, self.e_th)


    # Drop columns from data if score is less than threshold
    def drop_columns(self, data, scores, threshold):
        drop_columns = []

        # Loop through each of the IG values (Ignore last value (target variable))
        for i in range(len(scores) - 1):
            # If the value is < threshold, add to list of columns to remove
            if scores[i] < threshold:
                drop_columns.append(data.columns[i])

        return data.drop(columns=drop_columns)


