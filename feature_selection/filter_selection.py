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
        pc_cols = self.anova(num_arr, target_data, target_name)

        # Concatinate results into one dataframe and return result
        return pd.concat([pc_cols], axis=1)


    # Pearson correlation code available from: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
    def anova(self, data, target_data, target_name):
        # Merge data and target data into same dataframe (Convert target data to float for correlation)
        #data[target_name] = target_data.astype('float')

        # get correlations between data and target using ANOVA
        corr = fs.f_classif(data, target_data)
        names = list(data.columns)

        print(corr[0])
        
        # Create plot of pearson correlation
        crr_plt = Plot()
        crr_plt.plot_pc(corr[0], names)
        del crr_plt

        # Drop columns from dataframe that do not meet threshold
        return self.drop_columns(data.loc[:, data.columns != target_name], corr[0], self.c_th)
        

    # Variance filter selection
    def variance_selection(self, data):
        selector = fs.VarianceThreshold(threshold=self.var_th)
        return selector.fit_transform(data)



    # Drop columns from data if score is less than threshold
    def drop_columns(self, data, scores, threshold):
        drop_columns = []

        # Loop through each of the IG values (Ignore last value (target variable))
        for i in range(len(scores) - 1):
            # If the value is < threshold, add to list of columns to remove
            if scores[i] < threshold:
                drop_columns.append(data.columns[i])

        return data.drop(columns=drop_columns)


