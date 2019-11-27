import matplotlib.pyplot as plt
import sklearn.feature_selection as fs
import pandas as pd
import numpy as np

# Pearson correlation code available from: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

def pearson_correlation(data, target_data, target_name, c):
    # Merge data and target data into same dataframe
    data[target_name] = target_data.values

    # get correlations between data features
    corr = data.corr()
    names = list(data.columns)
    get_pc_graph(corr, names)

    # identify correlations with output variable
    target = corr[target_name][:-1]

    return target[(abs(target) > c)]

# Plot pearson correlation
def get_pc_graph(corr, names):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    figure.colorbar(cax)
    ticks = np.arange(0,14,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    plt.show()

# VAriance filter selection
def variance_selection(data, threshold):
    selector = fs.VarianceThreshold(threshold=threshold)
    return selector.fit_transform(data)
