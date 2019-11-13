import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Pearson correlation code available from: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

def pearson_correlation(data):
    # get correlations between data features
    corr = data.corr()
    names = list(data.columns)

    # Plot matrix
    figure = plt.figure()
    ax = figure.add_subplot(111)
    cax = ax.matshow(corr, vmin=0, vmax=1)
    figure.colorbar(cax)
    ticks = np.arange(0,14,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()
