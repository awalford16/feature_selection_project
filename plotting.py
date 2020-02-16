import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class Plot:
    # Create new images for each new dataset
    def __init__(self):
        self.stack_number = 0

    def stack_plot(self, df, col1, col2):
        df2 = pd.DataFrame(df, columns=[col1, col2])

        # Change 0 and 1 values to negative and positive for clarity
        target = {0: 'Negative', 1: 'Positive'}
        df2[col2] = df2[col2].map(target)

        # Create groups of data
        df2 = df2.groupby([col2, col1])[col2].count().unstack(col1)

        # Create plot and configure labels
        stack = df2.plot(kind='bar', stacked=True, rot=0)
        stack.legend(['Female', 'Male'])
        stack.set_xlabel('Heart Disease')
        stack.set_ylabel('Count')

        fig = stack.get_figure()
        fig.savefig(os.path.join('plots', f'stack_plot_{self.stack_number}.png'))
        self.stack_number += 1

    # Plot pearson correlation
    def plot_pc(self, corr, names):
        figure = plt.figure(1)
        ax = figure.add_subplot(111)
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        figure.colorbar(cax)
        ticks = np.arange(0,14,1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticklabels(names)
        plt.savefig(os.path.join('plots','corr.png'))

    # Create bar chart for p-values of chi-square results
    def plot_chi(self, p_values):
        # Number figures for multiple plots
        plt.figure(2)
        p_values.plot.bar(rot=0)
        plt.savefig(os.path.join('plots', 'chi_pvalues.png'))