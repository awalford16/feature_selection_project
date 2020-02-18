import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats

class Plot:
    # Create new images for each new dataset
    def __init__(self):
        self.stack_number = 0
        self.box_number = 0
        self.hist_number = 0

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


    # Box plot numerical data with target
    def box_plot(self, df, cols, target):
        plt.figure()
        df.boxplot(column=cols)
        plt.savefig(os.path.join('plots', f'boxplots_{self.box_number}.png'), format='png')
        self.box_number += 1
        plt.close()


    # Identiy outliers with Isolation Forest
    def iso_forest(self, df, cols):
        th = 3
        z_score = np.abs(stats.zscore(df))
        
        _, ax = plt.subplots()
        ax.matshow(z_score)
        plt.savefig(os.path.join('plots', f'iso_forest_{self.hist_number}.png'), format='png')
        plt.close()

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