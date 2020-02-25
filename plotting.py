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
    def box_plot(self, df, cols, target, title):
        plt.figure()
        df.boxplot(column=cols)
        plt.suptitle(title)
        plt.savefig(os.path.join('plots', f'boxplots_{self.box_number}.png'), format='png')
        self.box_number += 1
        plt.close()


    # Identiy outliers with Isolation Forest
    def rad_plot(self, df, cols, target_name):
        df2 = pd.DataFrame(df, columns=cols)
        
        # Change 0 and 1 values to negative and positive for clarity
        target = {0: 'Negative', 1: 'Positive'}
        df2[target_name] = df2[target_name].map(target)

        plt.figure()
        pd.plotting.radviz(df2, target_name, color=['green', 'blue'])
        plt.savefig(os.path.join('plots', f'rad_{self.hist_number}.png'), format='png')
        self.hist_number += 1
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
        plt.close()

    # Create bar chart for p-values of chi-square results
    def plot_fs(self, values, name, title):
        # Number figures for multiple plots
        plt.figure(2)
        values = values.sort_values()
        values.plot.bar(rot=45)
        plt.title(title)
        plt.xlabel('Feature')
        plt.ylabel('Score')
        plt.savefig(os.path.join('plots', f'{name}.png'))
        plt.close()