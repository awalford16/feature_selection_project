import matplotlib.pyplot as plt
import numpy as np
import os

class Plot:
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