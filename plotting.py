import matplotlib.pyplot as plt
import numpy as np

class Plot:
    # Plot pearson correlation
    def plot_pc(self, corr, names):
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