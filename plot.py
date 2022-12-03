import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def barplot(ax, data, *, x, y, title='', ylabel='count', palette='viridis'):
    """
    palette: viridis, Blues_r, rocket_r, Spectral
    """
    sns.barplot(
        ax=ax,
        x=x,
        y=y,
        data=data,
        edgecolor='black',
        ci=False,
        palette=palette,
    )
    ax.bar_label(ax.containers[0])
    ax.set(
        title=title,
        yticks=[],
        ylabel=ylabel,
        xlabel='',
    )

def conf_matrix(ax, y_true, y_pred, *, ticklabels=None):
    ax.set_title('Binary sentiment')
    cf = confusion_matrix(
            y_true, y_pred,
            normalize='all',
        )
    sns.heatmap(
        cf/np.sum(cf),
        annot=True,
        cbar=False,
        ax=ax,
        fmt=".2%",
        cmap="Blues",
    )
    if ticklabels:
        ax.xaxis.set_ticklabels(ticklabels)
        ax.yaxis.set_ticklabels(ticklabels)