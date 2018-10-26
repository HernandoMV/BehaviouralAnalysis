# plot_utils
import numpy as np
import matplotlib.pyplot as plt
from math import *


def axvlines(xs, ax=None, **plot_kwargs):
    """
    Function from StackExchange
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param ax: The axis (or none to use gca)
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scaley = False, **plot_kwargs)
    return plot


def summary_figure(mydict):
    # generate a summary figure of the training
    # calculate the number of subplots needed
    subplot_time_length = 300  # 5 minutes
    init_time = mydict['Moving_az']['MovingAzimuthTimes'][0]
    final_time = mydict['Moving_az']['MovingAzimuthTimes'][mydict['Moving_az'].shape[0] - 1]
    duration = final_time - init_time
    number_of_subplots = int(floor(duration / subplot_time_length + 1))

    fig, axs = plt.subplots(number_of_subplots, 1, sharey=False, sharex=False, figsize=(18, 3 * number_of_subplots),
                            dpi=80, facecolor='w', edgecolor='k')
    for i in range(0, number_of_subplots):
        axs[i].plot('MovingAzimuthTimes', 'MovingAzimuthValues', data=mydict['Moving_az'], color='blue', linewidth=1)
        axs[i].plot('TrialSideTimes', 'TrialSideVM', data=mydict['Trial_side'], color='grey', marker='.', linewidth=0.5)
        axs[i].plot(mydict['Target_reached'], len(mydict['Target_reached']) * [100], '.', color='green', alpha=.5)
        axs[i].plot(mydict['Wrong_reached'], len(mydict['Wrong_reached']) * [100], '.', color='red', alpha=.5)
        axs[i].set_xlim(init_time + (i) * subplot_time_length, init_time + (i + 1) * subplot_time_length)
        axvlines(mydict['Lick_events'], ax=axs[i], linewidth=0.2, color='gray')
    axs[i].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(mydict['Main_name'] + '_summary', size=24)
    fig.text(0.5, 0, 'Time', ha='center')
    fig.text(0, 0.5, 'Moving Azimuth', va='center', rotation='vertical')

    return fig
