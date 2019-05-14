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
    plot = ax.plot(x_points, y_points, scaley=False, **plot_kwargs)
    return plot


def summary_figure(mydict, subplot_time_length=300):
    # generate a summary figure of the training
    # calculate the number of subplots needed
    # subplot_time_length = 300  # 5 minutes
    init_time = mydict['Moving_az']['MovingAzimuthTimes'][0]
    final_time = mydict['Moving_az']['MovingAzimuthTimes'][mydict['Moving_az'].shape[0] - 1]
    duration = final_time - init_time
    number_of_subplots = int(floor(duration / subplot_time_length + 1))

    fig, axs = plt.subplots(number_of_subplots, 1, sharey=False, sharex=False, figsize=(18, 3 * number_of_subplots),
                            dpi=80, facecolor='w', edgecolor='k')
    for i in range(0, number_of_subplots):
        summary_plot(mydict, ax=axs[i])
        axs[i].set_xlim(init_time + i * subplot_time_length, init_time + (i + 1) * subplot_time_length)
        axs[i].set_ylim(-50, 60)
    axs[i].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(mydict['Main_name'] + '_summary', size=24)
    fig.text(0.5, 0, 'Time', ha='center')
    fig.text(0, 0.5, 'Moving Azimuth', va='center', rotation='vertical')

    return fig


def summary_plot(mydict, ax=None):
    """
    Create a summary plot with info about the training session
    :param mydict: dictionary containing the data
    :param ax: axis of the plot
    :return: axis with plot data
    """
    if ax is None:
        ax = plt.gca()
    ax.plot('MovingAzimuthTimes', 'MovingAzimuthValues', data=mydict['Moving_az'], color='blue', linewidth=1)
    ax.plot('TrialSideTimes', 'TrialSideVM', data=mydict['Trial_side'], color='grey', marker='.', linewidth=0.5)
    ax.plot(mydict['Target_reached'], len(mydict['Target_reached']) * [50], '.', color='green', alpha=.5)
    ax.plot(mydict['Wrong_reached'], len(mydict['Wrong_reached']) * [50], '.', color='red', alpha=.5)
    axvlines(mydict['Lick_events'], ax=ax, linewidth=0.2, color='gray')

    return ax



def PlotPsychPerformance(dataDif=None, dataPerf=None, predictDif=None, ax=None,
                         realPred=None, fakePred=None, label = 'data', **plot_kwargs):
    #Plots different various features of the psychometric performance
    
    if ax is None:
        ax = plt.gca()
    
    # This plots all the fake data:
    #plt.plot(predictDif, fakePred, 'k-', lw=0.5, alpha=0.2)
    
    # plot percentiles if fake data is provided
    if fakePred is not None:
        percentiles = np.percentile(fakePred, [2.5, 97.5], axis=1).T
        ax.fill_between(predictDif.reshape(-1), percentiles[:,0], percentiles[:,1], alpha = 0.2, **plot_kwargs)

    # plot the psychometric performance if the predictions are provided
    if realPred is not None:
        ax.plot(predictDif.reshape(-1), realPred, '-', **plot_kwargs)
    
    # plot the data
    if dataPerf is not None:
        ax.plot(dataDif, dataPerf, 'o', ms = 8, label = label, **plot_kwargs)

    # make the plot pretty
    if dataDif is not None:
        ax.set_xticks(dataDif)
    ax.set_ylabel('% Rightward choices')
    ax.set_xlabel('% High tones')
    ax.set_xlim(0., 100.)
    ax.set_ylim(-2., 102.)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
    ax.tick_params(top='off', bottom='on', left='on', right='off', labelleft='on', labelbottom='on')
    # get rid of the frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return ax