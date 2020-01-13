# plot_utils
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import *
import pandas as pd
sys.path.append("../")  # for search in the path
import BehaviouralAnalysis.utils.custom_functions as cuf
import OpenEphys_Analysis.utils.custom_functions as OE
import seaborn as sns

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


def summary_figure_joystick(mydict, subplot_time_length=300):
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


def summary_plot_joystick(mydict, ax=None):
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
                         realPred=None, fakePred=None, errorBars=None,
                         label='data', **plot_kwargs):
    # Plots different various features of the psychometric performance
    
    if ax is None:
        ax = plt.gca()

    # This plots all the fake data:
    # plt.plot(predictDif, fakePred, 'k-', lw=0.5, alpha=0.2)

    # plot percentiles if fake data is provided
    if fakePred is not None:
        percentiles = np.percentile(fakePred, [2.5, 97.5], axis=1).T
        ax.fill_between(predictDif.reshape(-1), percentiles[:, 0], percentiles[:, 1], alpha=0.2, **plot_kwargs)

    # plot the psychometric performance if the predictions are provided
    if realPred is not None:
        ax.plot(predictDif.reshape(-1), realPred, '-', **plot_kwargs)

    # plot the error bars
    if errorBars is not None:
        for i, EBlength in enumerate(errorBars):
            ax.plot([dataDif[i], dataDif[i]], [dataPerf[i] - EBlength / 2, dataPerf[i] + EBlength / 2],
                    '-', **plot_kwargs)

    # plot the data
    if dataPerf is not None:
        ax.plot(dataDif, dataPerf, 'o', ms=8, label=label, **plot_kwargs)

    # make the plot pretty
    if dataDif is not None:
        ax.set_xticks(dataDif)
    ax.set_ylabel('% Rightward choices')
    ax.set_xlabel('% High tones')
    ax.set_xlim(0., 100.)
    ax.set_ylim(-2., 102.)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
    ax.tick_params(which='both', top='off', bottom='on', left='on', right='off',
                   labelleft='on', labelbottom='on')
    # get rid of the frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    # invert the axis as it looks more natural for a psychometric curve
    ax.invert_xaxis()

    return ax


def summary_matrix(df):
    # Initialize lists to save important data
    DifficultyValues = []
    PerformanceValues = []

    # process data from all experiments
    for counter, session in enumerate(pd.unique(df['SessionTime'])):
        predictDif, PsyPer, fakePredictions, predictPer, _ = \
            cuf.PP_ProcessExperiment(df[df['SessionTime'] == session])

        # append to lists, only the normal trials
        DifficultyValues.append(PsyPer['Difficulty'])
        PerformanceValues.append(PsyPer['Performance'])

        OE.update_progress(counter / len(pd.unique(df['SessionTime'])))

    OE.update_progress(1)

    # calculate difficulty levels
    difLevels = np.unique(np.concatenate(DifficultyValues).ravel())
    # Initialize the matrix
    matToPlot = np.full([len(difLevels), len(DifficultyValues)], np.nan)
    # Loop to fill it
    for i, dif in enumerate(difLevels):
        for j, per in enumerate(PerformanceValues):
            if dif in DifficultyValues[j]:
                idxOfDif = np.where(DifficultyValues[j] == dif)[0][0]
                matToPlot[i, j] = per[idxOfDif]

    # Transform to dataframe
    dfToPlot = pd.DataFrame(matToPlot)
    dfToPlot = dfToPlot.set_index(difLevels)  # set row names
    dfToPlot.columns = pd.unique(df['SessionTime'])  # set col names

    return dfToPlot


def summary_plot(dfToPlot, AnimalDF, ax):
    sns.set(style="white")
    sp = sns.heatmap(dfToPlot, linewidth=0.001, square=True, cmap="coolwarm",
                cbar_kws={"shrink": 0.6, 'label': '% Rightward choices'},
                ax = ax, vmin=0, vmax=100)
    # TODO: check that the size is proportional (area vs radius)
    # recalculate the number of trials as some might get grouped if they are on the same day. Do all below with the dataframe
    Protocols = [pd.unique(AnimalDF[AnimalDF['SessionTime']==session]['Protocol'])[0] \
                for session in pd.unique(AnimalDF['SessionTime'])]
    ntrialsDistribution = [len(AnimalDF[AnimalDF['SessionTime']==session]) for session in pd.unique(AnimalDF['SessionTime'])]
    Stimulations = [pd.unique(AnimalDF[AnimalDF['SessionTime']==session]['Stimulation'])[0] \
                for session in pd.unique(AnimalDF['SessionTime'])]
    Muscimol = [pd.unique(AnimalDF[AnimalDF['SessionTime']==session]['Muscimol'])[0] \
                for session in pd.unique(AnimalDF['SessionTime'])]
    difLevels = dfToPlot.index
    AnimalName = str(pd.unique(AnimalDF.AnimalID)[0])
    AnimalGroup = str(pd.unique(AnimalDF.ExperimentalGroup)[0])

    for pr_counter, prot in enumerate(np.unique(Protocols)):
        protIdx = [i for i, x in enumerate(Protocols) if x == prot]
        ax.scatter([x + 0.5 for x in protIdx], np.repeat(len(difLevels)+0.5, len(protIdx)), marker='o',
                s=[ntrialsDistribution[x]/5 for x in protIdx], label = prot) 
    # label the opto sessions
    for st_counter, stim in enumerate(np.unique(Stimulations)):
        stimIdx = [i for i, x in enumerate(Stimulations) if x == stim]
        ax.scatter([x + 0.5 for x in stimIdx], np.repeat(len(difLevels)+1.5, len(stimIdx)), marker='*', s=100, label = stim)
    # label the muscimol sessions
    for mus_counter, mus in enumerate(np.unique(Muscimol)):
        musIdx = [i for i, x in enumerate(Muscimol) if x == mus]
        ax.scatter([x + 0.5 for x in musIdx], np.repeat(len(difLevels)+2.5, len(musIdx)), marker='P', s=100, label = mus)
        
    ax.legend(loc=(0,1), borderaxespad=0., ncol = 5, frameon=True)
    ax.set_ylim([0, len(difLevels)+3])
    plt.ylabel('% High Tones')
    plt.xlabel('Session')
    sp.set_yticklabels(sp.get_yticklabels(), rotation=0)
    sp.set_xticklabels(sp.get_xticklabels(), rotation=45, horizontalalignment="right")
    sp.set_title(AnimalName + ' - ' + AnimalGroup + '\n\n', fontsize=20, fontweight=0)
    
    return ax
