from utils import custom_functions as cuf
import os
import glob
import ntpath
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import chain


# Animals to analyze
# animals_to_analyze = [''.join(['A2A', str(x)]) for x in range(10, 28)]
# animals_to_analyze = [''.join(['SP', str(x)]) for x in range(19, 36)]
# animals_to_analyze = [''.join(['QPM00', str(x)]) for x in range(1, 10)]
# animals_to_analyze = ['KAA682', 'KAA683', 'KAA684', 'KAA685', 'KAA686', 'KAA757',
#                       'C01', 'C02', 'C03', 'SomFlp04', 'SomFlp06']
animals_to_analyze = [''.join(['A2A', str(x)]) for x in [11,12,13,16,17,18,20,22]] + \
                     ['C01', 'C02', 'C03', 'SomFlp04', 'SomFlp06']
# animals_to_analyze = [''.join(['A2A', str(x)]) for x in [10,14,15,19,21,23,24,25,26,27]] + \
#                      ['C01', 'C02', 'C03', 'SomFlp04', 'SomFlp06']

# Name of batch
batch_name = 'D2-caspase_Apr2021'
# batch_name = 'PF-caspase'
# batch_name = 'Inference'
# create empty list
DataFrames = []
# Write the experimental groups
#eg_list = list(np.repeat('Sthita_cohort', 17))
# eg_list = list(np.repeat('Quentin_cohort', 10))
# eg_list = ['Cortex_Buffer', 'D2Cre-caspase', 'D2Cre-caspase', 'D2Cre-caspase',
#            'Cortex_Buffer', 'Cortex_Buffer', 'D2Cre-caspase', 'D2Cre-caspase', 'D2Cre-caspase',
#            'Cortex_Buffer', 'D2Cre-caspase', 'Cortex_Buffer', 'D2Cre-caspase',
#            'Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer']
# eg_list = list(np.repeat('drd1cre-controls', 6)) + list(np.repeat('6OHDA-controls', 5))
eg_list = list(np.repeat('d2-caspase', 8)) + list(np.repeat('6OHDA-controls', 5))


BpodProtocol = '/Two_Alternative_Choice/'
# Main directory of behavioural data to be saved, now computer dependent
GeneralDirectory = cuf.get_data_folder() + '/Behavioural_Data/Bpod_data/'
InputDirectory = '/mnt/hernandom/winstor/swc/sjones/data/bpod_raw_data/'

# Create out directory if it does not exist
batch_output = GeneralDirectory + batch_name + '_Analysis/'
if not os.path.isdir(batch_output):
    os.mkdir(batch_output)

# loop through
for egc, AnimalID in enumerate(animals_to_analyze):
    print('----- Analyzing animal {}'.format(AnimalID))
    # get experimental group
    ExpGroup = eg_list[egc]
    # create output
    outputDir = GeneralDirectory + AnimalID + BpodProtocol
    if not os.path.isdir(outputDir):
        os.mkdir(GeneralDirectory + AnimalID)
        os.mkdir(outputDir)

    ##
    # Read previously analyzed (or just read and converted) data
    update = False
    # in case this variable is created by the analysis of other animals:
    exp_group_in_dataset = ExpGroup
    if (os.path.isfile(outputDir + AnimalID + '_dataframe.pkl')):
        AnimalDF = pd.read_pickle(outputDir + AnimalID + '_dataframe.pkl')
        try:
            exp_group_in_dataset = AnimalDF.ExperimentalGroup[0].values[0]
        except:
            exp_group_in_dataset = AnimalDF.ExperimentalGroup[0]
        update = True
    # Check the files in the raw folder
    filelist = glob.glob(InputDirectory + AnimalID + BpodProtocol + 'Session Data/*.mat')
    # Check which files are not in the dataframe and separate them into old and new ones
    if update:
        existing_dates = pd.to_datetime(pd.unique(AnimalDF['FullSessionTime']))
        old_files, files_for_updating = cuf.split_files_into_old_and_new(filelist, existing_dates)
        # if no more files for updating, break loop
        if len(files_for_updating) < 1:
            print('No new files for this animal')
            continue
    else:
        files_for_updating = filelist

    # Print file titles that are older than the newest one in AnimalDF and print as previously discarded
    if update:
        if ExpGroup != exp_group_in_dataset:
            print('The experimental control that you have written for this mouse does not correspond to what I have found:')
            print(exp_group_in_dataset)
            print('I am overridding what you have written')
            print('')
            ExpGroup = exp_group_in_dataset
        print('Previously discarded (probably) files:')
        for i in old_files:
            print(ntpath.basename(i))
        print('If you want to incorporate any of these, delete the .pkl dataframe and start over')
    else:
        print('No previous file located, considering all new data')
    # New data
    print('')
    print('New data:')
    # Read new data that is not in the previously analized dataset
    ExperimentFiles, ExperimentData, ntrialsDistribution, Protocols, Stimulations, Muscimol =\
        cuf.ReadAnimalData(files_for_updating, printout=True)

    ##
    if update:
        # get info for the number of trials per session in the dataframe
        ntrials_df = AnimalDF.groupby('FullSessionTime').size()
        # assumes the start training day is the oldest of the dataframe
        # transform the datetime format into BpodDates
        bpod_times = [i.strftime('%Y%m%d_%H%M%S') for i in ntrials_df.index]
        # get the time differences
        df_time_differences = cuf.timeDifferences(bpod_times)

    # get the time differences of the old ones (insert the 0 and then remove it)
    if update:
        old_times = cuf.ParseForTimes([ntpath.basename(i) for i in old_files])
        old_times.insert(0, bpod_times[0])
        old_time_differences = cuf.timeDifferences(old_times)[1:len(old_times)]

    # get the time differences of the new ones (insert the 0 and then remove it)
    ExperimentDates_Raw = cuf.ParseForTimes(ExperimentFiles)
    if update:
        ExperimentDates_Raw.insert(0, bpod_times[0])
        TimeDifferences = cuf.timeDifferences(ExperimentDates_Raw)[1:len(ExperimentDates_Raw)]
        ExperimentDates_Raw = ExperimentDates_Raw[1:len(ExperimentDates_Raw)]
    else:
        TimeDifferences = cuf.timeDifferences(ExperimentDates_Raw)

    ##
    # Clean new data by number of trials
    # Remove those experiments for which a proper time has not been found (old ones that are missing a lot of variables)
    # Or those with low number of trials
    minNoOfTr = 30
    idxToRemove = cuf.identifyIdx(ExperimentDates_Raw, ntrialsDistribution, minNoOfTr)

    for idx in idxToRemove:
        print('deleting data for {} with {} trials'.format(ntpath.basename(ExperimentFiles[idx]), ntrialsDistribution[idx]))
        ExperimentData.pop(idx)
        ExperimentFiles.pop(idx)
        ExperimentDates_Raw.pop(idx)
        ntrialsDistribution.pop(idx)
        Protocols.pop(idx)
        Stimulations.pop(idx)
        Muscimol.pop(idx)

    ##
    # get all data into the dataframe
    # Transform experiment dates to datetime
    ExperimentDates = cuf.BpodDatesToTime(ExperimentDates_Raw)
    DataFrames = [cuf.SessionDataToDataFrame(AnimalID, ExpGroup, ExperimentDates[i], exp['SessionData'])
                  for i, exp in enumerate(ExperimentData)]

    if len(DataFrames) > 0:
        AnimalDF_new = pd.concat(DataFrames, ignore_index=True)
        if update:
            AnimalDF = pd.concat([AnimalDF, AnimalDF_new], ignore_index=True)
        else:
            AnimalDF = AnimalDF_new

    # convert some NaNs to 0s (old data not having some fields)
    if 'RewardChange' not in AnimalDF:
        AnimalDF['RewardChange'] = 0
        AnimalDF['RewardChangeBlock'] = 0
    AnimalDF.RewardChange.fillna(0, inplace=True)
    AnimalDF.RewardChangeBlock.fillna(0, inplace=True)

    ##
    # save the dataframe
    AnimalDF.to_pickle(outputDir + AnimalID + '_dataframe.pkl')
    # append dataframe to batch dataframe
    DataFrames.append(AnimalDF)

    # plot
    fig, ax = plt.subplots(figsize=(15,5))
    ax.axhline(50, ls='--', alpha=0.4, color='k')
    ax.axhline(100, ls='--', alpha=0.4, color='k')
    sns.lineplot(x = AnimalDF.index, y = 'CumulativePerformance', data=AnimalDF, hue = 'Protocol',
                marker=".", alpha = 0.05, markeredgewidth=0, linewidth=0)
        
    lgd = plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
    for l in lgd.get_lines():
        l.set_alpha(1)
        l.set_linewidth(4)

    plt.savefig(outputDir + AnimalID + '_CumulativePerformance.pdf',
                transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight')
    


##
print('Analyzing all animals')
all_animals_df_name = batch_output + batch_name + '_dataframe.pkl'

# join dataframes
if len(DataFrames) > 0 or not os.path.isfile(all_animals_df_name):
    # Read the dataframes and merge them
    DataFrames = []
    for AID in animals_to_analyze:
        DFfile = GeneralDirectory + AID + BpodProtocol + AID + '_dataframe.pkl'
        ADF = pd.read_pickle(DFfile)
        DataFrames.append(ADF)
    AnimalsDF = pd.concat(DataFrames, ignore_index=True)

    # Create a unique ID for every session
    def mergeStrings(df):
        return df['AnimalID'] + ' ' + df['SessionTime']


    AnimalsDF['SessionID'] = AnimalsDF[['AnimalID', 'SessionTime']].apply(mergeStrings, axis=1)

    # Create a cumulative trial number for every animal BE AWARE THAT SESSIONS MIGHT HAVE BEEN REMOVED BEFORE SO THIS NUMBER IS NOT EXACT
    CumTrialsList = []
    for Aid in pd.unique(AnimalsDF['AnimalID']):
        CumTrialsList.append(np.arange(len(AnimalsDF[AnimalsDF['AnimalID'] == Aid])) + 1)
    # flatten the list of lists
    AnimalsDF['CumulativeTrialNumber'] = np.array(list(chain(*[x for x in CumTrialsList])))

    # Restart the count of CumulativeTrialNumber for every protocol
    AnimalsDF['CumulativeTrialNumberByProtocol'] = np.nan

    for Aid in pd.unique(AnimalsDF['AnimalID']):
        for Prot in pd.unique(AnimalsDF['Protocol']):
            conditions = np.logical_and(AnimalsDF['AnimalID'] == Aid, AnimalsDF['Protocol'] == Prot)
            AnimalsDF.CumulativeTrialNumberByProtocol.loc[AnimalsDF[conditions].index] = \
                np.arange(len(AnimalsDF[conditions])) + 1

    # Calculate performance of the past X trials
    PAST_WINDOW = 20
    CumPerList = []
    for Sid in pd.unique(AnimalsDF['SessionID']):
        CumPerList.append(cuf.perf_window_calculator(AnimalsDF[AnimalsDF['SessionID'] == Sid], PAST_WINDOW))
    # flatten the list of lists
    AnimalsDF['CurrentPastPerformance20'] = np.array(list(chain(*[x for x in CumPerList])))

    # Calculate performance of the past X trials
    PAST_WINDOW = 100
    CumPerList = []
    for Sid in pd.unique(AnimalsDF['SessionID']):
        CumPerList.append(cuf.perf_window_calculator(AnimalsDF[AnimalsDF['SessionID']==Sid], PAST_WINDOW))
    # flatten the list of lists
    AnimalsDF['CurrentPastPerformance100'] = np.array(list(chain(*[x for x in CumPerList])))

    # Number of pokes in the center
    AnimalsDF['NoOfCenterPokes'] = AnimalsDF[['TrialEvents', 'TrialStates']].apply(cuf.CalculateMidPokes, axis=1)

    # Time waiting in the middle
    AnimalsDF['MiddleWaitTime'] = AnimalsDF[['TrialEvents', 'TrialStates']].apply(cuf.MidPortWait, axis=1)

    # Time they take to innitiate the trial
    AnimalsDF['TrialInitiationTime'] = AnimalsDF[['TrialEvents']].apply(cuf.CalculateTrialInitiationTime, axis=1)

    # Calculate the right bias
    AnimalsDF['RightBias'] = cuf.CalculateRBiasWindow(np.array(AnimalsDF['FirstPoke']),\
                                                                np.array(AnimalsDF['FirstPokeCorrect']), 50)

    # Calculate speed over the surrounding 6 trials
    SP_WINDOW = 6
    speed_list = []
    for Sid in pd.unique(AnimalsDF['SessionID']):
        speed_list.append(cuf.speed_window_calculator(AnimalsDF[AnimalsDF['SessionID']==Sid], SP_WINDOW))
    # flatten the list of lists
    AnimalsDF['TrialsSpeed'] = np.array(list(chain(*[x for x in speed_list])))

    # calculate if the previous trial was a success or not
    AnimalsDF['PrevTrialSuccess'] = np.insert(np.array(AnimalsDF['FirstPokeCorrect'][:-1]), 0, np.nan)

    # Save the dataframe
    AnimalsDF.to_pickle(all_animals_df_name)

else:
    print('No dataframes changed, reading from file, probably repeating plots so this might be useless')
    AnimalsDF = pd.read_pickle(all_animals_df_name)

# rename the Experimental Group
AnimalsDF.ExperimentalGroup = AnimalsDF.ExperimentalGroup.astype('str')
for animal_id, experimental_group in zip(animals_to_analyze, eg_list):
    AnimalsDF.at[AnimalsDF.AnimalID == animal_id, 'ExperimentalGroup'] = experimental_group

# Save the dataframe
AnimalsDF.to_pickle(all_animals_df_name)

# plot
column_to_plot = 'CumulativePerformance'

fig, axs = plt.subplots(len(pd.unique(AnimalsDF['Protocol'])), 1, figsize=(17, 7 * len(pd.unique(AnimalsDF['Protocol']))), sharex=True)

axs = axs.ravel()

fig.subplots_adjust(hspace=0.3)
for ax in axs:
    ax.axhline(50, ls='--', alpha=0.4, color='k')
    ax.axhline(100, ls='--', alpha=0.4, color='k')

for ax, prot in enumerate(pd.unique(AnimalsDF['Protocol'])):
    sns.lineplot(x='CumulativeTrialNumberByProtocol', y=column_to_plot,
                 data=AnimalsDF[AnimalsDF['Protocol'] == prot],
                 ax=axs[ax], hue='ExperimentalGroup',
                 marker=".", alpha=0.05, markeredgewidth=0, linewidth=0)

    axs[ax].set_title(prot)
    axs[ax].set_ylim(bottom=40)

for ax in axs:
    ax.xaxis.set_tick_params(which='both', labelbottom=True)

plt.savefig(batch_output + column_to_plot + 'ByProtocol_Grouped_AnimalSelection.pdf',
            transparent=True, bbox_inches='tight')


# plot
column_to_plot = 'CumulativePerformance'

fig, axs = plt.subplots(len(pd.unique(AnimalsDF['Protocol'])), 1, figsize=(17, 7 * len(pd.unique(AnimalsDF['Protocol']))), sharex=True)
axs = axs.ravel()

fig.subplots_adjust(hspace=0.3)
for ax in axs:
    ax.axhline(50, ls='--', alpha=0.4, color='k')
    ax.axhline(100, ls='--', alpha=0.4, color='k')

for ax, prot in enumerate(pd.unique(AnimalsDF['Protocol'])):
    sns.lineplot(x='CumulativeTrialNumberByProtocol', y=column_to_plot,
                 data=AnimalsDF[AnimalsDF['Protocol'] == prot],
                 ax=axs[ax], hue='ExperimentalGroup',
                 marker=".", alpha=0.05, markeredgewidth=0, linewidth=0,
                 # the following line splits the data and does not compute confidence intervals and mean
                 units="AnimalID", estimator=None)

    axs[ax].set_title(prot)
    axs[ax].set_ylim(bottom=40)

for ax in axs:
    ax.xaxis.set_tick_params(which='both', labelbottom=True)

plt.savefig(batch_output + column_to_plot + 'ByProtocol_AnimalSelection.pdf',
            transparent=True, bbox_inches='tight')



fig, axs = plt.subplots(len(pd.unique(AnimalsDF['AnimalID'])), 1, figsize=(17, 7 * len(pd.unique(AnimalsDF['AnimalID']))), sharex=True)
axs = axs.ravel()
fig.subplots_adjust(hspace=0.3)
for ax in axs:
    ax.axhline(50, ls='--', alpha=0.4, color='k')
    ax.axhline(100, ls='--', alpha=0.4, color='k')

for ax, prot in enumerate(pd.unique(AnimalsDF['AnimalID'])):
    sns.lineplot(x='CumulativeTrialNumber', y=column_to_plot,
                 data=AnimalsDF[AnimalsDF['AnimalID'] == prot],
                 ax=axs[ax], hue='Protocol',
                 marker=".", alpha=0.05, markeredgewidth=0, linewidth=0)

    axs[ax].set_title(prot)
    axs[ax].set_ylim(bottom=40)

for ax in axs:
    ax.xaxis.set_tick_params(which='both', labelbottom=True)

plt.savefig(batch_output + column_to_plot + 'Individual_animals.pdf',
            transparent=True, bbox_inches='tight')


print(AnimalsDF.groupby(['AnimalID', 'ExperimentalGroup', 'Protocol']).max()['CumulativeTrialNumberByProtocol'])