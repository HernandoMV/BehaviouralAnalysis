from utils import custom_functions as cuf
from utils import plot_utils
import os
import glob
import ntpath
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import chain
import math


# Animals to analyze
# animals_to_analyze = [''.join(['A2A', str(x)]) for x in range(10, 28)]
# animals_to_analyze = [''.join(['SP', str(x)]) for x in range(19, 36)]
# animals_to_analyze = [''.join(['QPM00', str(x)]) for x in range(7, 10)] + [''.join(['QPM0', str(x)]) for x in range(10, 13)]
# animals_to_analyze = ['KAA682', 'KAA683', 'KAA684', 'KAA685', 'KAA686', 'KAA757',
#                       'C01', 'C02', 'C03', 'SomFlp04', 'SomFlp06']
# animals_to_analyze = [''.join(['A2A', str(x)]) for x in [11,12,13,16,17,18,20,22]] + \
#                      ['C01', 'C02', 'C03', 'SomFlp04', 'SomFlp06']
# animals_to_analyze = [''.join(['CL-', f"{x:02}"]) for x in range(1, 12)]
# animals_to_analyze = [''.join(['A2A', str(x)]) for x in [10,14,15,19,21,23,24,25,26,27]] + \
#                      [''.join(['CL-', f"{x:02}"]) for x in range(1, 12)]
# animals_to_analyze = [''.join(['LTD-', f"{x:02}"]) for x in range(1, 7)]
# animals_to_analyze = ['C01', 'C02', 'C03', 'SomFlp04', 'SomFlp06']
# animals_to_analyze = ['KAA682', 'KAA683', 'KAA684', 'KAA685', 'KAA686', 'KAA757']
# animals_to_analyze = ['DRD101', 'DRD102', 'DRD103']
# animals_to_analyze = ['CAA679', 'CAA680', 'CAA681', 'CAA720', 'CAA721', 'CAA722', 'KAA536', 'KAA537', 'KAA706']
# animals_to_analyze = [''.join(['KAA', str(x)]) for x in range(327, 333)]
# animals_to_analyze = [''.join(['PH', str(x)]) for x in range(301, 307)]
# animals_to_analyze = [''.join(['YMLS0', str(x)]) for x in range(1, 5)]
# animals_to_analyze = ['PH307', 'PH308', 'PH310']
# animals_to_analyze = [''.join(['CTRL0', str(x)]) for x in range(1, 6)] + [''.join(['FI4TM0', str(x)]) for x in range(1, 4)] + [''.join(['A2A0', str(x)]) for x in range(4, 8)]
# animals_to_analyze = ['pdyn01', 'pdyn02', 'pdyn03', 'pdyn04']
# animals_to_analyze = [''.join(['KAA', str(x)]) for x in range(927, 932)]
# animals_to_analyze = [''.join(['LFPo0', str(x)]) for x in range(7, 10)] + [''.join(['LFPo', str(x)]) for x in range(10, 13)]
# animals_to_analyze = [''.join(['CPH0', str(x)]) for x in range(1, 7)]
# animals_to_analyze = [''.join(['EY14-0', str(x)]) for x in range(1, 7)]
# animals_to_analyze = [''.join(['A2A', str(x)]) for x in range(28, 34)]
# animals_to_analyze = ['D1opto-01', 'D1opto-02', 'D2opto-01', 'D2opto-02']
# animals_to_analyze = ['N01', 'N02', 'N03', 'N05', 'Somcre04', 'SL_NMDA1', 'SL_NMDA2', 'SL_NMDA3']
# animals_to_analyze = [''.join(['EY14-0', str(x)]) for x in range(1, 7)] + [''.join(['A2A', str(x)]) for x in range(28, 34)]
# animals_to_analyze = [''.join(['varS', str(x)]) for x in range(1, 7)]
# animals_to_analyze = [''.join(['D1opto-0', str(x)]) for x in range(1,10)] + [''.join(['D2opto-0', str(x)]) for x in range(1,8)]
# animals_to_analyze = [''.join(['CL-', f"{x:02}"]) for x in range(13, 30)] + [''.join(['LFP', f"{x:02}"]) for x in range(16, 22)]
# animals_to_analyze = [''.join(['D1opto-', str(x)]) for x in range(10,15)] + [''.join(['D2opto-0', str(x)]) for x in range(8,10)] + [''.join(['D2opto-', str(x)]) for x in range(10,13)]
animals_to_analyze = [''.join(['DAopto-0', str(x)]) for x in range(1,10)] + ['DAopto-10']

# Name of batch
# batch_name = 'D2-caspase_Apr2021'
# batch_name = 'PF-caspase'
# batch_name = 'Controls_Quentin_Inference-2'
# batch_name = 'Chronic_lesion_Jun2021'
# batch_name = 'Chronic_lesion_and_controls_Jul2021'
# batch_name = 'LTD_Jul2021'
# batch_name = 'Controls_6OHDA'
# batch_name = 'Controls_drd1KD-Laura'
# batch_name = 'Controls_d1-and-d2-casp-POST'
# batch_name = 'Controls_CPH-Matthew'
# batch_name = 'Controls_ZIP-Yasmin'
# batch_name = 'Controls_muscimol-2'
# batch_name = 'Caspase-d1-pre_HMV-cohort-1'
# batch_name = 'Caspase-d2-pre_HMV-cohort-2'
# batch_name = 'D1andD2opto'
# batch_name = 'NMDA'
# batch_name = 'variable_intensity_test'
# batch_name = 'Chronic_lesion_and_controls_III_Dec2021'
# batch_name = 'D1andD2opto-learning_Dec21'
batch_name = 'DAoptostimulation_Mar22'

# create empty list
DataFrames = []
# Write the experimental groups
# eg_list = list(np.repeat('Sthita_cohort', 17))
# eg_list = list(np.repeat('Quentin_cohort', 10))
# eg_list = ['Cortex_Buffer', 'D2Cre-caspase', 'D2Cre-caspase', 'D2Cre-caspase',
#            'Cortex_Buffer', 'Cortex_Buffer', 'D2Cre-caspase', 'D2Cre-caspase', 'D2Cre-caspase',
#            'Cortex_Buffer', 'D2Cre-caspase', 'Cortex_Buffer', 'D2Cre-caspase',
#            'Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer']
# eg_list = list(np.repeat('drd1cre-controls', 6)) + list(np.repeat('6OHDA-controls', 5))
# eg_list = list(np.repeat('d2-caspase', 8)) + list(np.repeat('6OHDA-controls', 5))
# eg_list = list(np.repeat('NA', 12))
# eg_list = list(np.repeat('LTD', 6))
# eg_list = list(np.repeat('Lesion', 10)) + ['Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer', 'Lesion', 'Cortex_Buffer', 'Cortex_Buffer', 'Lesion',
#            'Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer']
# eg_list = ['Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer', 'Lesion', 'Cortex_Buffer', 'Cortex_Buffer', 'Lesion',
#            'Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer', 'Cortex_Buffer']
# eg_list = list(np.repeat('Cortex_Buffer', 5))
# eg_list = list(np.repeat('For_muscimol', 4))
# eg_list = ['cre-neg', 'cre-pos', 'cre-pos', 'cre-neg', 'cre-pos', 'cre-pos']
# eg_list = ['Caspase', 'Caspase', 'GFP', 'GFP', 'GFP', 'GFP']
# eg_list = list(np.repeat('optoinhibition', 16))
# eg_list = list(np.repeat('NMDA', 8))
# eg_list = ['control', 'd1-caspase', 'd1-caspase', 'control', 'd1-caspase', 'd1-caspase', 'd2-caspase', 'd2-caspase', 'control', 'control', 'control', 'control']
# eg_list = list(np.repeat('varint', 6))
# eg_list = list(np.repeat('Lesion', 12)) + list(np.repeat('Control', 11))# + list(np.repeat('Control-LFP', 6))
# eg_list = list(np.repeat('optoinhibition', 10))
eg_list = list(np.repeat('DAoptostimulation', 10))

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
    print('')
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
        if old_files:
            print('Previously discarded (probably) files:')
            for i in old_files:
                print(ntpath.basename(i))
            print('If you want to incorporate any of these, delete the .pkl dataframe and start over')
    else:
        print('No previous file located, considering all new data')
    # New data
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
    minNoOfTr = 60
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
    sns.lineplot(x = AnimalDF.index, y = 'CumulativePerformance', data=AnimalDF, hue='Protocol',
                marker=".", alpha=0.05, markeredgewidth=0, linewidth=0)

    lgd = plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
    for l in lgd.get_lines():
        l.set_alpha(1)
        l.set_linewidth(4)

    plt.savefig(outputDir + AnimalID + '_CumulativePerformance.pdf',
                transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.close(fig)


    # plot each session
    # Make a plot with the performance for all sessions
    # generate a list of the conditions, colors and labels
    CondList = [(AnimalDF['OptoStim'] == 0),
                (AnimalDF['OptoStim'] == 1)]
    ColorList = ['c', 'm']
    LabelList = ['Normal', 'Opto']

    fig, axs = plt.subplots(math.ceil(len(pd.unique(AnimalDF['SessionTime']))/4), 4,
                            figsize=(15, len(pd.unique(AnimalDF['SessionTime']))),
                            facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.2, wspace=.2)
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        if i < len((pd.unique(AnimalDF['SessionTime']))):
            ax.hlines(50, 0, 100, linestyles='dotted', alpha=0.4)
        ax.axis('off')
    # process data from all experiments
    for counter, session in enumerate(pd.unique(AnimalDF['SessionTime'])):
        ax = axs[counter]
        for i, condition in enumerate(CondList):
            predictDif, PsyPer, fakePredictions, predictPer, _ = \
            cuf.PP_ProcessExperiment(AnimalDF[(AnimalDF['SessionTime'] == session) & condition], bootstrap=5)
            if PsyPer:
                plot_utils.PlotPsychPerformance(dataDif=PsyPer['Difficulty'], dataPerf=PsyPer['Performance'],
                                                predictDif=predictDif, ax=ax, fakePred=fakePredictions,
                                                realPred=predictPer, color=ColorList[i], label=LabelList[i])

        ax.get_legend().remove()
        ax.text(.5, .95, str(counter) + ': ' + (session),
                        horizontalalignment='center', fontweight='bold', transform=ax.transAxes)

        axs[counter].text(.5,.85, AnimalDF[(AnimalDF['SessionTime'] == session)].Protocol.unique()[0],
                        horizontalalignment='center', transform=axs[counter].transAxes)
        axs[counter].text(.5,.75, AnimalDF[(AnimalDF['SessionTime'] == session)].Stimulation.unique()[0],
                        horizontalalignment='center', transform=axs[counter].transAxes)
        axs[counter].text(.5,.65, 'No of trials: ' + str(len(AnimalDF[(AnimalDF['SessionTime'] == session)])),
                        horizontalalignment='center', transform=axs[counter].transAxes)

        ax.axis('on')
        # remove some ticks
        ax.tick_params(which='both', top=False, bottom='on', left='on', right=False,
                    labelleft='on', labelbottom='on')
        if not ax.is_first_col():
            ax.set_ylabel('')
            ax.set_yticks([])
        if not ax.is_last_row():
            ax.set_xlabel('')
            ax.set_xticks([])
        plt.tight_layout()

    plt.savefig(outputDir + AnimalID + '_psychometricPerformanceAllSessions.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)

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


    # Remove trials in which the animals are disengaged
    # Create also a column that holds the ITIs
    ITIs_list = []
    disengaged_global_indexes = []

    for Sid in pd.unique(AnimalsDF['SessionID']):
        itis = cuf.itis_calculator(AnimalsDF[AnimalsDF['SessionID']==Sid])
        ITIs_list.append(itis)
        # Identify the trials where there is disengament
        dis_idx = cuf.find_disengaged_trials(itis)
        # print information of percentage removed
        print('{0}, with {1} trials, {2:%} are removed'.format(Sid, len(itis), len(dis_idx)/len(itis)))
        # Find their global index
        dgi = list(AnimalsDF[AnimalsDF['SessionID']==Sid].index[dis_idx])
        # Add them to list
        disengaged_global_indexes = disengaged_global_indexes + dgi

    # flatten the list of lists
    AnimalsDF['ITIs'] = np.array(list(chain(*[x for x in ITIs_list])))

    # remove these trials from the dataframe
    AnimalsDF = AnimalsDF.drop(disengaged_global_indexes)


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
    AnimalsDF['RightBias'] = cuf.CalculateRBiasWindow(np.array(AnimalsDF['FirstPoke']), np.array(AnimalsDF['FirstPokeCorrect']), 50)

    # Calculate speed over the surrounding 6 trials
    SP_WINDOW = 6
    speed_list = []
    for Sid in pd.unique(AnimalsDF['SessionID']):
        speed_list.append(cuf.speed_window_calculator(AnimalsDF[AnimalsDF['SessionID']==Sid], SP_WINDOW))
    # flatten the list of lists
    AnimalsDF['TrialsSpeed'] = np.array(list(chain(*[x for x in speed_list])))

    # calculate if the previous trial was a success or not
    AnimalsDF['PrevTrialSuccess'] = np.insert(np.array(AnimalsDF['FirstPokeCorrect'][:-1]), 0, np.nan)

else:
    print('No dataframes changed, reading from file, probably repeating plots so this might be useless')
    AnimalsDF = pd.read_pickle(all_animals_df_name)

# rename the Experimental Group
AnimalsDF.ExperimentalGroup = AnimalsDF.ExperimentalGroup.astype('str')
for animal_id, experimental_group in zip(animals_to_analyze, eg_list):
    AnimalsDF.at[AnimalsDF.AnimalID == animal_id, 'ExperimentalGroup'] = experimental_group

# Save the dataframe
AnimalsDF.to_pickle(all_animals_df_name)


# plot for the individual animals
column_to_plot = 'CurrentPastPerformance100'

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

    axs[ax].set_title(prot + str(pd.unique(AnimalsDF[AnimalsDF['AnimalID'] == prot].ExperimentalGroup)[0]))
    axs[ax].set_ylim(bottom=40)

for ax in axs:
    ax.xaxis.set_tick_params(which='both', labelbottom=True)

plt.savefig(batch_output + column_to_plot + 'Individual_animals.pdf',
            transparent=True, bbox_inches='tight')
plt.close(fig)

# plot also the heatmaps with training information
fig, ax = plt.subplots(len(pd.unique(AnimalsDF.AnimalID)), 1, figsize=(17,5 * len(pd.unique(AnimalsDF.AnimalID))))
axs = ax.ravel()
fig.subplots_adjust(hspace=1.3)
for i, animal in enumerate(pd.unique(AnimalsDF.AnimalID)):
    aDF = AnimalsDF[AnimalsDF.AnimalID == animal]
    dfToPlot = plot_utils.summary_matrix(aDF)
    axs[i] = plot_utils.summary_plot(dfToPlot, aDF, axs[i], top_labels=['Punish', 'CenterPortDuration', 'BiasCorrection'])

plt.savefig(batch_output + column_to_plot + 'psychometricPerformanceAllSessionsHeatmap.pdf',
            transparent=True,dpi = 500, bbox_inches='tight')
plt.close(fig)

# select only auditory and limit to 5000 trials
protocols_selected = ['Auditory']
AnimalsDF = AnimalsDF[AnimalsDF.Protocol.isin(protocols_selected)]
trials_up_to = 5000
AnimalsDF = AnimalsDF[AnimalsDF['CumulativeTrialNumberByProtocol'] < trials_up_to]



# create a publication-quality figure, grouping the data by the Experimental Group, and showing the continous monitoring
# of learning
col_to_plot = 'CurrentPastPerformance100'
df_sel = AnimalsDF
data_mean = df_sel.groupby(['CumulativeTrialNumberByProtocol','ExperimentalGroup'])[col_to_plot].mean().reset_index()
st_err_mean = df_sel.groupby(['CumulativeTrialNumberByProtocol','ExperimentalGroup'])[col_to_plot].std().reset_index()
data_mean['low_bound'] = data_mean[col_to_plot] - st_err_mean[col_to_plot]
data_mean['high_bound'] = data_mean[col_to_plot] + st_err_mean[col_to_plot]

fig1 = plt.figure(figsize=(8, 4))
plt.axhline(50, ls='dotted', alpha=0.4, color='k')
plt.axhline(75, ls='dotted', alpha=0.4, color='k')
plt.axhline(100, ls='dotted', alpha=0.4, color='k')
plt.axvline(2000, ls='dotted', alpha=0.4, color='k')
plt.axvline(3500, ls='dotted', alpha=0.4, color='k')
for i,eg in enumerate(pd.unique(df_sel.ExperimentalGroup)):
    df = data_mean[data_mean.ExperimentalGroup == eg].copy()
    x = df.CumulativeTrialNumberByProtocol
    plt.plot(x, df[col_to_plot], label=eg)
    #plt.plot(data_mean[data_mean.ExperimentalGroup==eg][col_to_plot], linestyle='--', color=LSpalette[i], label='95% ci')
    #plt.plot(neg_ci, linestyle='--', color='k')
    y1 = df['low_bound']
    y2 = df['high_bound']
    plt.fill_between(x, y1, y2, where=y2 >= y1, alpha=.2, interpolate=False)

plt.ylabel(col_to_plot)
plt.xlabel('trial number')
plt.ylabel('task performance (%)')
plt.legend(loc=(0.76,0.3), frameon=False)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# remove the legend as the figure has it's own
# ax.get_legend().remove()

ax.set_xlim((0, 5000))
ax.set_ylim((45,105))

plt.title(batch_name + ' Task learning progression')

plt.savefig(batch_output + column_to_plot + '_Performance_between_groups.pdf', transparent=True, bbox_inches='tight')

plt.close(fig1)


# plot
fig, axs = plt.subplots(len(pd.unique(AnimalsDF['Protocol'])), 1, figsize=(17, 7 * len(pd.unique(AnimalsDF['Protocol']))), sharex=True)

if axs.numRows > 1:
    axs = axs.ravel()
else:
    axs = [axs,]

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
plt.close(fig)

# plot

fig, axs = plt.subplots(len(pd.unique(AnimalsDF['Protocol'])), 1, figsize=(17, 7 * len(pd.unique(AnimalsDF['Protocol']))), sharex=True)
if axs.numRows > 1:
    axs = axs.ravel()
else:
    axs = [axs,]

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
plt.close(fig)


# print number of trials per protocol
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(AnimalsDF.groupby(['AnimalID', 'ExperimentalGroup', 'Protocol']).max()['CumulativeTrialNumberByProtocol'])
