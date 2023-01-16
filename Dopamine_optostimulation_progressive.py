"""
Dopamine_optostimulation_progressive.py

Test whether optostimulation of dopamine axons produces a progressive
effect on the bias
"""

#%% Import libraries
%load_ext autoreload
%autoreload 2
from utils import custom_functions as cuf
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from ast import literal_eval

# %%
# Define how to group the trials
# trials w/o stimulation
ini_trials = 150
# size of the window
trials_window_size = 150
# sampling step
sampling_step = 25

# %% Get data
# data_path = "/mnt/c/Users/herny/Documents/GitHub/APE_paper/data/DA-optostimulation_dataframe.csv"
data_path = "/home/hernandom/GitHub/APE_paper/data/DA-optostimulation_dataframe.csv"
dao_df = pd.read_csv(data_path, index_col=0)

#%%
# reconvert column to a diccionary
dao_df['FullGUI'] = [literal_eval(x) for x in dao_df.FullGUI]

# #%% See columns
# # dao_df.columns

# #%%
# # get the number of trials per session
# mti = dao_df.groupby(['AnimalID', 'SessionTime'])['TrialIndex'].max()
# mti
# #%%
# # plot them
# plt.scatter(np.repeat(1, len(mti)), mti, alpha=0.2)

# #%% Select only one session to code the script
# example_session = 'DAopto-07 Apr07 15:06'
# session_df = dao_df[dao_df['SessionID'] == example_session]
# session_df.TrialIndex.max()


#%%
# Generate a function to group numbers through a running window
def generate_trial_groups(ini_trials,
                          max_trials,
                          sampling_step,
                          trials_window_size):
    trial_groups = []
    # add as the first group the unstimulated trials
    trial_groups.append(np.arange(1, (ini_trials + 1)))
    tws_half = trials_window_size // 2
    for i in range(ini_trials, max_trials, sampling_step):
        if (i + tws_half) > max_trials:
            break
        # create a sequence of numbers
        tr_sequence = np.arange(i - tws_half, i + tws_half)
        trial_groups.append(tr_sequence)
        
    return trial_groups

# %% Create function to calculate the bias for each group

def calculate_running_bias(session_df, trial_groups):
    # create a list to store the bias
    running_bias_list = []
    # create a list to store the mean trial index
    mean_ti_list = []
    # select the non stimulated df
    non_stim_df = session_df[session_df['TrialIndex'].isin(trial_groups[0])]
    
    # calculate the bias
    for _, trial_group in enumerate(trial_groups):
        # select the df
        df = session_df[session_df['TrialIndex'].isin(trial_group)]
        # calculate the bias
        bias_to_right = cuf.get_general_right_bias(df, non_stim_df)
        # append the bias
        running_bias_list.append(bias_to_right)
        # calculate the mean trial index
        mean_ti = np.mean(trial_group)
        # append the mean trial index
        mean_ti_list.append(mean_ti)
        
    return running_bias_list, mean_ti_list

# #%% Get the bias as the session progresses

# # maximum number of trials
# max_trials = session_df.TrialIndex.max()

# # %% 
# # generate the groups
# trial_groups = generate_trial_groups(ini_trials, max_trials,
#                                      sampling_step, trials_window_size)

# # Calculate the bias for each group
# rbl, mtil = calculate_running_bias(session_df, trial_groups)
# # %%
# # plot the bias
# plt.plot(mtil, rbl, 'o-')

# %% Do this for all the sessions


# %%
# Generate a dataframe that holds this information with some more
def generate_opto_dataset(dao_df):

    BRS = ["tStr", "NAc"]
    PS = ["Left", "Right"]
    PI = ["Center", "Side"]

    data = []

    cols = [
            "AnimalID",
            "SessionID",
            "Ntrials",
            "Protocol",
            "Stim",
            "FiberSide",
            "FiberArea",
            "StimSide",
            "StimPort",
            "Contralateral",
            "InitialBias",
            "ProgressionBlockIndex",
            "BiasToStimMovement",
        ]

    for session in dao_df.SessionID.unique():
        session_df = dao_df[dao_df['SessionID'] == session].copy()
        # get animal name
        animalid = session_df.AnimalID.unique()[0]
        # get number of trials
        ntrials = session_df.shape[0]
        # protocol
        protocol = session_df.Protocol.unique()[0]
        # is it a stimulated session?
        stim = session_df.Stimulation.unique()[0] != "NoStimulation"
        # which fiber was plugged in
        fiberside = session_df.Stimulation.unique()[0]
        # which brain area is this fiber over
        fiberarea = BRS[int(session_df.iloc[0].FullGUI["FiberLocation"]) - 1]
        # which one of the side ports, or trial type, was stimulated
        stimside = PS[int(session_df.iloc[0].FullGUI["JOPSide"]) - 1]
        # in which one of the ports did stimulation occurred
        stimport = PI[int(session_df.iloc[0].FullGUI["OptoState"]) - 1]
        # is the fiber contralateral to the port
        contralateral = True
        if (fiberside == stimside) or fiberside == "Both":
            contralateral = False
        
        # generate the groups
        trial_groups = generate_trial_groups(ini_trials, ntrials, sampling_step, trials_window_size)
        # what is the initial bias of the mouse in trials before stimulation
        ini_sess = session_df[session_df.TrialIndex < ini_trials].copy()
        initialbias = np.mean(
            cuf.get_choices(ini_sess["FirstPoke"], ini_sess["TrialHighPerc"])[1]
        )
        
        # Calculate the bias for each group
        rbl, mtil = calculate_running_bias(session_df, trial_groups)
        # is this bias positive towards the stimulated movement?
        if stimside == "Right":
            rbl = rbl
        if stimside == "Left":
            rbl = [-x for x in rbl]
        # fill the data
        for i, (bias, ti) in enumerate(zip(rbl, mtil)):
            data.append(
                [
                    animalid,
                    session,
                    ntrials,
                    protocol,
                    stim,
                    fiberside,
                    fiberarea,
                    stimside,
                    stimport,
                    contralateral,
                    initialbias,
                    ti,
                    bias,
                ]
            )

    # create dataframe
    return pd.DataFrame(data, columns=cols)

# %%
opto_df = generate_opto_dataset(dao_df=dao_df)

# %% remove fibers
# remove tStr DAopto-05 L as the fiber is a bit anterior
idx_to_remove = opto_df[np.logical_and(np.logical_and(opto_df.AnimalID == 'DAopto-05',
                                                      opto_df.FiberArea == 'tStr'),
                                       opto_df.FiberSide == 'Left')].index

opto_df.drop(idx_to_remove, inplace=True)

# %%
# Redo plot above for the example
# plt.plot(opto_df[opto_df["SessionID"] == example_session].ProgressionBlockIndex,
#          opto_df[opto_df["SessionID"] == example_session].BiasToStimMovement,
#          'o-')
# %%
# Select some conditions only
#### conditions 
# remove sessions in which initially the mouse is very bias
extreme_ini_bias = 100/3
conditions = opto_df.Protocol.isin(['Aud_Psycho', 'Auditory'])
conditions = np.logical_and(conditions, extreme_ini_bias < opto_df.InitialBias)
conditions = np.logical_and(conditions, (100 - extreme_ini_bias) > opto_df.InitialBias)
# select those sessions for which the stimulation happens in contralateral trials
# this emulates the physiological conditions
conditions = np.logical_and(conditions, opto_df.Contralateral == True)
# select those sessions in which the stimulation happens when the animal is in the cent port
# this tests for state-action associations
conditions = np.logical_and(conditions, opto_df.StimPort == 'Center')

# select only the tail
# conditions = np.logical_and(conditions, opto_df.FiberArea=='tStr')

opto_df_sel = opto_df[conditions].copy()


# %%
# # plot all sessions
# sns.lineplot(data=opto_df_sel,
#              x="ProgressionBlockIndex",
#              y="BiasToStimMovement",
#              hue="FiberArea")
# %%
# TODO: Identify repeated sessions and calculate mean
# make a function to calculate the mean of the items in opto_df_sel that have the same AnimalID,
# FiberSide, FiberArea, StimSide, StimPort and Protocol and store it in a new dataframe 
# called opto_df_mean that has the same columns as opto_df_sel


# %%
# Identify repeated sessions and calculate mean


def find_indexes_of_repeated_cases(opto_df_sel, same_columns):
    # Find indexes of repeated cases
    equal_indexes = []

    for index in opto_df_sel.index:
        data = opto_df_sel.loc[index][same_columns].values
        i_list = []
        for i in opto_df_sel.index:
            if np.array_equal(data, opto_df_sel.loc[i][same_columns].values):
                i_list.append(i)
        if len(i_list) > 1:
            if i_list not in equal_indexes:
                equal_indexes.append(i_list)

    return equal_indexes


def merge_repeated_cases_for_dopamine_optostimulation(opto_df_sel):

    # Find indexes of repeated cases
    same_columns = [
        "AnimalID",
        "FiberSide",
        "FiberArea",
        "StimSide",
        "StimPort",
        "Contralateral",
        "ProgressionBlockIndex"
    ]
    equal_indexes = find_indexes_of_repeated_cases(opto_df_sel, same_columns)

    # Combine those cases
    for case in equal_indexes:
        sub_df = opto_df_sel.loc[case].copy()
        # create new instance to add to the dataframe,
        # initiating it in the first index of the set
        new_element = sub_df.iloc[0].copy()
        # change relevant values
        new_element.SessionID = "merge"
        new_element.Ntrials = np.mean(sub_df.Ntrials.values)
        new_element.Protocol = "merge"
        new_element.InitialBias = np.nan
        new_element.BiasToStimMovement = np.mean(sub_df.BiasToStimMovement.values)

        # remove old indexes
        opto_df_sel.drop(case, inplace=True)
        # add new row
        opto_df_sel = opto_df_sel.append(new_element)
    opto_df_sel.sort_index(inplace=True)

    return opto_df_sel


# %%
ods_merge = merge_repeated_cases_for_dopamine_optostimulation(opto_df_sel)


# # %%
# # plot all sessions
# sns.lineplot(data=ods_merge,
#              x="ProgressionBlockIndex",
#              y="BiasToStimMovement",
#              hue="FiberArea")

# %%
# Add a shuffled trials dataset


# 1. shuffle the trial numbers for each session in the initial dataset
shuffled_df = dao_df.copy()
for session in shuffled_df.SessionID.unique():
    shuffled_df.loc[shuffled_df.SessionID == session, "TrialIndex"] = np.random.permutation(
        shuffled_df.loc[shuffled_df.SessionID == session, "TrialIndex"].values
    )

# %%
# 2. generate the opto dataset for the shuffled trials
shuffled_opto_df = generate_opto_dataset(dao_df=shuffled_df)

# %%
# 3. remove fibers
# remove tStr DAopto-05 L as the fiber is a bit anterior
idx_to_remove = shuffled_opto_df[np.logical_and(np.logical_and(shuffled_opto_df.AnimalID == 'DAopto-05',
                                                        shuffled_opto_df.FiberArea == 'tStr'),
                                        shuffled_opto_df.FiberSide == 'Left')].index

shuffled_opto_df.drop(idx_to_remove, inplace=True)

# %%
# 4. select the same sessiosn as in the real data for the tStr
conditions = np.logical_and(shuffled_opto_df.SessionID.isin(opto_df_sel.SessionID.unique()),
                            shuffled_opto_df.FiberArea=='tStr')

shuffled_opto_df_sel = shuffled_opto_df[conditions].copy()

# %%
# 5. merge repeated sessions
shuff_ods_merge = merge_repeated_cases_for_dopamine_optostimulation(shuffled_opto_df_sel)

# %%
# 6. Merge with the real data
shuff_ods_merge.FiberArea = shuff_ods_merge.FiberArea + '_shuff'
# merge the shuffled data with the real data
all_data = pd.concat([ods_merge, shuff_ods_merge], axis=0)

# %%
# remove NAc for now
all_data = all_data[all_data.FiberArea != 'NAc']

# %%
# # plot it together with the real tStr data
# sns.lineplot(data=all_data,
#              x="ProgressionBlockIndex",
#              y="BiasToStimMovement",
#              hue="FiberArea")


# %%
# get the number of events in each block
events_per_trial_block = all_data[all_data.FiberArea == 'tStr']['ProgressionBlockIndex'].value_counts()
events_per_trial_block.sort_index(inplace=True)
# plot it
sns.lineplot(x=events_per_trial_block.index, y=events_per_trial_block.values)

# %%
# select a cutoff of minimum 3 sessions per block
etb_mask = list(events_per_trial_block[events_per_trial_block >= 3].index)
# %%
# # replot
# sns.lineplot(data=all_data[all_data.ProgressionBlockIndex.isin(etb_mask)],
#              x="ProgressionBlockIndex",
#              y="BiasToStimMovement",
#              hue="FiberArea")

# %%
# Find significance between real and shuffled data
# for each block, compare the real data to the shuffled data
# and find the pvalue of the difference

# create an empty dictionary to store the pvalues
pval_dict = {}

for block in etb_mask:
    # get the real data
    real_data = all_data[np.logical_and(all_data.FiberArea == 'tStr',
                                        all_data.ProgressionBlockIndex == block)]
    # get the shuffled data
    shuff_data = all_data[np.logical_and(all_data.FiberArea == 'tStr_shuff',
                                         all_data.ProgressionBlockIndex == block)]
    # compare the two distributions
    pval = stats.ranksums(real_data.BiasToStimMovement.values,
                          shuff_data.BiasToStimMovement.values,
                          alternative='greater').pvalue
    # store the pvalue in the dictionary
    pval_dict[block] = pval
    
# %%
# plot the significant pvalues on top of the graph
significance_mask = np.array(list(pval_dict.values())) < 0.05

biasplot = sns.lineplot(data=all_data[all_data.ProgressionBlockIndex.isin(etb_mask)],
                        x="ProgressionBlockIndex",
                        y="BiasToStimMovement",
                        hue="FiberArea",
                        legend=False)
# plot the pvalues
# get axis from plot
ax = biasplot.axes
# get the original y limit
orig_y_lim = ax.get_ylim()[1]
# plot the pvalues
for i, pvalpos in enumerate(significance_mask):
    if pvalpos:
        ax.text(x=list(pval_dict.keys())[i], y=0.9 * orig_y_lim, s='*')

data_directory = ''
plt.savefig(data_directory + 'DAstim_progressive.pdf', transparent=True, bbox_inches='tight')


# TODO: as alternative, find pval of the trend

# %%
