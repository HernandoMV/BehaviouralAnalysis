"""
Dopamine_optostimulation_progressive.py

Test whether optostimulation of dopamine axons produces a progressive
effect on the bias
"""

#%% Import libraries
%load_ext autoreload
%autoreload 2
from utils import plot_utils
from utils import custom_functions as cuf

import os
import sys

import glob
import ntpath
import matplotlib
%matplotlib inline
import matplotlib.pylab as plt
import numpy as np
import math
import seaborn as sns
import pandas as pd
import warnings
from itertools import chain
import random
from datetime import datetime
from scipy import stats
from ast import literal_eval


#%% Get data
data_path = "/mnt/c/Users/herny/Documents/GitHub/APE_paper/data/DA-optostimulation_dataframe.csv"
dao_df = pd.read_csv(data_path, index_col=0)

#%%
# reconvert column to a diccionary
dao_df['FullGUI'] = [literal_eval(x) for x in dao_df.FullGUI]
#%% See columns
dao_df.columns

#%%
# get the number of trials per session
mti = dao_df.groupby(['AnimalID', 'SessionTime'])['TrialIndex'].max()
mti
#%%
# plot them
plt.scatter(np.repeat(1, len(mti)), mti, alpha=0.2)

#%% Select only one session to code the script
example_session = 'DAopto-07 Apr07 15:06'
session_df = dao_df[dao_df['SessionID'] == example_session]
session_df.TrialIndex.max()


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

#%% Get the bias as the session progresses
# Define how to group the trials

# trial w/o stimulation
ini_trials = 150
# size of the window
trials_window_size = 150
# sampling step
sampling_step = 50
# maximum number of trials
max_trials = session_df.TrialIndex.max()

# %% 
# generate the groups
trial_groups = generate_trial_groups(ini_trials, max_trials, sampling_step, trials_window_size)

# Calculate the bias for each group
rbl, mtil = calculate_running_bias(session_df, trial_groups)
# %%
# plot the bias
plt.plot(mtil, rbl, 'o-')

# %% Do this for all the sessions

# %%
# Generate a dataframe that holds this information with some more
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
opto_df = pd.DataFrame(data, columns=cols)

# %%
# Redo plot above for the example
plt.plot(opto_df[opto_df["SessionID"] == example_session].ProgressionBlockIndex,
         opto_df[opto_df["SessionID"] == example_session].BiasToStimMovement,
         'o-')
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
conditions = np.logical_and(conditions, opto_df.Contralateral==True)
# select those sessions in which the stimulation happens when the animal is in the cent port
# this tests for state-action associations
conditions = np.logical_and(conditions, opto_df.StimPort=='Center')

# select only the tail
conditions = np.logical_and(conditions, opto_df.FiberArea=='tStr')

opto_df_sel = opto_df[conditions].copy()

# %%
# remove tStr DAopto-05 L as the fiber is a bit anterior
idx_to_remove = opto_df_sel[np.logical_and(np.logical_and(opto_df_sel.AnimalID=='DAopto-05',
                                                          opto_df_sel.FiberArea=='tStr'),
                                           opto_df_sel.FiberSide=='Left')].index

opto_df_sel.drop(idx_to_remove, inplace=True)

# %%
# plot all sessions
sns.lineplot(data=opto_df_sel,
             x="ProgressionBlockIndex",
             y="BiasToStimMovement",
             hue="AnimalID")
# %%
# TODO: Identify repeated sessions and calculate mean
# make a function to calculate the mean of the items in opto_df_sel that have the same AnimalID,
# FiberSide, FiberArea, StimSide, StimPort and Protocol and store it in a new dataframe 
# called opto_df_mean that has the same columns as opto_df_sel


# %%
# Identify repeated sessions and calculate mean
