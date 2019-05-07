# custom_functions.py

import numpy as np
import re
import ntpath
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# define a function that returns only those indices of a binary! vector (0 or 1) where some values are
# first different than 0


def first_diff_zero(array):
    # create a new vector that is the same but shifted
    # move everything one space forward
    newarray = np.concatenate((0, array), axis=None)[0:len(array)]
    difarray = array - newarray
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    # find which indexes are 1
    indexes = get_indexes(1, difarray)
    return indexes


def time_to_zero(input_list):
    return list(np.array(input_list) - input_list[0])


def ParseForTimes(files):
    # looks for 8digits followed by underscore and 6digits (bpod style)
    dates = []
    for title in files:
        try:
            match = re.search(r'\d{8}_\d{6}', ntpath.basename(title))
            dates.append(match.group())
        except:
            dates.append('notFound')
    return dates


def PsychPerformance(trialsDif, sideSelected):    
    # function to calculate psychometric performance and fit logistic regression to the data
    # returns a dictionary

    if trialsDif.any(): # in case an empty thing is passed
    
        # masks to remove nans for logistic regression
        nan_mask = ~(np.isnan(trialsDif) | np.isnan(sideSelected))
        #logistic regression
        try:
            clf = LogisticRegression().fit(trialsDif[nan_mask, np.newaxis], sideSelected[nan_mask])
        except:
            # in case a model cannot be fitted (e.g. mouse always goes to the left)
            # fit model on dummy data
            clf = LogisticRegression().fit(np.array([0,0,100,100]).reshape(-1, 1), np.array([1,0,1,0]))
        # Calculate performance
        # Initialize values
        difficulty = np.unique(trialsDif[~np.isnan(trialsDif)])
        performance = np.full(len(difficulty), np.nan)
        for i in range(len(difficulty)):
            if np.nansum(sideSelected[trialsDif==difficulty[i]])>0:
                performance[i] = 100 * (np.nanmean(sideSelected[trialsDif==difficulty[i]]) - 1)
            else:
                performance[i] = np.nan
    
    
        DictToReturn = {
                'Difficulty': difficulty,
                'Performance': performance,
                'Logit': clf
                }
    else:
        DictToReturn = {}
        
    return DictToReturn


def splitOpto(SessionData):
    # SessionData comes from bpod: ExperimentData[x]['SessionData']
    # Returns two dictionaries
    
    Trials_normalMask = SessionData['OptoStim'] == 0
    Trials_optoMask = SessionData['OptoStim'] == 1
    
    # selection of normal and opto trials
    normalTrials_sideSelected = SessionData['FirstPoke'][Trials_normalMask]
    normalTrials_difficulty = SessionData['TrialHighPerc'][Trials_normalMask]
    optoTrials_sideSelected = SessionData['FirstPoke'][Trials_optoMask]
    optolTrials_difficulty = SessionData['TrialHighPerc'][Trials_optoMask]
    
    # create dictionaries
    NormalTrials = {
        'SideSelected': normalTrials_sideSelected,
        'Difficulty': normalTrials_difficulty   
    }
    
    OptoTrials = {
        'SideSelected': optoTrials_sideSelected,
        'Difficulty': optolTrials_difficulty
    }
    
    return NormalTrials, OptoTrials


def generate_fake_data(trialsDif, sideSel):
    # Generates data for bootstrapping, sampling and replacing, so each
    # unique trialsDif maintains the same size
    
    fake_side_sel = np.empty_like(sideSel)
    for curr_diff in np.unique(trialsDif):
        diff_mask = trialsDif == curr_diff
        population = sideSel[diff_mask]
        fake_side_sel[diff_mask] = np.random.choice(population, len(population))

    return fake_side_sel


def BootstrapPerformances(trialsDif, sideSelected, ntimes, prediction_difficulties):
    # Bootstrap data and return logistic regression predictions for each sampled model
    # remove nans
    nan_mask = ~(np.isnan(sideSelected) | np.isnan(trialsDif))
    difficulties = trialsDif[nan_mask]
    sideselection = sideSelected[nan_mask]

    predictPerFake = np.empty((len(prediction_difficulties), ntimes))
    for i in range(predictPerFake.shape[1]):
        # create fake data
        fake_data = generate_fake_data(difficulties, sideselection)
        try:
            clf_fake = LogisticRegression().fit(difficulties.reshape(-1, 1), fake_data)
        except:
            # in case a model cannot be fitted (e.g. mouse always goes to the left)
            # fit model on dummy data
            clf_fake = LogisticRegression().fit(np.array([0,0,100,100]).reshape(-1, 1), np.array([1,0,1,0]))
            
        predictPerFake[:, i] = 100 * clf_fake.predict_proba(prediction_difficulties)[:, 1]
        
    return predictPerFake


def SessionDataToDataFrame (AnimalID, SessionID, SessionData):
    # function to create a dataframe out of the session
    # each trial is an entry on the dataframe

    # if the session is empty output a message
    if not 'nTrials' in SessionData:
        print('Session is empty')
        return pd.DataFrame()
    
    numberOfTrials = SessionData['nTrials']
    
    # protocol information 
    ts = SessionData['TrialSettings']
    protocols = [ts[0]['GUIMeta']['TrainingLevel']['String'][x] for x in [y['GUI']['TrainingLevel'] - 1 for y in ts]]
    stimulations = [ts[0]['GUIMeta']['OptoStim']['String'][x] for x in [y['GUI']['OptoStim'] - 1 for y in ts]]    
    if not np.logical_and(len(protocols) == numberOfTrials, len(stimulations) == numberOfTrials):
        print('protocols and/or stimulations length do not match with the number of trials')
        return pd.DataFrame()
    
    # trial events
    trev = [x['Events'] for x in SessionData['RawEvents']['Trial']]
    if not len(trev)==numberOfTrials:
        print('trial events length do not match with the number of trials')
        return pd.DataFrame()
    
    # trial states
    trst = [x['States'] for x in SessionData['RawEvents']['Trial']]
    if not len(trst)==numberOfTrials:
        print('trial states length do not match with the number of trials')
        return pd.DataFrame()
    
    # calculate the cumulative performance
    firstpokecorrect = SessionData['FirstPokeCorrect'][0:numberOfTrials]
    correct_cp = np.cumsum(firstpokecorrect == 1)
    incorrect_cp = np.cumsum(firstpokecorrect == 0)
    cumper = 100 * correct_cp / (correct_cp + incorrect_cp)
    
    DFtoReturn = pd.DataFrame({'AnimalID': np.repeat(AnimalID, numberOfTrials),
                               'SessionTime': np.repeat(SessionID, numberOfTrials),
                               'Protocol': protocols,
                               'Stimulation': stimulations,
                               'TrialIndex': list(range(numberOfTrials)),
                               'TrialHighPerc': SessionData['TrialHighPerc'][0:numberOfTrials],
                               'Outcomes': SessionData['Outcomes'][0:numberOfTrials],
                               'OptoStim': SessionData['OptoStim'][0:numberOfTrials],
                               'ChosenSide': SessionData['ChosenSide'][0:numberOfTrials],
                               'OptoStim': SessionData['OptoStim'][0:numberOfTrials],
                               'FirstPokeCorrect': firstpokecorrect,
                               'FirstPoke': SessionData['FirstPoke'][0:numberOfTrials],
                               'TrialSide': SessionData['TrialSide'][0:numberOfTrials],
                               'TrialSequence': SessionData['TrialSequence'][0:numberOfTrials],
                               'ResponseTime': SessionData['ResponseTime'][0:numberOfTrials],
                               'TrialStartTimestamp': SessionData['TrialStartTimestamp'],
                               'CumulativePerformance': cumper,
                               'TrialEvents': trev,
                               'TrialStates': trst
                              })
    
    return DFtoReturn


