# custom_functions.py

import numpy as np
import re
import ntpath
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
import pandas as pd
import datetime
from itertools import chain, compress
import sys
sys.path.append("../")  # go to parent
import BehaviouralAnalysis.utils.load_nested_structs as load_ns
#import glob
import socket
import scipy.optimize as opt


def first_diff_zero(array):
    # define a function that returns only those indices of a binary! vector (0 or 1)
    # where some values are first different than 0
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
        except Exception:
            dates.append('notFound')
    return dates


def BpodDatesToTime(inputDates):
    # assumes input style YYYYMMDD_HHMMSS
    # returns a time object
    outputDates = []
    for date in inputDates:
        try:
            x = datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), int(date[9:11]), int(date[11:13]))
            outputDates.append(x)
        except Exception:
            outputDates.append('notFound')
    return(outputDates)


def PsychPerformance(trialsDif, sideSelected):
    # function to calculate psychometric performance and fit logistic regression to the data
    # returns a dictionary

    if trialsDif.any():  # in case an empty thing is passed

        # masks to remove nans for logistic regression
        nan_mask = ~(np.isnan(trialsDif) | np.isnan(sideSelected))
        # logistic regression
        if len(np.unique(sideSelected)) > 1:
            clf = LogisticRegressionCV(cv=3).fit(trialsDif[nan_mask, np.newaxis], sideSelected[nan_mask])
        else:
            # in case a model cannot be fitted (e.g. mouse always goes to the left)
            # fit model on dummy data
            clf = LogisticRegressionCV(cv=3).fit(np.array([0, 0, 0, 100, 100, 100]).reshape(-1, 1), np.array([1, 0, 1, 0, 1, 0]))
        # Calculate performance
        # Initialize values
        difficulty = np.unique(trialsDif[~np.isnan(trialsDif)])
        performance = np.full(len(difficulty), np.nan)
        for i in range(len(difficulty)):
            if np.nansum(sideSelected[trialsDif == difficulty[i]]) > 0:
                performance[i] = 100 * (np.nanmean(sideSelected[trialsDif == difficulty[i]]) - 1)
            else:
                performance[i] = np.nan

        DictToReturn = {
            'Difficulty': difficulty,
            'Performance': performance,
            'Logit': clf}
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
            clf_fake = LogisticRegressionCV(cv=3).fit(difficulties.reshape(-1, 1), fake_data)
            predictPerFake[:, i] = 100 * clf_fake.predict_proba(prediction_difficulties)[:, 1]
        except Exception:
            # in case a model cannot be fitted (e.g. mouse always goes to the left)
            # fit model on dummy data
            clf_fake = LogisticRegressionCV(cv=3).fit(np.array([0, 0, 0, 100, 100, 100]).reshape(-1, 1), np.array([1, 0, 1, 0, 1, 0]))
        

    return predictPerFake


def SessionDataToDataFrame(AnimalID, ExperimentalGroup, SessionID, SessionData):
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

    # muscimol
    yList = []
    for y in ts:
        try:
            yList.append(y['GUI']['Muscimol'] - 1)
        except Exception:
            yList.append(0)
    muscimol = []
    for x in yList:
        try:
            muscimol.append(ts[0]['GUIMeta']['Muscimol']['String'][x])
        except Exception:
            muscimol.append('No')

    # punish method
    yList = []
    for y in ts:
        try:
            yList.append(y['GUI']['Punish'] - 1)
        except Exception:
            yList.append(0)
    punish = []
    for x in yList:
        try:
            punish.append(ts[0]['GUIMeta']['Punish']['String'][x])
        except Exception:
            punish.append('No')

    # reward change
    yList = []
    reward_change_block = []
    for y in ts:
        try:
            yList.append(y['GUI']['RewardChange'] - 1)
            reward_change_block.append(y['RewardChangeBlock'])
        except Exception:
            yList.append(0)
            reward_change_block.append(0)
    reward_change = []
    for x in yList:
        try:
            reward_change.append(ts[0]['GUIMeta']['RewardChange']['String'][x])
        except Exception:
            reward_change.append('No')

    if not np.logical_and(len(protocols) == numberOfTrials, len(stimulations) == numberOfTrials):
        print('protocols and/or stimulations length do not match with the number of trials')
        return pd.DataFrame()
    CenterPortDuration = [x['GUI']['CenterPortDuration'] for x in ts]
    Contingency = [x['GUI']['Contingency'] for x in ts]
    RewardAmount = [x['GUI']['RewardAmount'] for x in ts]
    PunishDelay = [x['GUI']['PunishDelay'] for x in ts]
    BiasCorrection = [x['GUI']['BiasCorrection'] for x in ts]
    FullGUI = [x['GUI'] for x in ts]

    # trial events
    trev = [x['Events'] for x in SessionData['RawEvents']['Trial']]
    if not len(trev) == numberOfTrials:
        print('trial events length do not match with the number of trials')
        return pd.DataFrame()

    # trial states
    trst = [x['States'] for x in SessionData['RawEvents']['Trial']]
    if not len(trst) == numberOfTrials:
        print('trial states length do not match with the number of trials')
        return pd.DataFrame()

    # calculate the cumulative performance
    firstpokecorrect = SessionData['FirstPokeCorrect'][0:numberOfTrials]
    correct_cp = np.cumsum(firstpokecorrect == 1)
    incorrect_cp = np.cumsum(firstpokecorrect == 0)
    # the following line gives an error sometimes
    cumper = 100 * correct_cp / (correct_cp + incorrect_cp)

    # calculate when there is a side-switching event
    TriSide = np.array(SessionData['TrialSide'][0:numberOfTrials])
    SwitchSide = 1 * ((TriSide - np.insert(TriSide[:-1], 0, 0)) != 0)

    # add information about the choice in the previous trial'
    FirstPoke = SessionData['FirstPoke'][0:numberOfTrials]
    PrevTriChoice = np.insert(np.asfarray(FirstPoke[:-1]), 0, np.nan)

    # create a nice ID for the session (pretty date/time)
    prettyDate = SessionID.strftime("%b%d %H:%M")

    DFtoReturn = pd.DataFrame({'AnimalID': pd.Series(np.repeat(AnimalID, numberOfTrials)).astype("category"),
                               'ExperimentalGroup': pd.Series(np.repeat(ExperimentalGroup, numberOfTrials)).astype("category"),
                               'SessionTime': pd.Series(np.repeat(prettyDate, numberOfTrials)).astype("category"),
                               'FullSessionTime': np.repeat(SessionID, numberOfTrials),
                               'Protocol': protocols,
                               'Stimulation': stimulations,
                               'Muscimol': muscimol,
                               'RewardChange': reward_change,
                               'RewardChangeBlock': reward_change_block,
                               'CenterPortDuration': CenterPortDuration,
                               'Contingency': Contingency,
                               'RewardAmount': RewardAmount,
                               'PunishDelay': PunishDelay,
                               'Punish': punish,
                               'BiasCorrection': BiasCorrection,
                               'TrialIndex': list(range(numberOfTrials)),
                               'TrialHighPerc': SessionData['TrialHighPerc'][0:numberOfTrials],
                               'Outcomes': SessionData['Outcomes'][0:numberOfTrials],
                               'OptoStim': SessionData['OptoStim'][0:numberOfTrials],
                               'FirstPokeCorrect': firstpokecorrect,
                               'FirstPoke': FirstPoke,
                               'TrialSide': TriSide,
                               'TrialSequence': SessionData['TrialSequence'][0:numberOfTrials],
                               'ResponseTime': SessionData['ResponseTime'][0:numberOfTrials],
                               'TrialStartTimestamp': SessionData['TrialStartTimestamp'],
                               'CumulativePerformance': cumper,
                               'SwitchSide': SwitchSide,
                               'PreviousChoice': PrevTriChoice,
                               'TrialEvents': trev,
                               'TrialStates': trst,
                               'FullGUI': FullGUI
                               })

    return DFtoReturn


def identifyIdx(datatimes, ntrialsList, ntrials_thr):
    idxlist = []
    for i in (range(len(datatimes))):
        if np.logical_or(datatimes[i] == 'notFound', ntrialsList[i] < ntrials_thr):
            idxlist.append(i)
    return sorted(idxlist, reverse=True)


# Analyze this with the optotrials as well
def AnalyzeSwitchTrials(df):
    # df is a dataframe containing the following columns:
    # 'SwitchSide'
    # 'FirstPokeCorrect'
    # 'SessionTime'
    # 'OptoStim'
    # it returns a different dataframe with information grouped for a bar plot

    # get info for the sessions
    sessionsID = pd.unique(df['SessionTime'])
    # initialize list to hold dataframes
    sessionsInfo = []   

    # fill the new dataframe with info for each session
    for session in sessionsID:
        # get the dataframe for that session
        Sdf = df[df['SessionTime'] == session]
        # split the dataset into opto and normal
        Ndf = Sdf[Sdf['OptoStim'] == 0]
        Odf = Sdf[Sdf['OptoStim'] == 1]
        # percentage of correct trials on stay trials without stimulation
        StayNoStim = 100 * np.sum(Ndf[Ndf['SwitchSide'] == 0]['FirstPokeCorrect'] == 1)/len(Ndf[Ndf['SwitchSide'] == 0])
        # percentage of correct trials on switch trials without stimulation
        SwitchNoStim = 100 * np.sum(Ndf[Ndf['SwitchSide'] == 1]['FirstPokeCorrect'] == 1)/len(Ndf[Ndf['SwitchSide'] == 1])
        # percentage of correct trials on stay trials with stimulation
        StayStim = 100 * np.sum(Odf[Odf['SwitchSide'] == 0]['FirstPokeCorrect'] == 1)/len(Odf[Odf['SwitchSide'] == 0])
        # percentage of correct trials on switch trials with stimulation
        SwitchStim = 100 * np.sum(Odf[Odf['SwitchSide'] == 1]['FirstPokeCorrect'] == 1)/len(Odf[Odf['SwitchSide'] == 1])
        # fill the dataframe
        SessionDF = pd.DataFrame({'SessionTime': np.repeat(session, 4),
                                  'Condition': np.array(['Normal_Stay', 'Normal_Switch', 'Opto_Stay', 'Opto_Switch']),
                                  'PercCorrect': np.array([StayNoStim, SwitchNoStim, StayStim, SwitchStim])
                                 })
        # append it to list
        sessionsInfo.append(SessionDF)

    # merge into a single df and return
    return pd.concat(sessionsInfo, ignore_index=True)


# Analyze this with the trial side as well
def AnalyzeSwitchTrials_for_sides(df):
    # df is a dataframe containing the following columns:
    # 'SwitchSide'
    # 'FirstPokeCorrect'
    # 'SessionTime'
    # 'TrialSide'
    # it returns a different dataframe with information grouped for a bar plot

    # get info for the sessions
    sessionsID = pd.unique(df['SessionTime'])
    # initialize list to hold dataframes
    sessionsInfo = []   

    # fill the new dataframe with info for each session
    for session in sessionsID:
        # get the dataframe for that session
        Sdf = df[df['SessionTime'] == session]
        # split the dataset into opto and normal
        Ndf = Sdf[Sdf['TrialSide'] == 1]
        Odf = Sdf[Sdf['TrialSide'] == 2]
        # percentage of correct trials on stay trials without stimulation
        StayNoStim = 100 * np.sum(Ndf[Ndf['SwitchSide'] == 0]['FirstPokeCorrect'] == 1)/len(Ndf[Ndf['SwitchSide'] == 0])
        # percentage of correct trials on switch trials without stimulation
        SwitchNoStim = 100 * np.sum(Ndf[Ndf['SwitchSide'] == 1]['FirstPokeCorrect'] == 1)/len(Ndf[Ndf['SwitchSide'] == 1])
        # percentage of correct trials on stay trials with stimulation
        StayStim = 100 * np.sum(Odf[Odf['SwitchSide'] == 0]['FirstPokeCorrect'] == 1)/len(Odf[Odf['SwitchSide'] == 0])
        # percentage of correct trials on switch trials with stimulation
        SwitchStim = 100 * np.sum(Odf[Odf['SwitchSide'] == 1]['FirstPokeCorrect'] == 1)/len(Odf[Odf['SwitchSide'] == 1])
        # fill the dataframe
        SessionDF = pd.DataFrame({'SessionTime': np.repeat(session, 4),
                                  'TrialSide': np.array(['Left_Stay', 'Left_Switch', 'Right_Stay', 'Right_Switch']),
                                  'PercCorrect': np.array([StayNoStim, SwitchNoStim, StayStim, SwitchStim])
                                 })
        # append it to list
        sessionsInfo.append(SessionDF)

    # merge into a single df and return
    return pd.concat(sessionsInfo, ignore_index=True)


# function to process the data of an experiment for psychometric performance plots:
def PP_ProcessExperiment(SessionData, bootstrap=None, error_bars=None):
    # SessionData is a dataframe that needs to have the following column names:
    # 'TrialHighPerc'
    # 'FirstPoke'

    diffs = np.array(SessionData['TrialHighPerc'])
    choices = np.array(SessionData['FirstPoke'])

    # Calculate psychometric performance parameters
    PsyPer = PsychPerformance(trialsDif=diffs, sideSelected=choices)
    # predict data
    predictDif = np.linspace(1, 100, 2000).reshape(-1, 1)
    if PsyPer:
        predictPer = 100 * PsyPer['Logit'].predict_proba(predictDif)[:,1]

        # Calculate the error bars if asked to
        if error_bars is not None:
            EBdata = SessionData.groupby(by=error_bars).apply(getEBdata)
            # flatten the lists
            EB_diffs_flat = list(chain(*[x['Difficulty'] for x in EBdata]))
            EB_perfs_flat = list(chain(*[x['Performance'] for x in EBdata]))
            # calculate error bars for each difficulty
            Std_list = [np.std(list(compress(EB_perfs_flat, EB_diffs_flat == dif))) for dif in PsyPer['Difficulty']]
        else:
            Std_list = np.nan
    else:  # needed for the return
        predictPer = np.nan
        Std_list = np.nan

    # Bootstrap on fake data (generated inside the bootstrap function)
    fakePredictions = np.nan
    if bootstrap is not None:
        np.random.seed(12233)  # fixed random seed for reproducibility
        if PsyPer:
            fakePredictions = BootstrapPerformances(trialsDif=diffs,
                                                    sideSelected=choices,
                                                    ntimes=bootstrap,
                                                    prediction_difficulties=predictDif)

    # return what is needed for the plot
    return predictDif, PsyPer, fakePredictions, predictPer, Std_list


def getEBdata(SessionData):
    # SessionData is a dataframe that needs to have the following column names:
    # 'TrialHighPerc'
    # 'FirstPoke'

    diffs = np.array(SessionData['TrialHighPerc'])
    choices = np.array(SessionData['FirstPoke'])

    PsyPer = PsychPerformance(trialsDif=diffs, sideSelected=choices)

    return PsyPer


def timeDifferences(listOfDates):
    '''
    Return the absolute time, in days, of elements in a list of dates, related to the first
    Assumes data is in order (would return negative values otherwise)
    :param listOfDates: list of size X of dates. Format: YYYYMMDD_HHMMSS
    :return: array of size X of absolute time
    '''

    if len(listOfDates) == 0:
        return []

    abstimeList = []
    for date in listOfDates:
        strList = [int(date[0:4]), int(date[4:6]), int(date[6:8]), int(date[9:11]), int(date[11:13]), int(date[13:15])]
        intList = list(map(int, strList))
        # Calculate absolute time in days

        multipliers = [365, 30, 1, 1 / 24, 1 / (24 * 60), 1 / (24 * 60 * 60)]
        mulList = [a * b for a, b in zip(intList, multipliers)]
        abstime = sum(mulList)
        abstimeList.append(abstime)

    diftime = np.array(abstimeList) - abstimeList[0]

    return diftime


def RBias(FirstPokes, FirstPokesCorrect):
    '''
    %Returns the bias to the right
    % FirstPokes is a vector of 1s and 2s (Left or Right), indicating the poked port
    % FirstPokesCorrect is a 0 and 1 vector (wrong or correct poke)
    % Both could have NaN values

    % Returns from -1 to 1. 0 Being not biased, 1 being Right-biased, and
    % -1 being left-biased. It is a conservative function. E.g, in a 50-50
    % trial chance, and being totally biased to one side, only half of the
    % trials would be wrong, so the function would output +/-0.5.

    % Correct trials based on proportion of wrong pokes
    % Determine the proportion of wrong pokes to the right side
    '''
    WrongSides = FirstPokes[FirstPokesCorrect == 0]
    if len(WrongSides) < 1:
        RBias = 0
    else:
        WrongSideProportion = len(WrongSides) / len(FirstPokes)  # from 0 to 1
        WrongRightsProportion = WrongSideProportion * np.nansum(WrongSides == 2) / len(WrongSides)
        WrongLeftsProportion = WrongSideProportion * np.nansum(WrongSides == 1) / len(WrongSides)

        RBias = WrongRightsProportion - WrongLeftsProportion
    return RBias


def CalculateRBiasWindow(FirstPokes, FirstPokesCorrect, Window):
    '''Calculates RBias over the lenght of the vectors FirstPokes and
    FirstPokesCorrect using a Window. Returns vector of same lenght'''
    # Create empty vector
    RBiasVector = np.empty(len(FirstPokes))
    RBiasVector[:] = np.nan
    for i in range(Window, len(FirstPokes)):
        win = range((i - Window), i)
        RBiasVector[i] = RBias(FirstPokes[win], FirstPokesCorrect[win])

    return RBiasVector


# calculate the number of times they go to the middle (anxiousness?)
def CalculateMidPokes(df):
    return np.sum(df['TrialEvents']['Port2In'] <= df['TrialStates']['WaitForResponse'][0])
    # this might fail if WaitForResponse is empty...


# quantify how long they wait in the middle
def MidPortWait(df):
    timeOut = df['TrialStates']['WaitForResponse'].astype('float32')[0]
    PortIn = df['TrialEvents']['Port2In']
    # sometimes this is an integer (rarely)
    if isinstance(PortIn, int):
        PortIn = float(PortIn)
    if not isinstance(PortIn, float):
        PortIn = PortIn.astype('float32')  # does not work for int
        PortInIdx = np.where(PortIn < timeOut)[0][-1]
        PortInTime = PortIn[PortInIdx]
    else:
        PortInTime = PortIn

    PortTime = timeOut - PortInTime
    return PortTime


# quantify the time they take to initiate a trial (from trialstart to center poke in)
def CalculateTrialInitiationTime(df):
    # the first time they poke
    try:
        return float(df.TrialEvents['Port2In'][0])
    except Exception:
        return float('NaN')


def AnalyzePercentageByDay(rdf):
    # df is a dataframe containing the following columns:
    # 'FirstPokeCorrect'
    # 'TrainingDay'
    # 'AnimalID'
    # 'Protocol'
    # 'ExperimentalGroup'
    # it returns a different dataframe with information grouped for a bar plot
    AnimalIDs = pd.unique(rdf['AnimalID'])
    animalsInfo = []
    for animalid in AnimalIDs:
        df = rdf[rdf['AnimalID'] == animalid]
        # get info for the sessions
        TrainingDays = pd.unique(df['TrainingDay'])
        # initialize value for cumulative trials
        CumTrials = 0
        # fill the new dataframe with info for each session
        for session in TrainingDays:
            # get the dataframe for that session
            Sdf = df[df['TrainingDay'] == session]
            # protocol and ExperimentalGroup
            prot = Sdf.Protocol.iloc[0]
            inj = Sdf.ExperimentalGroup.iloc[0]
            # percentage of correct trials
            PercCorrect = 100 * np.sum(Sdf['FirstPokeCorrect']) / len(Sdf)
            # number of trials per session
            NumOfTrials = len(Sdf)
            # cumulative trials
            CumTrials = CumTrials + NumOfTrials
            # fill the dataframe
            SessionDF = pd.DataFrame({
                                      'AnimalID': animalid,
                                      'SessionTime': session,
                                      'PercCorrect': np.array([PercCorrect]),
                                      'NumberOfTrials': NumOfTrials,
                                      'CumulativeTrials': CumTrials,
                                      'Protocol': prot,
                                      'ExperimentalGroup': inj
                                     })
            # append it to list
            animalsInfo.append(SessionDF)
    # merge into a single df and return

    return pd.concat(animalsInfo, ignore_index=True)


def ReadAnimalData(filelist, printout=True):
    # Reads all data from one animal and one protocol

    # Initialize return lists
    ExperimentFiles = []  # to store experiment names
    ExperimentData = []  # to store the dictionaries
    ntrialsDistribution = []  # to visualize the distribution of the number of trials
    Protocols = []  # store the protocols
    Stimulations = []  # store the stimulated protocols
    Muscimol = []  # store information about the muscimol
    counter = 0
    # sort the file list
    filelist = sorted(filelist, key=str.casefold)
    for file in filelist:
        # Read data
        data = load_ns.loadmat(file)

        # if that session is empty skip it:
        if data is None or (not 'nTrials' in data['SessionData']):
            print('Skipping file {0}'.format(file))
            continue

        ntrials = data['SessionData']['nTrials'] # this sometimes fails if the session is empty

        ExperimentFiles.append(file)
        # Parse the settings of the trials
        trial_settings = data['SessionData']['TrialSettings']
        try:
            for trial_num, trial in enumerate(trial_settings):
                trial_settings[trial_num] = load_ns._todict(trial)
            data['SessionData']['TrialSettings'] = trial_settings
        except Exception:
            data['SessionData']['TrialSettings'] = np.nan

        # Get info for the settings from the first trial
        try:
            protocol = trial_settings[0]['GUIMeta']['TrainingLevel']['String'][
                trial_settings[0]['GUI']['TrainingLevel'] - 1]
        except Exception:
            protocol = 'Unknown'

        try:
            stimulation = trial_settings[0]['GUIMeta']['OptoStim']['String'][
                trial_settings[0]['GUI']['OptoStim'] - 1]
        except Exception:
            stimulation = 'unknown'

        try:
            muscimol = trial_settings[0]['GUIMeta']['Muscimol']['String'][
                trial_settings[0]['GUI']['Muscimol'] - 1]
        except Exception:
            muscimol = 'unknown'

        if printout:
            print('{}: {}, {} trials on {}, stim {}, muscimol {}'.format(\
                    counter, ntpath.basename(file), ntrials, protocol, stimulation, muscimol))

        ntrialsDistribution.append(ntrials)
        Protocols.append(protocol)
        Stimulations.append(stimulation)
        Muscimol.append(muscimol)

        # as RawEvents.Trial is a cell array of structs in MATLAB,
        # we have to loop through the array and convert the structs to dicts
        trial_raw_events = data['SessionData']['RawEvents']['Trial']
        try:
            for trial_num, trial in enumerate(trial_raw_events):
                trial_raw_events[trial_num] = load_ns._todict(trial)
            data['SessionData']['RawEvents']['Trial'] = trial_raw_events
        except Exception:
            data['SessionData']['RawEvents']['Trial'] = np.nan

        # Save the data in a list
        ExperimentData.append(data)
        counter += 1

    return ExperimentFiles, ExperimentData, ntrialsDistribution, Protocols, Stimulations, Muscimol


def get_new_files(filelist, existing_dates):
    """
    Compares dates in files to a datetime dataset to check for existing data
        :param filelist: list of full paths to bpod files
        :type filelist: list of strings
        :param existing_dates: time objects in datetime format
        :returns: subset of filelist
    """
    filenames = [ntpath.basename(x) for x in filelist]
    dates = BpodDatesToTime(ParseForTimes(filenames))
    dates_formatted = [str(i) for i in dates]
    existing_dates_formatted = [str(i) for i in existing_dates]
    new_dates = list(set(dates_formatted) - set(existing_dates_formatted))
    new_idx = [i for i, n in enumerate(dates_formatted) if n in new_dates]
    new_files = [filelist[i] for i in new_idx]
    return new_files


def split_files_into_old_and_new(filelist, existing_dates):
    """
    Compares dates in files to a datetime dataset to split them into new files and old files
        :param filelist: list of full paths to bpod files
        :type filelist: list of strings
        :param existing_dates: time objects in datetime format
        :returns: two subsets of filelist
    """
    # files with a new date
    dif_files = get_new_files(filelist, existing_dates)
    # compare dates and split
    # idx of old_files
    filenames = [ntpath.basename(x) for x in dif_files]
    dates = BpodDatesToTime(ParseForTimes(filenames))
    old_idx = [i for i, n in enumerate(dates) if n < existing_dates.max().to_pydatetime()]
    # split
    old_files = [dif_files[i] for i in old_idx]
    new_files = [dif_files[i] for i in list(set(range(len(dif_files))) - set(old_idx))]

    return old_files, new_files


def perf_window_calculator(df, window):
    """
    Calculate the performance of the last X trials

    """
    firstpokecorrect = df['FirstPokeCorrect']  # 0s and 1s
    # create empty vector of the same size
    perf_window = np.full(len(firstpokecorrect), np.nan)
    for i in range(window - 1, len(perf_window)):
        perf_window[i] = np.nansum(firstpokecorrect[i - window + 1: i + 1]) / window * 100
    return perf_window


# calculate the trials per minute that animals do by fitting a line
def trials_per_minute(trial_index, trial_start_timestamp):
    """
    function to calculate the speed of the mouse in trials per minute
    param trial_index: pandas.core.series.Series with the trial index
    param trial_start_timestamp: pandas.core.series.Series with the trial start time in seconds
    returns a value which is the trials per minute
    """
    lrmodel = LinearRegression().fit(trial_index[:, np.newaxis], trial_start_timestamp)

    return 60 * 1 / lrmodel.coef_[0]


def speed_window_calculator(df, window):
    """
    Calculate the speed over X trials

    """
    trial_index = df.TrialIndex
    trial_start_timestamp = df.TrialStartTimestamp
    # create empty vector of the same size
    speed_window = np.full(len(trial_index), np.nan)
    for i in range(int(window / 2) - 1, len(speed_window) - int(window / 2)):
        win_idx_low = i - int(window / 2) + 1
        win_idx_high = i + int(window / 2)
        speed_window[i] = trials_per_minute(trial_index[win_idx_low: win_idx_high],
                                            trial_start_timestamp[win_idx_low: win_idx_high])
    return speed_window


def itis_calculator(df):
    # df is a behavioural dataframe

    # find inter-trial-intervals
    itis = np.diff(df.TrialStartTimestamp)
    # append a 0 at the beginning so it matches the trial indexes
    # how long did the mouse take to initiate this trial from the previous?
    itis = np.insert(itis, 0, 0)

    return itis


def find_disengaged_trials(itis):
    # itis is a vector of inter trial intervals
    # this function returns indexes

    disengaged_indexes = np.where(itis > 3 * np.median(itis))

    return disengaged_indexes


def sigmoid_func(x, slope, bias, upper_lapse, lower_lapse):
    return (upper_lapse - lower_lapse) / (1 + np.exp(-slope * (x - bias))) + lower_lapse


def linear_func(x, beta, alpha):
    return beta * x + alpha


def fit_custom_sigmoid(difficulty, performance):
    # scale the data
    xdatasc = (difficulty - difficulty.mean()) / difficulty.std()
    ydatasc = performance / 100

    cost_func = lambda x: np.mean(np.abs(sigmoid_func(xdatasc, x[0], x[1], x[2], x[3]) - ydatasc))

    res = opt.minimize(cost_func, [-3, 0, 1, 0])

    #rescale
    slope = res.x[0] / difficulty.std()
    bias = res.x[1] * difficulty.std() + difficulty.mean()
    upper_lapse = res.x[2] * 100
    lower_lapse = res.x[3] * 100

    return slope, bias, upper_lapse, lower_lapse


def get_random_optolike_choices(df, n_times=100):
    '''
    gets a dataframe that has optostimulated trials,
    and returns, per each difficulty,
    choices sampled randomly from the non-stimulated trials, n_times
    '''
    normal_df, opto_df = splitOpto(df)
    fake_opto_side_sel_samples = np.zeros((n_times, len(opto_df['SideSelected'])))

    for i in range(n_times):
        fake_opto_side_sel = np.empty_like(opto_df['SideSelected'])
        for curr_diff in np.unique(opto_df['Difficulty']):
            diff_opto_mask = opto_df['Difficulty'] == curr_diff
            diff_normal_mask = normal_df['Difficulty'] == curr_diff
            population = normal_df['SideSelected'][diff_normal_mask]
            fake_opto_side_sel[diff_opto_mask] = np.random.choice(population, sum(diff_opto_mask))
        fake_opto_side_sel_samples[i] = fake_opto_side_sel

    return fake_opto_side_sel_samples


def get_mean_and_std_of_random_optolike_choices(df, n_times=100):

    # deprecated

    '''
    gets a dataframe that has optostimulated trials, and
    outputs, per difficulty, the mean and the std of
    choices sampled randomly from the non-stimulated trials, n_times
    '''
    normal_df, opto_df = splitOpto(df)

    available_difficulties = np.unique(opto_df['Difficulty'])
    random_means = np.zeros_like(available_difficulties)
    random_std = np.zeros_like(available_difficulties)

    for k, curr_diff in enumerate(available_difficulties):
        diff_opto_mask = opto_df['Difficulty'] == curr_diff
        diff_normal_mask = normal_df['Difficulty'] == curr_diff
        population = normal_df['SideSelected'][diff_normal_mask]
        if len(population) == 0:
            sys.exit('No normal trials with that difficulty')
        fake_opto_side_sel_list = np.zeros(n_times)
        for i in range(n_times):
            fake_opto_side_sel_list[i] = np.nanmean(np.random.choice(population, sum(diff_opto_mask)))
        random_means[k] = np.nanmean(fake_opto_side_sel_list)
        random_std[k] = np.nanstd(fake_opto_side_sel_list)

    df_to_return = pd.DataFrame({
        'Difficulty': available_difficulties,
        'Mean_of_choice': 100 * (random_means - 1),
        'Std_of_choice': 100 * random_std
    })

    return df_to_return


def get_choices(sideSelected, trialsDif):
    '''
    returns mean of choices per difficulty
    '''
    # Calculate performance
    # Initialize values
    difficulty = np.unique(trialsDif[~np.isnan(trialsDif)])
    choice_mean = np.full(len(difficulty), np.nan)
    for i in range(len(difficulty)):
        if np.nansum(sideSelected[trialsDif == difficulty[i]]) > 0:
            choice_mean[i] = 100 * (np.nanmean(sideSelected[trialsDif == difficulty[i]]) - 1)

        else:
            choice_mean[i] = np.nan

    return difficulty, choice_mean



DATA_FOLDER_PATHS = {
    'nailgun': '/home/hernandom/data',
    'HMVergara-Laptop': '/mnt/c/Users/herny/Desktop/SWC/Data'
}


def get_data_folder():
    """
    Selects data folder depending on computer name
    """
    computer_name = socket.gethostname()
    try:
        return DATA_FOLDER_PATHS[computer_name]
    except KeyError:
        raise KeyError(f"Unknown data path for computer {computer_name}.")
