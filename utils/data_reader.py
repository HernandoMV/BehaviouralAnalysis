# Data_reader.py

from scipy.io import loadmat
import pandas as pd
from . import custom_functions


def mat_parser(path_to_file):
    mat = loadmat(path_to_file, squeeze_me=True)

    # Get the name of the experiment
    main_name = mat['block']['expRef'][()]

    # Get events, inputs and outputs
    events = mat['block']['events'][()]
    # outputs = mat['block']['outputs'][()]
    inputs = mat['block']['inputs'][()]

    # Get Moving Azimuth from events
    moving_az = pd.DataFrame(data=[list(events['MovingAzimuthValues'][()]),
                                   custom_functions.time_to_zero(list(events['MovingAzimuthTimes'][()]))])
    moving_az = moving_az.transpose()
    moving_az.columns = ['MovingAzimuthValues', 'MovingAzimuthTimes']

    # Get data about the side of the stimulus
    trial_side = pd.DataFrame(data=[custom_functions.time_to_zero(list(events['TrialSideTimes'][()])),
                                    list(events['TrialSideValues'][()])])
    trial_side = trial_side.transpose()
    trial_side.columns = ['TrialSideTimes', 'TrialSideValues']
    trial_side['TrialSideVM'] = trial_side['TrialSideValues'] * 80

    # Get data about the successes and failures
    target_reached = pd.DataFrame(
        data=[custom_functions.time_to_zero(list(events['TargetReachedTimes'][()])),
              list(events['TargetReachedValues'][()])])
    target_reached = target_reached.transpose()
    target_reached.columns = ['TargetReachedTimes', 'TargetReachedValues']
    target_reached_events = \
        target_reached.iloc[custom_functions.first_diff_zero(target_reached['TargetReachedValues'])][
            'TargetReachedTimes']

    wrong_reached = pd.DataFrame(data=[custom_functions.time_to_zero(list(events['WrongReachedTimes'][()])),
                                       list(events['WrongReachedValues'][()])])
    wrong_reached = wrong_reached.transpose()
    wrong_reached.columns = ['WrongReachedTimes', 'WrongReachedValues']
    wrong_reached_events = wrong_reached.iloc[custom_functions.first_diff_zero(wrong_reached['WrongReachedValues'])][
        'WrongReachedTimes']

    # Get data about the licks
    licks = pd.DataFrame(data=[custom_functions.time_to_zero(list(inputs['lickTimes'][()])),
                               list(inputs['lickValues'][()])])
    licks = licks.transpose()
    licks.columns = ['lickTimes', 'lickValues']

    data_dict = {
        "Main_name": main_name,
        "Moving_az": moving_az,
        "Trial_side": trial_side,
        "Target_reached": target_reached_events,
        "Wrong_reached": wrong_reached_events,
        "Licks": licks
    }
    return data_dict
