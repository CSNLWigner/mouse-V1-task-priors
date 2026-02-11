import os
import h5py
import sqlite3
import numpy as np
import pandas as pd
from .config import BASEDATAPATH

DATASET1 = {
    'CellTable_path': os.path.join(BASEDATAPATH, 'DATASET1_CellTable.csv'),
    'TrialTable_path': os.path.join(BASEDATAPATH, 'DATASET1_TrialTable.csv'),
    'ExperimentTable_path': os.path.join(BASEDATAPATH, 'DATASET1_ExperimentTable.csv'),
    'CellTrialTable_path': os.path.join(BASEDATAPATH, 'DATASET1_CellTrialTable.db'),
    'CellTrialTable_csv_path': os.path.join(BASEDATAPATH, 'DATASET1_CellTrialTable'),
}
PERMISSABLE_FIT_TYPES = [
    'circular_gaussian_360', 
    'direction_selective_circular_gaussian', 
    'circular_gaussian_180',
]
task_experimental_conditions = [
    ('D1', 45,),
    ('D1', 135,),
    ('D2', 45,),
    ('D2', 90,),
    ('D3', 45,),
    ('D3', 75,),
    ('D4', 45,),
    ('D4', 70,),
    ('D5', 45,),
    ('D5', 65,),
    ('D6', 45,),
    ('D6', 60,),
]


def load_tables(DATASET=DATASET1, include_ExperimentTable=False):
    df_celltable = pd.read_csv(DATASET['CellTable_path'])
    df_trialtable = pd.read_csv(DATASET['TrialTable_path'])
    df_experimenttable = pd.read_csv(DATASET['ExperimentTable_path'])
    if include_ExperimentTable:
        return df_celltable, df_trialtable, df_experimenttable
    else:
        return df_celltable, df_trialtable

def retrieve_experiment_ids(loaded_tables,):
    df_celltable, df_trialtable = loaded_tables
    return list(df_celltable['Experiment'].unique())

def retrieve_relevant_experiment_ids(visual_stim, block, loaded_tables,):
    assert isinstance(visual_stim, int), 'visual_stim is not integer'
    assert isinstance(block, str), 'block is not string'
    
    df_celltable, df_trialtable = loaded_tables
    experiments_relevant = df_trialtable[
        (df_trialtable['Visual_Stim'] == visual_stim) &\
        (df_trialtable['Block'] == block)
    ]
    return list(experiments_relevant['Experiment'].unique())

def retrieve_recording_day_experiment_ids(behav_cond, loaded_tables,):
    assert isinstance(behav_cond, str), 'behav_cond is not string'
    
    df_celltable, df_trialtable = loaded_tables
    experiments_relevant = df_trialtable[
        (df_trialtable['Behav_Cond'] == behav_cond)
    ]
    return list(experiments_relevant['Experiment'].unique())

def construct_trial_bank(experiment_ids, loaded_tables):
    """Construct a trial bank with all experiments and trials."""
    assert isinstance(experiment_ids, list), 'experiment_ids is not list'

    df_celltable, df_trialtable = loaded_tables
    trial_bank = {}
    
    for experiment_id in experiment_ids:
        df_experiment = df_trialtable[df_trialtable['Experiment'] == experiment_id]
        trials = np.sort(df_experiment['Trial'].unique())
        trial_bank[experiment_id] = trials

    return trial_bank

def construct_filtered_trial_bank(experiment_ids, visual_stim, block, loaded_tables):
    """Construct a trial bank with for trials with a particular stimulus in a particular behavioral context."""
    assert isinstance(experiment_ids, list), 'experiment_ids is not list'

    df_celltable, df_trialtable = loaded_tables
    trial_bank = {}
    
    for experiment_id in experiment_ids:
        df_experiment = df_trialtable[df_trialtable['Experiment'] == experiment_id]
        df_relevant_trials = df_experiment[
            (df_experiment['Visual_Stim'] == visual_stim) &\
            (df_experiment['Block'] == block)
        ]
        trials = np.sort(df_relevant_trials['Trial'].unique())
        trial_bank[experiment_id] = trials

    return trial_bank

def construct_testing_trial_bank(experiment_ids, behav_cond, visual_stim, loaded_tables):
    """Construct a trial bank with visual stimulus from the testing context in a specific recording day."""
    assert isinstance(experiment_ids, list), 'experiment_ids is not list'
    assert isinstance(behav_cond, str), 'behav_cond is not string'
    assert isinstance(visual_stim, int), 'visual_stim is not integer'

    df_celltable, df_trialtable = loaded_tables
    trial_bank = {}

    for experiment_id in experiment_ids:
        df_experiment = df_trialtable[df_trialtable['Experiment'] == experiment_id]
        df_relevant_trials = df_experiment[
            (df_experiment['Behav_Cond'] == behav_cond) &\
            (df_experiment['Block'] == 'Visual') &\
            (df_experiment['Visual_Stim'] == visual_stim)
        ]
        trials = np.sort(df_relevant_trials['Trial'].unique())
        trial_bank[experiment_id] = trials

    return trial_bank

def construct_tuning_trial_bank(experiment_ids, behav_cond, visual_stim, loaded_tables):
    """Construct a trial bank with visual stimulus from the testing context in a specific recording day."""
    assert isinstance(experiment_ids, list), 'experiment_ids is not list'
    assert isinstance(behav_cond, str), 'behav_cond is not string'
    assert isinstance(visual_stim, int), 'visual_stim is not integer'

    df_celltable, df_trialtable = loaded_tables
    trial_bank = {}

    for experiment_id in experiment_ids:
        df_experiment = df_trialtable[df_trialtable['Experiment'] == experiment_id]
        df_relevant_trials = df_experiment[
            (df_experiment['Behav_Cond'] == behav_cond) &\
            (df_experiment['Block'] == 'Orientation Tuning') &\
            (df_experiment['Visual_Stim'] == visual_stim)
        ]
        trials = np.sort(df_relevant_trials['Trial'].unique())
        trial_bank[experiment_id] = trials

    return trial_bank

def construct_multi_day_trial_bank(recording_days, experiment_block, vis_stim, tables):
    trial_bank = {}
    for day in recording_days:
        relevant_experiment_ids = retrieve_recording_day_experiment_ids(day, tables)

        if experiment_block == "testing":
            trial_bank_day = construct_testing_trial_bank(
                relevant_experiment_ids, day, 
                vis_stim, tables,
            )
        elif experiment_block == "tuning":
            trial_bank_day = construct_tuning_trial_bank(
                relevant_experiment_ids, day, 
                vis_stim, tables,
            )
        else:
            raise ValueError(f'Invalid experiment_block of {experiment_block}')

        trial_bank.update(trial_bank_day)

    return trial_bank

def construct_behavior_trial_bank(experiment_ids, day, visual_stim, response, loaded_tables):
    assert isinstance(experiment_ids, list), 'experiment_ids is not list'
    assert isinstance(day, str), 'day is not string'
    assert isinstance(visual_stim, int), 'visual_stim is not integer'
    assert response in [0,1], 'response is not a boolean'

    df_celltable, df_trialtable = loaded_tables
    trial_bank = {}

    for experiment_id in experiment_ids:
        df_experiment = df_trialtable[df_trialtable['Experiment'] == experiment_id]
        df_relevant_trials = df_experiment[
            (df_experiment['Behav_Cond'] == day) &\
            (df_experiment['Block'] == 'Visual') &\
            (df_experiment['Response'] == response) &\
            (df_experiment['Visual_Stim'] == visual_stim)
        ]
        trials = np.sort(df_relevant_trials['Trial'].unique())
        trial_bank[experiment_id] = trials

    return trial_bank

def compute_trial_percentages(df):
    """
    Compute Standardized_Trial and Trial_Percentage for each mouse in the dataset.
    
    Parameters:
        df (pd.DataFrame): The trial table DataFrame.
    
    Returns:
        pd.DataFrame: A copy of the DataFrame with added Standardized_Trial and Trial_Percentage.
    """
    df_copy = df.copy()
    df_copy["Standardized_Trial"] = df_copy.groupby("Experiment")["Trial"].rank(method="first")
    df_copy["Trial_Percentage"] = df_copy.groupby("Experiment")["Standardized_Trial"].transform(lambda x: x / x.max())
    return df_copy

def construct_trial_bank_by_percentage_behavior(experiment_ids, visual_stim, behav_cond, exp_range, behavior, loaded_tables):
    """
    Constructs a trial bank with trials from the visual block that fall within a given experiment range.

    Parameters:
        experiment_ids (list): List of experiment identifiers.
        visual_stim (int): The visual stimulus condition to filter.
        behav_cond (str): The behavioral condition (e.g., "D1").
        exp_range (tuple): A tuple (min_percentage, max_percentage) specifying the range of trials 
                           to extract based on the experiment progress (range: [0, 1]).
        behavior (list): A list of 0 or 1 corresponding to licking.
        loaded_tables (tuple): Contains (df_celltable, df_trialtable).

    Returns:
        dict: A dictionary mapping `experiment_id` to a list of filtered trial numbers.
    """
    assert isinstance(experiment_ids, list), "experiment_ids should be a list"
    assert isinstance(visual_stim, int), "visual_stim should be an integer"
    assert isinstance(behav_cond, str), "behav_cond should be a string"
    assert isinstance(exp_range, tuple) and len(exp_range) == 2, "exp_range should be a tuple of two floats"
    assert 0 <= exp_range[0] < exp_range[1] <= 1, "exp_range values should be between 0 and 1"
    assert isinstance(behavior, list), "behavior should be a list"

    df_celltable, df_trialtable = loaded_tables
    df_trialtable = df_trialtable[df_trialtable["Block"] == "Visual"]
    df_trialtable = compute_trial_percentages(df_trialtable)  # Compute trial percentage within each experiment

    trial_bank = {}

    for experiment_id in experiment_ids:
        df_experiment = df_trialtable[df_trialtable["Experiment"] == experiment_id]
        df_relevant_trials = df_experiment[
            (df_experiment["Visual_Stim"] == visual_stim)
            & (df_experiment["Behav_Cond"] == behav_cond)
            & (df_experiment["Trial_Percentage"] >= exp_range[0])
            & (df_experiment["Trial_Percentage"] <= exp_range[1])
            & (df_experiment["Response"].isin(behavior))
        ]

        trials = np.sort(df_relevant_trials["Trial"].unique())
        trial_bank[experiment_id] = trials

    return trial_bank

def retrieve_admissible_neurons_old(relevant_experiments, loaded_tables, fit_types=PERMISSABLE_FIT_TYPES):
    """This function retrieves neurons WITHOUT the bioRxiv inclusion criteria."""
    assert isinstance(relevant_experiments, list), 'relevant_experiments is not list'
    
    df_celltable, df_trialtable = loaded_tables
    admissible_neurons = df_celltable[
        (df_celltable['Experiment'].isin(relevant_experiments)) &\
        (~df_celltable['DeconvA_Tau'].isna()) &\
        (df_celltable['Best_Fit_spikes_2'].isin(fit_types))
    ]
    return admissible_neurons[['Cell','Experiment','Best_Fit_spikes_2','Pref_Orientation_spikes_2']]

def retrieve_admissible_neurons(
    relevant_experiments, 
    loaded_tables, 
    fit_key='Best_Fit_spikes_2',
    orientation_key='Pref_Orientation_spikes_2',
    fit_types=PERMISSABLE_FIT_TYPES,
):
    """
    This function retrieves neurons VIA the bioRxiv inclusion criteria.
    Inclusion criteria are taken from https://zenodo.org/records/8109858 
    Naive animals do not have `Best_Fit_spikes_2` key in dataframe, so the fit_key is used.
    """
    assert isinstance(relevant_experiments, list), 'relevant_experiments is not list'
    
    df_celltable, df_trialtable = loaded_tables
    admissible_neurons = df_celltable[
        (df_celltable['Experiment'].isin(relevant_experiments)) &\
        (~df_celltable[fit_key].isna()) &\
        (df_celltable['DeconvCorr'] > 0.8) &\
        (df_celltable['isCell'] > 0.8) &\
        (df_celltable['spikeProb_TCTrials'] > 0.1) &\
        (df_celltable['roundness'] > 0.20) &\
        (df_celltable['nPix']*df_celltable['magFactor'] > 15)
    ].copy()
    admissible_neurons = admissible_neurons[['Cell','Experiment',fit_key,orientation_key]]
    if orientation_key == 'Pref_Orientation_spikes_1':
        admissible_neurons.loc[:, 'Pref_Orientation_spikes_2'] = admissible_neurons[orientation_key]
    return admissible_neurons

def retrieve_trial_matrix(experiment_id, trial_number, DATASET=DATASET1,):
    hdf5_path = os.path.join(
        DATASET['CellTrialTable_csv_path'], 
        experiment_id[:3], 
        f"{experiment_id}.h5",
    )
    
    with h5py.File(hdf5_path, 'r') as hdf:
        trial_name = f'trial_{trial_number}'
        trial_matrix = hdf[trial_name][:]
        
    return trial_matrix
    
def retrieve_spike_trace(experiment_id, trial_number, cell_number, DATASET=DATASET1,):
    """
    This function is not efficient. It's useful for intuitive looking code, but 
    large querries need optimization so that accessing HDF5 files is efficient.
    """
    hdf5_path = os.path.join(
        DATASET['CellTrialTable_csv_path'], 
        experiment_id[:3], 
        f"{experiment_id}.h5",
    )

    cell_index = cell_number - 1 # Very important to decrease by 1 for proper indexing
    with h5py.File(hdf5_path, 'r') as hdf:
        trial_name = f'trial_{trial_number}'
        trial_matrix = hdf[trial_name][:]
        cell_trial_spike_train = trial_matrix[:, cell_index]

    return cell_trial_spike_train

def sample_trial_and_retrieve_trace(rng, neuron_row, trial_bank,):
    experiment_id = neuron_row['Experiment']
    cell_id = neuron_row['Cell'].item()

    trial_array = trial_bank[experiment_id]
    trial_id = rng.choice(trial_array).item()

    spike_trace = retrieve_spike_trace(experiment_id, trial_id, cell_id,)
    return spike_trace

def sample_trial_and_retrieve_trace_FAST(rng, neuron_tuple, trial_bank,):
    experiment_id = neuron_tuple.Experiment
    cell_id = neuron_tuple.Cell

    trial_array = trial_bank[experiment_id]
    trial_id = rng.choice(trial_array).item()

    spike_trace = retrieve_spike_trace(experiment_id, trial_id, cell_id,)
    return spike_trace

def retrieve_tuning_orientations(tables):
    df_celltable, df_trialtable = tables
    tuning_stimulus = np.sort(df_trialtable[df_trialtable['Block'] == 'Orientation Tuning']['Visual_Stim'].unique())
    return tuning_stimulus

def two_second_stimulus_interval(trial_length, starttime=0, stoptime=2):
    xval = np.linspace(0, (trial_length-1) * 0.0646, trial_length) - 1.5  # Time axis
    whenStim = (xval > starttime) & (xval < stoptime)  # Define the time window
    return whenStim

def retrieve_unique_trials_in_experiment(experiment_id, block, loaded_tables):
    """Return a numpy array of trial numbers from a particular task block in an experiment."""
    assert isinstance(experiment_id, str), 'experiment_ids is not list'

    df_celltable, df_trialtable = loaded_tables
    df_experiment = df_trialtable[df_trialtable['Experiment'] == experiment_id]
    df_relevant_trials = df_experiment[(df_experiment['Block'] == block)]
    trials = np.sort(df_relevant_trials['Trial'].unique())
    return trials

def load_experiment_into_memory(experiment_id, trial_array, DATASET=DATASET1, tqdm_disable=True,):
    """Load data from a single experiment from specified trials into memory."""
    hdf5_path = os.path.join(
        DATASET['CellTrialTable_csv_path'], 
        experiment_id[:3], 
        f"{experiment_id}.h5"
    )

    with h5py.File(hdf5_path, 'r') as hdf:
        # Load all trials for the current experiment
        experiment_data = {
            f'trial_{trial_id}': hdf[f'trial_{trial_id}'][:]
            for trial_id in trial_array
        }

    return experiment_data