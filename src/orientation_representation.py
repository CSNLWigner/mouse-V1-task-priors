import os
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d
from scipy import stats

from .data_access import (
DATASET1, 
two_second_stimulus_interval,
construct_behavior_trial_bank,
retrieve_recording_day_experiment_ids, 
retrieve_admissible_neurons, 
)


def enforce_spike_trace_length(spike_trace, desired_length):
    original_trace_len = len(spike_trace)
    
    if original_trace_len == desired_length:
        return spike_trace
        
    elif original_trace_len > desired_length:
        return spike_trace[:desired_length]

    else:
        length_difference = desired_length - original_trace_len
        return np.pad(spike_trace, (0, length_difference), mode='constant')

def load_all_experiments_into_memory(trial_bank, DATASET=DATASET1, tqdm_disable=True,):
    experiment_data = {}

    for experiment_id in tqdm(trial_bank.keys(), disable=tqdm_disable):
        hdf5_path = os.path.join(
            DATASET['CellTrialTable_csv_path'], 
            experiment_id[:3], 
            f"{experiment_id}.h5"
        )

        with h5py.File(hdf5_path, 'r') as hdf:
            # Load all trials for the current experiment
            experiment_data[experiment_id] = {
                f'trial_{trial_id}': hdf[f'trial_{trial_id}'][:]
                for trial_id in trial_bank[experiment_id]
            }

    return experiment_data

def retrieve_spike_traces_from_memory(rng, trial_matrix, experiment_id, trial_bank, df_experiment, trial_length, experiment_data):
    trial_ids = rng.choice(trial_bank[experiment_id], size=len(df_experiment), replace=True,)
    df_experiment['trial_ids'] = trial_ids

    for row in df_experiment.itertuples():
        cell_index = row.Cell - 1  # Adjust for 0-based index
        trial_name = f'trial_{row.trial_ids}'
        orientation_index = int(row.Pref_Orientation_spikes_2) - 1  # Adjust for 0-based index

        # Retrieve spike trace from the preloaded data
        spike_trace = experiment_data[experiment_id][trial_name][:, cell_index]
        trial_matrix[orientation_index, :] += enforce_spike_trace_length(spike_trace, trial_length)

    return trial_matrix

def generate_resampled_trial_matrix(admissible_neurons, trial_bank, trial_length, experiment_data, random_seed, tqdm_disable=True):
    rng = np.random.default_rng(random_seed) # Very important to use the random_seed
    
    # First
    trial_matrix = np.zeros((360, trial_length,))

    # Second
    sampled_neurons = admissible_neurons.sample(250, random_state=random_seed, replace=False)

    # Third, Fourth, Fifth
    for experiment_id, df_experiment in tqdm(sampled_neurons.groupby(['Experiment']), disable=tqdm_disable):
         trial_matrix = retrieve_spike_traces_from_memory(
             rng, 
             trial_matrix, 
             experiment_id[0], 
             trial_bank, 
             df_experiment,
             trial_length,
             experiment_data,
         )

    # Sixth
    trial_matrix = trial_matrix - np.mean(trial_matrix)

    # Seventh
    trial_matrix = gaussian_filter1d(trial_matrix, sigma=6, axis=0, mode='wrap')

    return trial_matrix

def construct_sampled_orientation_representation_LEGACY(rng_seed, admissible_neurons, trial_bank, trial_length, experiment_data, num_trials=1000):
    """Use the newer `construct_sampled_orientation_representation` function on line 273."""
    rng = np.random.default_rng(rng_seed) # Very important to use the rng_seed
    random_seeds = rng.integers(low=0, high=2**32, size=num_trials)

    results = []
    for random_seed in tqdm(random_seeds):
        trial_matrix = generate_resampled_trial_matrix(
            admissible_neurons, 
            trial_bank, 
            trial_length, 
            experiment_data, 
            random_seed, 
        )
        results.append(trial_matrix)
        
    return np.mean(np.array(results), axis=0)

def construct_orientation_representation_neurons(admissible_neurons, experiment_data, trial_length, tqdm_disable=True):
    # Dictionary to store traces organized by orientation -> neuron -> trials
    orientation_dict = {i: [] for i in range(360)}
    
    for row in tqdm(admissible_neurons.itertuples(), disable=tqdm_disable):
        experiment_id = row.Experiment
        cell_index = row.Cell - 1
        orientation_index = int(row.Pref_Orientation_spikes_2) - 1
        
        neuron_trials = []
        for trial_id in experiment_data[experiment_id]:
            spike_trace = experiment_data[experiment_id][trial_id][:, cell_index]
            spike_trace = enforce_spike_trace_length(spike_trace, trial_length)
            # Subtract mean activity from before stimulus presentation
            spike_trace = spike_trace - np.mean(spike_trace[:23])
            neuron_trials.append(spike_trace)
        
        if len(neuron_trials) > 0:
            # Stack trials for this neuron into a 2D array (trials x timeframes)
            neuron_array = np.array(neuron_trials)
            orientation_dict[orientation_index].append(neuron_array)
    
    return orientation_dict


# helper to concatenate activities from neuronstrials data into single array
def concatenate_orientation_activities(activity_neuronstrials, orientation_idx, binwidth=6):
    max_key = max(activity_neuronstrials.keys()) if activity_neuronstrials else 0
    orientation_range = 360 if max_key >= 180 else 180
    all_values = []
    for oidx in range(orientation_idx-binwidth//2, orientation_idx+binwidth//2 + 1):
        if orientation_range==360:
            # For 360: wrap around
            key = (oidx-1) % 360
        else:             # For 180: clip to valid range, no wrapping
            key = oidx - 1
            if key<0 or key>=180:
                continue
        
        if key not in activity_neuronstrials:
            continue
        neurons_list = activity_neuronstrials[key]
        for neuron_trials in neurons_list:
            # neuron_trials shape: (n_trials, n_frames)
            all_values.append(neuron_trials.flatten())
    if not all_values:
        return np.array([])
    return np.concatenate(all_values)



def compile_activity_profile_neuronstrials(activity_neuronstrials):
    """
    Compile activity profile from neuronstrials data by averaging across frames only.
    
    For each orientation bin, finds the minimum number of trials across all neurons
    in that bin, truncates all neurons to that number of trials, then averages across
    frames and neurons to produce a vector of trials for each orientation.
    
    Parameters:
    -----------
    activity_neuronstrials : dict
        Dictionary mapping orientation indices to lists of neuron trial arrays.
        Each neuron array has shape (n_trials, n_frames).
    
    Returns:
    --------
    activity_profile : np.array
        2D array of shape (orientations, min_trials) with mean activity per orientation and trial.
    """
    max_key = max(activity_neuronstrials.keys()) if activity_neuronstrials else 0
    # orientation_range = 360 if max_key >= 180 else 180
    orientation_range = 180
    
    # First pass: find global minimum number of trials across all orientations
    global_min_trials = float('inf')
    for orientation_idx in range(orientation_range):
        if orientation_idx in activity_neuronstrials and len(activity_neuronstrials[orientation_idx]) > 0:
            neurons_list = activity_neuronstrials[orientation_idx]
            min_trials_this_ori = min(neuron_trials.shape[0] for neuron_trials in neurons_list)
            global_min_trials = min(global_min_trials, min_trials_this_ori)
    
    if global_min_trials == float('inf'):
        global_min_trials = 0
    
    activity_profile = np.full((orientation_range, global_min_trials), np.nan)
    
    for orientation_idx in range(orientation_range):
        if orientation_idx not in activity_neuronstrials:
            continue
            
        neurons_list = activity_neuronstrials[orientation_idx]
        
        if len(neurons_list) == 0:
            continue
        
        # Truncate each neuron to global_min_trials and average across frames
        neuron_trial_means = []
        for neuron_trials in neurons_list:
            # Randomly sample trials without replacement
            trial_indices = np.random.permutation(neuron_trials.shape[0])[:global_min_trials]
            truncated = neuron_trials[trial_indices, :]
            # Average across frames (axis=1) to get one value per trial
            trial_means = np.nanmean(truncated, axis=1)
            neuron_trial_means.append(trial_means)
        
        # Average across neurons (axis=0) to get one value per trial for this orientation
        activity_profile[orientation_idx, :] = np.nanmean(neuron_trial_means, axis=0)
    
    return activity_profile



# Helper to compute aggregate statistics from neuronstrials data
def compute_orientation_stats(activity_neuronstrials, orientation_idx, binwidth=6, autocorrcorrection=6):
    """Compute mean, std, count over all neurons, trials, and frames for given orientation."""
    # Collect all values: flatten across neurons, trials, and frames

    all_values = concatenate_orientation_activities(activity_neuronstrials, orientation_idx, binwidth=binwidth)

    if len(all_values)==0:
        return np.nan, np.nan, 0

    mean_val = np.nanmean(all_values)
    std_val = np.nanstd(all_values)
    count_val = np.sum(~np.isnan(all_values))/autocorrcorrection
    
    return mean_val, std_val, count_val

def compile_std_neuronstrials(activity_neuronstrials, orientations, binwidth=6, autocorrcorrection=6):
    # establishing confidence intervals around the profile curves
    stds_nt = np.zeros(len(orientations))
    for ori in orientations:
        oix = (ori-1) % 360
        mean_nt, std_nt, count_nt = compute_orientation_stats(activity_neuronstrials, oix, binwidth=binwidth, autocorrcorrection=autocorrcorrection)
        stds_nt[oix] = std_nt/np.sqrt(count_nt) if count_nt>0 else 0.0
    stds_nt = gaussian_filter1d(stds_nt, sigma=6, mode='wrap')
    return stds_nt


def compute_orientation_ttest(activity_neuronstrials, orientation_idx1, orientation_idx2=None, fixed_value=None, one_tailed=False, binwidth=6, autocorrcorrection=6):
    """
    Compute t-statistic comparing activity at one or two orientations.
    
    Parameters:
    -----------
    activity_neuronstrials : dict
        Dictionary mapping orientation indices to lists of neuron trial arrays
    orientation_idx1 : int
        First orientation index to compare
    orientation_idx2 : int, optional
        Second orientation index to compare. If None, uses fixed_value instead
    fixed_value : float, optional
        Fixed value to compare against if orientation_idx2 is None
    one_tailed : bool, optional
        If True, performs one-tailed test (assumes orientation_idx1 > orientation_idx2 or fixed_value)
        If False, performs two-tailed test (default)
        
    Returns:
    --------
    t_stat : float
        T-statistic value
    p_val : float
        P-value
    """
    # Collect all values for first orientation (with binwidth)
    all_values1 = []
    for oidx in range(orientation_idx1-binwidth//2, orientation_idx1+binwidth//2 + 1):
        neurons_list = activity_neuronstrials[oidx]
        for neuron_trials in neurons_list:
            all_values1.append(neuron_trials.flatten())
    
    if not all_values1:
        return np.nan, np.nan
    
    all_values1 = np.concatenate(all_values1)
    all_values1 = all_values1[~np.isnan(all_values1)]
    
    if len(all_values1) == 0:
        return np.nan, np.nan

    
    # Two-sample t-test
    if orientation_idx2 is not None:
        # Collect all values for second orientation (with binwidth)
        all_values2 = []
        for oidx in range(orientation_idx2-binwidth//2, orientation_idx2+binwidth//2 + 1):
            neurons_list = activity_neuronstrials[oidx]
            for neuron_trials in neurons_list:
                all_values2.append(neuron_trials.flatten())
        
        if not all_values2:
            return np.nan, np.nan
        
        all_values2 = np.concatenate(all_values2)
        all_values2 = all_values2[~np.isnan(all_values2)]
        
        if len(all_values2) == 0:
            return np.nan, np.nan

        
        # Compute two-sample t-statistic
        mean1, mean2 = np.mean(all_values1), np.mean(all_values2)
        var1, var2 = np.var(all_values1, ddof=1), np.var(all_values2, ddof=1)
        n1, n2 = len(all_values1)/autocorrcorrection, len(all_values2)/autocorrcorrection
        
        pooled_se = np.sqrt(var1/n1 + var2/n2)
        if pooled_se == 0:
            return np.nan, np.nan
        
        t_stat = (mean1 - mean2) / pooled_se
        df = n1 + n2 - 2
        
        if one_tailed:
            # One-tailed: testing if orientation_idx1 > orientation_idx2
            p_val = 1 - stats.t.cdf(t_stat, df)
        else:
            # Two-tailed
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
    # One-sample t-test against fixed value
    elif fixed_value is not None:
        mean1 = np.mean(all_values1)
        std1 = np.std(all_values1, ddof=1)
        n1 = len(all_values1)/autocorrcorrection
        
        if std1 == 0 or n1 == 0:
            return np.nan, np.nan
        
        t_stat = (mean1 - fixed_value) / (std1 / np.sqrt(n1))
        df = n1 - 1
        
        if one_tailed:
            # One-tailed: testing if orientation_idx1 > fixed_value
            p_val = 1 - stats.t.cdf(t_stat, df)
        else:
            # Two-tailed
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
    else:
        raise ValueError("Must provide either orientation_idx2 or fixed_value")
    
    return t_stat, p_val

def bootstrap_peakshift_fromstimulus(activity_neuronstrials, presented_stimulus, num_bootstraps=1000, binwidth=6, autocorrcorrection=6, neurons_per_orientation=None, tqdm_disable=True):
    rng = np.random.default_rng()
    peak_shifts = []

    for _ in tqdm(range(num_bootstraps), disable=tqdm_disable):
        # Resample neurons with replacement for each orientation
        resampled_activity = {}
        for orientation_idx in range(180):
            neurons_list = activity_neuronstrials[orientation_idx]
            if len(neurons_list) == 0:
                resampled_activity[orientation_idx] = []
                continue
            # Use neurons_per_orientation if specified, otherwise use original length
            sample_size = neurons_per_orientation if neurons_per_orientation is not None else len(neurons_list)
            indices = rng.choice(len(neurons_list), size=sample_size, replace=True)
            resampled_neurons = [neurons_list[i] for i in indices]
            resampled_activity[orientation_idx] = resampled_neurons

        # Compute mean activity for each orientation
        mean_activities = []
        for orientation_idx in range(180):
            mean_val, _, _ = compute_orientation_stats(
                resampled_activity, 
                orientation_idx, 
                binwidth=binwidth, 
                autocorrcorrection=autocorrcorrection
            )
            mean_activities.append(mean_val)
        
        mean_activities = np.array(mean_activities)

        # Identify peak orientation
        peak_orientation = np.nanargmax(mean_activities)
        # Compute shift from presented stimulus
        shift = abs(peak_orientation - presented_stimulus)
        peak_shifts.append(shift)

    return np.array(peak_shifts)



def construct_orientation_representation(admissible_neurons, experiment_data, trial_length, tqdm_disable=True):
    trial_matrix = np.zeros((360, trial_length,))
    count_array = np.zeros(360)
    
    for row in tqdm(admissible_neurons.itertuples(), disable=tqdm_disable):
        experiment_id = row.Experiment
        cell_index = row.Cell - 1
        cell_spike_trace = np.zeros(trial_length)
        if hasattr(row, 'Pref_Orientation_spikes_1') and pd.notna(row.Pref_Orientation_spikes_1):
            r = row.Pref_Orientation_spikes_1
        elif hasattr(row, 'Pref_Orientation_spikes_2') and pd.notna(row.Pref_Orientation_spikes_2):
            r = row.Pref_Orientation_spikes_2
        orientation_index = int(r) - 1
    
        for trial_id in experiment_data[experiment_id]:
            spike_trace = experiment_data[experiment_id][trial_id][:,cell_index]
            spike_trace = enforce_spike_trace_length(spike_trace, trial_length)
            cell_spike_trace += spike_trace

        if len(experiment_data[experiment_id].keys()) > 0:
            cell_spike_trace = cell_spike_trace / len(experiment_data[experiment_id].keys())
            # Subtract mean activity from before stimulus presentation
            cell_spike_trace = cell_spike_trace - np.mean(cell_spike_trace[:23])
            # Previous implementations performed baseline subtraction differently
            trial_matrix[orientation_index, :] += cell_spike_trace
            count_array[orientation_index] += 1

    safe_count_array = np.where(count_array == 0, 1, count_array)
    trial_matrix = trial_matrix / safe_count_array[:, np.newaxis]
    trial_matrix = gaussian_filter1d(trial_matrix, sigma=6, axis=0, mode='wrap')
    
    return trial_matrix

def construct_orientation_representation_RAW(admissible_neurons, experiment_data, trial_length, tqdm_disable=True):
    """
    No subtracting mean or gaussian filtering is performed.
        - This method is depreciated given that baseline subtraction is not performed on whole trial matrices.
    """
    trial_matrix = np.zeros((360, trial_length,))
    count_array = np.zeros(360)
    
    for row in tqdm(admissible_neurons.itertuples(), disable=tqdm_disable):
        experiment_id = row.Experiment
        cell_index = row.Cell - 1
        cell_spike_trace = np.zeros(trial_length)
        orientation_index = int(row.Pref_Orientation_spikes_2) - 1
    
        for trial_id in experiment_data[experiment_id]:
            spike_trace = experiment_data[experiment_id][trial_id][:,cell_index]
            spike_trace = enforce_spike_trace_length(spike_trace, trial_length)
            cell_spike_trace += spike_trace

        if len(experiment_data[experiment_id].keys()) > 0:
            cell_spike_trace = cell_spike_trace / len(experiment_data[experiment_id].keys())
            trial_matrix[orientation_index, :] += cell_spike_trace
            count_array[orientation_index] += 1

    safe_count_array = np.where(count_array == 0, 1, count_array)
    trial_matrix = trial_matrix / safe_count_array[:, np.newaxis]
    
    return trial_matrix

def plot_orientation_representation(trial_matrix, title, save_fig=False):
    plt.figure(figsize=(8, 4))
    plt.imshow(
        trial_matrix,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest',
        vmin=-0.01,
        vmax=0.04
    )
    plt.colorbar(label='APrE/trial')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Preferred orientations')
    
    # Set the y-axis ticks and labels
    y_ticks = np.arange(29, 360, 30)  # Ticks starting from 30 to 360 with steps of 30
    plt.yticks(y_ticks, y_ticks + 1)  # Adding 1 to range because the index is zero-based
    
    if save_fig:
        plt.savefig(save_fig)
    plt.show()

def plot_orientation_activity(activity_matrix, presented_stimulus, title=False, save_fig=False):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    
    # Left Panel: Orientation Space Activity Map
    im = axs[0].imshow(
        activity_matrix,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest',
        vmin=-0.01,
        vmax=0.04
    )
    axs[0].axhline(y=presented_stimulus-1, color='tab:red', linestyle='--')
    axs[0].axvline(x=24, color='tab:green', linestyle='--')
    axs[0].axvline(x=54, color='tab:green', linestyle='--')
    fig.colorbar(im, ax=axs[0], label='APrE/trial')
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('Preferred orientations')
    axs[0].set_title('Orientation Space Activity Map')
    y_ticks = np.arange(29, 360, 30)  # Ticks starting from 30 to 360 with steps of 30
    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_ticks + 1)  # Adding 1 to range because the index is zero-based
    
    # Right Panel: Orientation Space Activity Profile
    whenStim = two_second_stimulus_interval(activity_matrix.shape[-1])
    activity_profile = activity_matrix[:, whenStim]
    mean_response = np.nanmean(activity_profile, axis=1)
    std_response = np.nanstd(activity_profile, axis=1)
    
    axs[1].fill_between(
        np.arange(1, 361),
        mean_response - std_response,
        mean_response + std_response,
        alpha=0.2,
        label='±1 SD'
    )
    axs[1].axhline(y=0.0, color='tab:grey', linestyle='--',)
    axs[1].plot(np.arange(1, 361), mean_response, label='Mean Activity', color='tab:blue')
    axs[1].axvline(x=presented_stimulus, color='tab:red', linestyle='--', label=f'{presented_stimulus}° stimulus')
    x_ticks = np.arange(30, 361, 30)
    axs[1].set_xticks(x_ticks)
    axs[1].set_xlabel('Preferred orientations')
    axs[1].set_ylabel('APrE/trial')
    axs[1].set_title('Orientation Space Activity Profile')
    axs[1].legend()

    if title:
        fig.suptitle(title, fontsize=16, y=1.05)
        
    if save_fig:
        plt.savefig(save_fig,  bbox_inches='tight')
    plt.show()

def sample_from_dict_with_rng(data_dict, rng):
    """
    Samples a string key and an integer value from a dictionary where 
    keys are strings, and values are numpy arrays of integers, ensuring 
    equal probability for each integer. A numpy random number generator
    object is used for the random sampling.

    Parameters:
        data_dict (dict): A dictionary where keys are strings, and values are numpy arrays of integers.
        rng (numpy.random.Generator): A numpy random number generator object.

    Returns:
        tuple: A randomly sampled (key, integer) pair.
    """
    # Flatten all arrays and keep track of their corresponding keys
    keys = []
    values = []
    for key, array in data_dict.items():
        keys.extend([key] * len(array))  # Repeat the key for each element in the array
        values.extend(array)            # Flatten the array

    # Convert keys and values to numpy arrays
    keys = np.array(keys)
    values = np.array(values)

    # Randomly choose an index using the provided RNG
    random_index = rng.choice(len(values))

    # Return the key and the value at the randomly selected index
    return keys[random_index].item(), values[random_index].item()

def construct_sampled_orientation_representation(rng_seed, trial_samples, trial_bank, admissible_neurons, experiment_data, trial_length, tqdm_disable=True):
    rng = np.random.default_rng(rng_seed) # Very important to use the rng_seed
    trial_matrix = np.zeros((360, trial_length,))
    count_array = np.zeros(360)

    for i in tqdm(range(trial_samples), disable=tqdm_disable):
        experiment_id, trial_id = sample_from_dict_with_rng(trial_bank, rng)
        trial_id = f'trial_{trial_id}'
        experiment_neurons = admissible_neurons[admissible_neurons['Experiment'] == experiment_id]

        for row in experiment_neurons.itertuples():
            cell_index = row.Cell - 1
            orientation_index = int(row.Pref_Orientation_spikes_2) - 1
        
            spike_trace = experiment_data[experiment_id][trial_id][:,cell_index]
            spike_trace = enforce_spike_trace_length(spike_trace, trial_length)

            trial_matrix[orientation_index, :] += spike_trace
            count_array[orientation_index] += 1

    safe_count_array = np.where(count_array == 0, 1, count_array)
    trial_matrix = trial_matrix / safe_count_array[:, np.newaxis]
    trial_matrix = trial_matrix - np.mean(trial_matrix[:,:23], axis=-1)[:, np.newaxis]
    trial_matrix = gaussian_filter1d(trial_matrix, sigma=6, axis=0, mode='wrap')

    return trial_matrix

def orientation_activity_balanced(rng_seed, trial_samples, day, visual_stimulus, response, tables, trial_length):
    relevant_experiment_ids = retrieve_recording_day_experiment_ids(day, tables)
    trial_bank = construct_behavior_trial_bank(
        relevant_experiment_ids, 
        day, 
        visual_stimulus, 
        response, 
        tables
    )
    
    admissible_neurons = retrieve_admissible_neurons(relevant_experiment_ids, tables)
    experiment_data = load_all_experiments_into_memory(trial_bank, tqdm_disable=False)

    activity_matrix = construct_sampled_orientation_representation(
        rng_seed, 
        trial_samples, 
        trial_bank, 
        admissible_neurons, 
        experiment_data, 
        trial_length, 
        tqdm_disable=False
    )
    return activity_matrix

def plot_activity_map(ax, activity_matrix, presented_stimulus, title):
    im = ax.imshow(
        activity_matrix,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest',
        vmin=-0.01,
        vmax=0.04
    )
    ax.axhline(y=presented_stimulus-1, color='tab:red', linestyle='--')  # Stimulus line
    ax.axvline(x=24, color='tab:green', linestyle='--')
    ax.axvline(x=54, color='tab:green', linestyle='--')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Preferred orientations')
    ax.set_title(title)
    y_ticks = np.arange(29, 360, 30)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks + 1)
    return im

def plot_activity_profile(ax, activity_matrix, presented_stimulus, title):
    whenStim = two_second_stimulus_interval(activity_matrix.shape[-1])
    activity_profile = activity_matrix[:, whenStim]
    mean_response = np.nanmean(activity_profile, axis=1)
    std_response = np.nanstd(activity_profile, axis=1)
    ax.fill_between(
        np.arange(1, 361),
        mean_response - std_response,
        mean_response + std_response,
        alpha=0.2,
        label='±1 SD'
    )
    ax.plot(np.arange(1, 361), mean_response, label='Mean Activity', color='tab:blue')
    ax.set_ylim(-0.02, 0.06)
    ax.axvline(x=presented_stimulus, color='tab:red', linestyle='--', label=f'{presented_stimulus}° stimulus')
    ax.axhline(y=0.0, color='tab:grey', linestyle='--')
    x_ticks = np.arange(30, 361, 30)
    ax.set_xticks(x_ticks)
    ax.set_xlabel('Preferred orientations')
    ax.set_ylabel('APrE/trial')
    ax.set_title(title)
    ax.legend()
