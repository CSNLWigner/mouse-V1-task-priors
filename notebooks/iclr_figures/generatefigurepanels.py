"""
Consolidated figure generation for ICLR figure panels.

This script provides:
- A small caching layer under cache/data/ to avoid recomputing data-heavy
  activity maps/profiles from notebooks.
- Shared utilities to compile and cache activity maps/profiles for specific
  days, stimuli, and behavior selections (both or lick-only).
- One master function per original notebook (named after the notebook file)
  that loads pre-collected data from cache and generates the figure panel.

Usage:
- Run the whole suite: python generatefigurepanels.py
- Run a single panel:  python generatefigurepanels.py prior_predictions_3

Notes:
- Figures are saved into results/iclr_figures/.
- Cached data are saved into cache/data/ as .npy when possible; if a dataset
  isn’t a clean ndarray, we fall back to pickle (.pkl).
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import pandas as pd
from typing import Iterable, List, Tuple, Dict, Any

from scipy.stats import norm, ttest_1samp, ttest_ind
from scipy.stats import f as f_dist
from scipy.signal import find_peaks


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Ensure project root is on sys.path so `src` imports work regardless of CWD
_HERE = os.path.abspath(os.path.dirname(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# Results and cache directories
RESULTS_DIR = os.path.join(_PROJ_ROOT, "results", "iclr_figures_bpteam")
CACHE_DIR = os.path.join(_PROJ_ROOT, "cache", "data")
PDF_DIR = "mouse-pdf"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# External project imports (lazy where useful)
from src.data_access import (
    load_tables,
    retrieve_experiment_ids,
    retrieve_recording_day_experiment_ids,
    retrieve_admissible_neurons,
    compute_trial_percentages,
    construct_trial_bank_by_percentage_behavior,
    two_second_stimulus_interval,
)
from src.orientation_representation import (
    load_all_experiments_into_memory,
    construct_orientation_representation_neurons,
    concatenate_orientation_activities,
    compile_activity_profile_neuronstrials,
    compute_orientation_stats,
    compile_std_neuronstrials,
    compute_orientation_ttest,
    bootstrap_peakshift_fromstimulus,
    construct_orientation_representation,
)
from src.orientation_characterization import (
    shift_signal_to_center,
    gain_measure,
    shift_signal_to_orientation,
)
from src.data_compilation import DATASET1, NAIVE
from src.measure_profile import measure_width, closest_peak
from src.directional_statistics import find_local_max_between


# -----------------------------
# Cache utilities
# -----------------------------

def _cache_path(name: str, ext: str = ".npy") -> str:
    safe = name.replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, safe + ext)


def cache_save(name: str, obj: Any) -> str:
    """Save object to cache as .npy when possible; else .pkl.

    Returns the written filepath.
    """
    # Prefer numpy format for arrays
    if isinstance(obj, np.ndarray):
        path = _cache_path(name, ".npy")
        # allow_pickle is False by default; standard ndarrays work fine
        np.save(path, obj)
        return path

    # Try to coerce list of equally-shaped arrays into a 2D ndarray
    if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, np.ndarray) for x in obj):
        try:
            arr = np.stack(obj)
            path = _cache_path(name, ".npy")
            np.save(path, arr)
            return path
        except Exception:
            pass

    # Fallback to pickle for arbitrary Python objects
    path = _cache_path(name, ".pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def cache_load(name: str) -> Any:
    """Load object from cache; tries .npy then .pkl."""
    npy = _cache_path(name, ".npy")
    pkl = _cache_path(name, ".pkl")
    if os.path.exists(npy):
        return np.load(npy, allow_pickle=True)
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"No cache found for {name}")


def cache_exists(name: str) -> bool:
    return os.path.exists(_cache_path(name, ".npy")) or os.path.exists(_cache_path(name, ".pkl"))


# -----------------------------
# Shared data compilation helpers
# -----------------------------

ORIENTATIONS = np.arange(1, 361)  # 1..360 inclusive, consistent with notebooks
TRIAL_LENGTH = 120
STIMULUS_ACTIVITY = (40, 60)

# Orientation display mode for figures:
# - "cut": plot only 0..180 portion of signals (no averaging)
# - "fold": average signal over 0..180 with 181..360 and plot 0..180
ORIENTATION_180_MODE = "cut"  # "cut" or "fold". Change to "fold" to average opposite orientations.

# Behavior aggregation mode for compiling activity:
# - "lick": use lick-only trials [1]
# - "both": use both lick and no-lick trials [0, 1]
BEHAVIOR_MODE = "both"  # "lick" or "both"


# -----------------------------
# Plot style helpers and constants
# -----------------------------

def rgb255(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)

# Primary line colors
COLOR_GO = rgb255(79, 143, 0)         # Go lines
COLOR_NOGO = rgb255(255, 38, 0)       # NoGo lines
COLOR_NAIVE = rgb255(101, 192, 255)   # Naive lines
COLOR_PRIOR = rgb255(121, 121, 121)   # Priors

# Stimulus vertical line colors
COLOR_STIM_GO = rgb255(140, 192, 76)
COLOR_STIM_NOGO = rgb255(255, 136, 116)

# Line widths
LW_PROFILE = 3
LW_STIMULUS = 1
LW_PRIOR_VLINE = 1
LW_PRIOR_PROFILE = 3

# Line styles
LS_SOLID = "-"
LS_DASHED = "--"


def behavior_list_from_mode() -> List[int]:
    """Return the behavior selection list based on BEHAVIOR_MODE.

    lick -> [1]
    both -> [0, 1]
    """
    if BEHAVIOR_MODE == "lick":
        return [1]
    return [0, 1]


def behavior_days_from_mode(n_days: int) -> List[List[int]]:
    """Return a list of identical behavior lists for n_days, based on BEHAVIOR_MODE."""
    beh = behavior_list_from_mode()
    return [beh for _ in range(n_days)]


def behavior_code() -> str:
    """Short code used in cache keys and directories: '1' for lick, '01' for both."""
    return "1" if BEHAVIOR_MODE == "lick" else "01"


def make_activity_cache_key(day: str, stim: int, exp_range: Tuple[float, float], behavior: Iterable[int]) -> str:
    beh_str = "".join(str(b) for b in behavior)
    return f"activity_map_day-{day}_stim-{stim}_range-{exp_range[0]:.2f}-{exp_range[1]:.2f}_beh-{beh_str}"


def collect_activity(
    tables: Tuple[np.ndarray, np.ndarray],
    recording_day: str,
    visual_stimulus: int,
    exp_range: Tuple[float, float],
    behavior: Iterable[int],
    trial_length: int = TRIAL_LENGTH,
    tqdm_disable: bool = False,
    ):
    """
    Returns a pickleable dictionary of orientation keys of
    lists of neurons as arrays of trials x timeframes suitable for caching.
    """
    relevant_experiment_ids = retrieve_recording_day_experiment_ids(recording_day, tables)
    trial_bank = construct_trial_bank_by_percentage_behavior(
        relevant_experiment_ids,
        visual_stimulus,
        recording_day,
        exp_range,
        list(behavior),
        tables,
    )

    # Determine appropriate fit and orientation keys based on available columns (naive vs trained)
    df_celltable, _ = tables
    if "Best_Fit_spikes_2" in df_celltable.columns and "Pref_Orientation_spikes_2" in df_celltable.columns:
        fit_key = "Best_Fit_spikes_2"
        orientation_key = "Pref_Orientation_spikes_2"
    else:
        fit_key = "Best_Fit_spikes_1"
        orientation_key = "Pref_Orientation_spikes_1"

    admissible_neurons = retrieve_admissible_neurons(
        relevant_experiment_ids,
        tables,
        fit_key=fit_key,
        orientation_key=orientation_key,
    )

    # Auto-detect dataset root (NAIVE vs DATASET1) for HDF5 loading
    dataset_root = DATASET1
    try:
        probe_exp = relevant_experiment_ids[0]
        naive_h5 = os.path.join(NAIVE["CellTrialTable_csv_path"], probe_exp[:3], f"{probe_exp}.h5")
        trained_h5 = os.path.join(DATASET1["CellTrialTable_csv_path"], probe_exp[:3], f"{probe_exp}.h5")
        if os.path.exists(naive_h5):
            dataset_root = NAIVE
        elif os.path.exists(trained_h5):
            dataset_root = DATASET1
    except Exception:
        dataset_root = DATASET1

    experiment_data = load_all_experiments_into_memory(trial_bank, DATASET=dataset_root, tqdm_disable=tqdm_disable)


    orientation_dict = construct_orientation_representation_neurons(
        admissible_neurons, experiment_data, trial_length, tqdm_disable=True)


    return orientation_dict



def compile_activity_map(
    tables: Tuple[np.ndarray, np.ndarray],
    recording_day: str,
    visual_stimulus: int,
    exp_range: Tuple[float, float],
    behavior: Iterable[int],
    trial_length: int = TRIAL_LENGTH,
    tqdm_disable: bool = False,
):
    """Replicates the notebook's activity_map_compilation and returns the trial x time matrix across orientations.

    Returns a numpy array suitable for caching.
    """
    relevant_experiment_ids = retrieve_recording_day_experiment_ids(recording_day, tables)
    trial_bank = construct_trial_bank_by_percentage_behavior(
        relevant_experiment_ids,
        visual_stimulus,
        recording_day,
        exp_range,
        list(behavior),
        tables,
    )

    # Determine appropriate fit and orientation keys based on available columns (naive vs trained)
    df_celltable, _ = tables
    if "Best_Fit_spikes_2" in df_celltable.columns and "Pref_Orientation_spikes_2" in df_celltable.columns:
        fit_key = "Best_Fit_spikes_2"
        orientation_key = "Pref_Orientation_spikes_2"
    else:
        fit_key = "Best_Fit_spikes_1"
        orientation_key = "Pref_Orientation_spikes_1"

    admissible_neurons = retrieve_admissible_neurons(
        relevant_experiment_ids,
        tables,
        fit_key=fit_key,
        orientation_key=orientation_key,
    )

    # Auto-detect dataset root (NAIVE vs DATASET1) for HDF5 loading
    dataset_root = DATASET1
    try:
        probe_exp = relevant_experiment_ids[0]
        naive_h5 = os.path.join(NAIVE["CellTrialTable_csv_path"], probe_exp[:3], f"{probe_exp}.h5")
        trained_h5 = os.path.join(DATASET1["CellTrialTable_csv_path"], probe_exp[:3], f"{probe_exp}.h5")
        if os.path.exists(naive_h5):
            dataset_root = NAIVE
        elif os.path.exists(trained_h5):
            dataset_root = DATASET1
    except Exception:
        dataset_root = DATASET1

    experiment_data = load_all_experiments_into_memory(trial_bank, DATASET=dataset_root, tqdm_disable=tqdm_disable)

    activity_map = construct_orientation_representation(
        admissible_neurons,
        experiment_data,
        trial_length,
    )
    return activity_map


def compile_activity_profile(activity_map: np.ndarray, stimulus_activity: Tuple[int, int] = STIMULUS_ACTIVITY) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std over stimulus window for each orientation (row)."""
    ap_mean = np.nanmean(activity_map[:, stimulus_activity[0] : stimulus_activity[1]], axis=-1)
    ap_std = np.nanstd(activity_map[:, stimulus_activity[0] : stimulus_activity[1]], axis=-1)
    return ap_mean, ap_std


def get_activity_neuronstrials_cached(
    tables: Tuple[np.ndarray, np.ndarray],
    day: str,
    stim: int,
    exp_range: Tuple[float, float],
    behavior: Iterable[int],
    dataset_tag: str | None = None):
    """Get or compute+cache the per-orientation neuron-trial activity dictionary.
    Returns a dictionary mapping orientation -> list of neurons as arrays of trials x timeframes.
    """

    key_base = make_activity_cache_key(day, stim, exp_range, behavior)
    if dataset_tag:
        key_base = f"{dataset_tag}_" + key_base
    key_neuronstrials = key_base + "_neuronstrials"

    if cache_exists(key_neuronstrials):
        return cache_load(key_neuronstrials)

    # Compute if not cached

    activity_neuronstrials = collect_activity(tables,day,stim,exp_range,behavior)
    cache_save(key_neuronstrials, activity_neuronstrials)

    return activity_neuronstrials


def get_activity_mapandprofile_cached(
    tables: Tuple[np.ndarray, np.ndarray],
    day: str,
    stim: int,
    exp_range: Tuple[float, float],
    behavior: Iterable[int],
    stimulus_activity: Tuple[int, int] = STIMULUS_ACTIVITY,
    dataset_tag: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or compute+cache the activity map and profile (mean, std)."""
    key_base = make_activity_cache_key(day, stim, exp_range, behavior)
    if dataset_tag:
        key_base = f"{dataset_tag}_" + key_base
    key_map = key_base + "_map"
    key_mean = key_base + "_profile_mean"
    key_std = key_base + "_profile_std"

    if cache_exists(key_map) and cache_exists(key_mean) and cache_exists(key_std):
        return cache_load(key_map), cache_load(key_mean), cache_load(key_std)
    # Compute if not cached, but check if map is cached already
    if cache_exists(key_map):
        amap = cache_load(key_map)
    else:
        amap = compile_activity_map(tables, day, stim, exp_range, behavior)
        cache_save(key_map, amap)
    ap_mean, ap_std = compile_activity_profile(amap, stimulus_activity=stimulus_activity)
    cache_save(key_mean, ap_mean)
    cache_save(key_std, ap_std)
    return amap, ap_mean, ap_std


def get_activity_profile_cached(
    tables: Tuple[np.ndarray, np.ndarray],
    day: str,
    stim: int,
    exp_range: Tuple[float, float],
    behavior: Iterable[int],
    stimulus_activity: Tuple[int, int] = STIMULUS_ACTIVITY,
    dataset_tag: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    amap, ap_mean, ap_std = get_activity_mapandprofile_cached(
        tables,
        day,
        stim,
        exp_range,
        behavior,
        stimulus_activity,
        dataset_tag,
    )
    return ap_mean, ap_std



def load_naive_potential_likelihood() -> np.ndarray:
    """Load potential likelihood from data. Cached as-is via numpy file.

    File expected at data/Naive_numpy_arrays/potential_likelihood.npy
    """
    path = os.path.join(_PROJ_ROOT, "data", "Naive_numpy_arrays", "potential_likelihood.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    # Also mirror into cache for uniformity
    cache_key = "naive_potential_likelihood"
    if cache_exists(cache_key):
        return cache_load(cache_key)
    arr = np.load(path)
    cache_save(cache_key, arr)
    return arr


def calculate_autocorrelation(df_subset):
    """Calculate autocorrelation function for activity values."""
    activities = df_subset['activity'].values
    
    # Remove NaN values
    activities = activities[~np.isnan(activities)]
    n = len(activities)
    
    print(f"  Autocorr calculation: {n} valid points after removing NaN")
    
    if n < 2:
        print(f"  Warning: Only {n} valid points, cannot calculate autocorrelation")
        return np.array([]), np.array([])
    
    # Check if variance is too small (essentially constant signal)
    var = np.var(activities)
    print(f"  Variance: {var:.6f}")
    if var < 1e-10:
        print(f"  Warning: Very low variance ({var}) in activity data, returning empty autocorrelation")
        return np.array([]), np.array([])
    
    # Normalize the data (subtract mean)
    activities_normalized = activities - np.mean(activities)
    
    # Calculate autocorrelation for different lags
    max_lag = min(n - 1, 50)  # Limit to reasonable lag
    lags = np.arange(0, max_lag + 1)
    autocorr = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        if lag == 0:
            autocorr[i] = 1.0
        else:
            # More robust calculation avoiding division issues
            if n - lag > 0:
                correlation = np.sum(activities_normalized[:-lag] * activities_normalized[lag:]) / (n - lag)
                autocorr[i] = correlation / var
            else:
                autocorr[i] = np.nan
    
    # Remove trailing NaN values
    valid_mask = ~np.isnan(autocorr)
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    return lags[valid_mask], autocorr[valid_mask]






def frameautocorrelation():
    # Load D1 raw activities for each animal and create frame-level DataFrame
    # DataFrame structure:
    #   - mouse: Mouse ID (first 3 chars of Experiment)
    #   - Experiment: Full experiment ID
    #   - Trial: Trial number
    #   - trial_type: "go" or "nogo" (corresponding to 45° Go and 90° NoGo stimuli)
    #   - Animal: "Naive" or "Trained" (dataset condition)
    #   - frame: Frame index within trial (0-based)
    #   - activity: Population activity (sum of spike traces across admissible neurons)
    
    # Prepare data collection lists
    frames_data = []


    if not cache_exists("df_frames_activity_D1"):
    
        # Process both Naive and Trained datasets
        for dataset_name, dataset_config in [("Naive", NAIVE), ("Trained", DATASET1)]:
            # Cache key for this dataset's tables
            cache_key = f"tables_{dataset_name.lower()}"
            if cache_exists(cache_key):
                tables = cache_load(cache_key)
            else:
                tables = load_tables(DATASET=dataset_config)
                cache_save(cache_key, tables)
            df_celltable, df_trialtable = tables

            # Get D1 experiment IDs for this dataset
            relevant_experiment_ids = retrieve_recording_day_experiment_ids("D1", tables)
            
            # Auto-detect appropriate fit/orientation keys
            if "Best_Fit_spikes_2" in df_celltable.columns and "Pref_Orientation_spikes_2" in df_celltable.columns:
                fit_key = "Best_Fit_spikes_2"
                orientation_key = "Pref_Orientation_spikes_2"
            else:
                fit_key = "Best_Fit_spikes_1"
                orientation_key = "Pref_Orientation_spikes_1"
            
            # Process both Go (45°) and NoGo (90°) trials
            for trial_type, visual_stim in [("go", 45), ("nogo", 90)]:
                # Build trial bank for this stimulus
                trial_bank = construct_trial_bank_by_percentage_behavior(
                    relevant_experiment_ids,
                    visual_stim,
                    "D1",
                    (0.0, 1.0),  # Full session
                    [0, 1],  # Both lick and no-lick
                    tables,
                )
                
                # Get admissible neurons
                admissible_neurons = retrieve_admissible_neurons(
                    relevant_experiment_ids,
                    tables,
                    fit_key=fit_key,
                    orientation_key=orientation_key,
                )
                
                # Load raw experiment data from HDF5 files
                experiment_data = load_all_experiments_into_memory(
                    trial_bank, 
                    DATASET=dataset_config, 
                    tqdm_disable=True
                )
                
                # Extract frame-level activity for each animal/trial
                for experiment_id in trial_bank.keys():
                    mouse_id = experiment_id[:3]  # Extract mouse ID from experiment ID
                    experiment_neurons = admissible_neurons[admissible_neurons['Experiment'] == experiment_id]
                    
                    # Process each trial for this experiment
                    for trial_id in trial_bank[experiment_id]:
                        trial_key = f'trial_{trial_id}'
                        if trial_key not in experiment_data[experiment_id]:
                            continue
                        
                        # Get raw spike traces for this trial: shape (frames, cells)
                        trial_data = experiment_data[experiment_id][trial_key]
                        n_frames = trial_data.shape[0]
                        
                        # Sum across all admissible neurons for this experiment to get population activity
                        cell_indices = experiment_neurons['Cell'].values - 1  # Convert to 0-based
                        if len(cell_indices) > 0:
                            population_activity = trial_data[:, cell_indices].sum(axis=1)
                            
                            # Add frame-level records
                            for frame_idx in range(n_frames):
                                frames_data.append({
                                    'mouse': mouse_id,
                                    'Experiment': experiment_id,
                                    'Trial': trial_id,
                                    'trial_type': trial_type,
                                    'Animal': dataset_name,
                                    'frame': frame_idx,
                                    'activity': population_activity[frame_idx]
                                })
        
        # Create DataFrame with all frame-level activity data
        df_frames_activity = pd.DataFrame(frames_data)
        cache_save("df_frames_activity_D1", df_frames_activity)
    else:
        print("Loading cached df_frames_activity_D1")
        df_frames_activity = cache_load("df_frames_activity_D1")

    print(f"Total frames in df_frames_activity: {len(df_frames_activity)}")
    print(f"Columns: {df_frames_activity.columns.tolist()}")
    print(f"Unique animals: {df_frames_activity['Animal'].unique()}")
    print(f"Value counts by Animal:\n{df_frames_activity['Animal'].value_counts()}")
    
    # Separate data by Animal condition
    df_naive = df_frames_activity[df_frames_activity['Animal'] == 'Naive']
    df_training = df_frames_activity[df_frames_activity['Animal'] == 'Trained']
    
    print(f"\nNaive data points: {len(df_naive)}")
    print(f"  Non-NaN: {df_naive['activity'].notna().sum()}")
    print(f"  NaN count: {df_naive['activity'].isna().sum()}")
    print(f"  Mean activity (excluding NaN): {df_naive['activity'].mean():.4f}")
    print(f"  Std: {df_naive['activity'].std():.4f}")
    print(f"  Min: {df_naive['activity'].min():.4f}, Max: {df_naive['activity'].max():.4f}")
    
    print(f"\nTrained data points: {len(df_training)}")
    print(f"  Non-NaN: {df_training['activity'].notna().sum()}")
    print(f"  NaN count: {df_training['activity'].isna().sum()}")
    if len(df_training) > 0 and df_training['activity'].notna().sum() > 0:
        print(f"  Mean activity (excluding NaN): {df_training['activity'].mean():.4f}")
        print(f"  Std: {df_training['activity'].std():.4f}")
        print(f"  Min: {df_training['activity'].min():.4f}, Max: {df_training['activity'].max():.4f}")
        print(f"  Sample activities: {df_training['activity'].head(20).values}")
    else:
        print("  WARNING: No valid data in Trained!")
    
    # Calculate autocorrelations
    lags_naive, autocorr_naive = calculate_autocorrelation(df_naive)
    lags_training, autocorr_training = calculate_autocorrelation(df_training)
    
    print(f"Naive autocorr length: {len(autocorr_naive)}, Training autocorr length: {len(autocorr_training)}")
    # Plot autocorrelation functions
    plt.figure(figsize=(8, 7))
    plt.subplots_adjust(left=0.20, bottom=0.15)
    
    if len(lags_naive) > 0:
        plt.plot(lags_naive, autocorr_naive, color=COLOR_NAIVE, lw=LW_PROFILE, label='Naive', marker='o', markersize=4)
    if len(lags_training) > 0:
        plt.plot(lags_training, autocorr_training, color='darkorange', lw=LW_PROFILE, label='Training', marker='o', markersize=4)
    
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Lag (frames)')
    plt.ylabel('Autocorrelation')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.3)
    sns.despine(top=True, right=True)
    
    plt.savefig('./results/iclr_figures_bpteam/width_characterization_mouse_autocorrelation.png')
    # plt.savefig('./results/iclr_figures_bpteam/mouse-pdf/Fig2G_width_characterization_mouse_autocorrelation.pdf')
    plt.close()

















# -----------------------------
# Figure: orientation_response_profiles_3E.ipynb
# -----------------------------

def orientation_response_profiles() -> str:
    """Generate NoGo activity profiles across D2–D6 and save figure.

    Returns the saved figure path.
    """
    tables = load_tables()
    tables_naive = load_tables(DATASET=NAIVE)

    # Day specifications (from notebook)
    days = ["D1", "D2", "D3", "D4", "D5", "D6"]
    stim_means = [135, 90, 75, 70, 65, 60]
    exp_ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)] # here even D2 is full session
    # Behavior selection per day based on global mode (uniform across days)
    behaviors = behavior_days_from_mode(len(days))
    binwidth = 6  # for peak neighbourhood statistics, including peak bootstrap
    num_bootstraps = 200 # 200
    neurons_per_orientation = 5 #5

    activities_neuronstrials: List[np.ndarray] = []
    activities_naive_neuronstrials: List[np.ndarray] = []
    profiles: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    profiles_naive: List[np.ndarray] = []
    stds_naive: List[np.ndarray] = []
    for day, stim, rng, beh in zip(days, stim_means, exp_ranges, behaviors):

        ap_mean_naive, ap_std_naive = get_activity_profile_cached(tables_naive, day, stim, rng, beh, dataset_tag="naive")
        profiles_naive.append(ap_mean_naive)
        stds_naive.append(ap_std_naive)

        ap_mean, ap_std = get_activity_profile_cached(tables, day, stim, rng, beh)
        profiles.append(ap_mean)
        stds.append(ap_std)
        

        # activity_neuronstrials = get_activity_neuronstrials_cached(tables, day, stim, rng, beh)
        # activities_neuronstrials.append(activity_neuronstrials)

        # activity_naive_neuronstrials = get_activity_neuronstrials_cached(tables_naive, day, stim, rng, beh, dataset_tag="naive")
        # activities_naive_neuronstrials.append(activity_naive_neuronstrials)

    return

    # print(tables)   
    # print(activities_neuronstrials[0].keys())


    fig, axs = plt.subplots(1, 5, figsize=(12, 2.4), sharey=True)

    for ax, activity_neuronstrials,profile, std, activity_naive_neuronstrials, profile_naive, std_naive, day, stim in \
        zip(axs, activities_neuronstrials, profiles, stds, activities_naive_neuronstrials, profiles_naive, stds_naive, days, stim_means):


        sems_trained_nt = compile_std_neuronstrials(activity_neuronstrials, ORIENTATIONS, binwidth=binwidth)[:180] * 2
        sems_naive_nt = compile_std_neuronstrials(activity_naive_neuronstrials, ORIENTATIONS, binwidth=binwidth)[:180] * 2


        # peak analysis
        # Find local maxima indices
        left_max_deg, left_max_mean, left_max_std = find_local_max_between(profile, std, 45, stim-5)
        right_max_deg, right_max_mean, right_max_std = find_local_max_between(profile, std, stim+5, 135)
        stim_mean, stim_std = profile[stim-1], std[stim-1]

        # Find local minimum around stimulus (NoGo shoulder dip)
        stim_window = 5  # degrees
        stim_min_deg = stim + np.argmin(profile[max(0, stim-stim_window-1):min(360, stim+stim_window)]) - stim_window
        stim_min_mean, stim_min_std = profile[stim_min_deg-1], std[stim_min_deg-1]
        
        # Convert degrees to 0-based indices for activities_neuronstrials dictionary
        left_max_idx = left_max_deg  # activities_neuronstrials uses 1-360 as keys
        right_max_idx = right_max_deg
        stim_idx = stim
        
        # Compute neurons based statistics at each critical orientation and its surroundings
        left_mean_nt, left_std_nt, left_count_nt = compute_orientation_stats(activity_neuronstrials, left_max_idx, binwidth=binwidth)
        right_mean_nt, right_std_nt, right_count_nt = compute_orientation_stats(activity_neuronstrials, right_max_idx, binwidth=binwidth)
        stim_mean_nt, stim_std_nt, stim_count_nt = compute_orientation_stats(activity_neuronstrials, stim_idx, binwidth=binwidth)

        # Compute t-tests comparing peaks to stimulus activity (one-tailed: peaks > stimulus)
        t_left_vs_stim, p_left_vs_stim = compute_orientation_ttest(
            activity_neuronstrials, orientation_idx1=left_max_idx, orientation_idx2=stim_idx, binwidth=binwidth, one_tailed=True
        )
        t_right_vs_stim, p_right_vs_stim = compute_orientation_ttest(
            activity_neuronstrials, orientation_idx1=right_max_idx, orientation_idx2=stim_idx, binwidth=binwidth, one_tailed=True
        )


        print(f"\n{day}: Trained - \nBimodal left max at {left_max_deg}° "
              f"(profile: {left_max_mean:.4f}±{left_max_std:.4f}, "
              f"neuronstrials: {left_mean_nt:.4f}±{left_std_nt/np.sqrt(left_count_nt):.4f}, n={left_count_nt:.1f}), "
              f"\nStim at {stim}° "
              f"(profile: {stim_mean:.4f}±{stim_std:.4f}, "
              f"neuronstrials: {stim_mean_nt:.4f}±{stim_std_nt/np.sqrt(stim_count_nt):.4f}, n={stim_count_nt:.1f}), "
              f"\nBimodal right max at {right_max_deg}° "
              f"(profile: {right_max_mean:.4f}±{right_max_std:.4f}, "
              f"neuronstrials: {right_mean_nt:.4f}±{right_std_nt/np.sqrt(right_count_nt):.4f}, n={right_count_nt:.1f})")
        print(f"  t-test left peak vs stim: t={t_left_vs_stim:.4f}, p={p_left_vs_stim:.6f}")
        print(f"  t-test right peak vs stim: t={t_right_vs_stim:.4f}, p={p_right_vs_stim:.6f}")



        # Add two-end arrows
        arrowcolor = 'black'
        corrector = 0.001
        # at peaks for D4 and D5
        if day in ["D4", "D5"]:
            # Left peak arrow
            ax.annotate('', xy=(left_max_deg, left_max_mean+corrector), xytext=(left_max_deg, stim_min_mean-corrector),
                arrowprops=dict(arrowstyle='<->', color=arrowcolor, lw=0.5))
            ax.text(left_max_deg-2, (left_max_mean+stim_min_mean)/2, '∗∗∗∗', ha='right', va='center', fontsize=3, color=arrowcolor, rotation=90)

            # Right peak arrow
            ax.annotate('', xy=(right_max_deg, right_max_mean+corrector), xytext=(right_max_deg, stim_min_mean-corrector),
                arrowprops=dict(arrowstyle='<->', color=arrowcolor, lw=0.5))
            ax.text(right_max_deg-2, (right_max_mean+stim_min_mean)/2, '∗∗∗∗', ha='right', va='center', fontsize=3, color=arrowcolor, rotation=90)
        


        # Add naive profile overlay
        x_naive, y_naive, ystd_naive = ORIENTATIONS[:180], profile_naive[:180], sems_naive_nt[:180]   # std_naive[:180]
        ax.plot(x_naive, y_naive, color=COLOR_NAIVE, lw=LW_PROFILE, alpha=0.6, label="Naive")
        ax.fill_between(x_naive, y_naive - ystd_naive, y_naive + ystd_naive, color=COLOR_NAIVE, alpha=0.2)
        # Trained profile
        x, y, ystd = ORIENTATIONS[:180], profile[:180], sems_trained_nt[:180]    # std[:180]
        ax.plot(x, y, color=COLOR_NOGO, lw=LW_PROFILE, label="Trained")
        ax.fill_between(x, y - ystd, y + ystd, color=COLOR_NOGO, alpha=0.3)
        if stim is not None:
            ax.axvline(x=stim, color=COLOR_STIM_NOGO, lw=LW_STIMULUS, ls=LS_DASHED, label=None) # f"{day}, {stim}°"
            



        peakshift_distribution = bootstrap_peakshift_fromstimulus(activity_neuronstrials, stim, num_bootstraps=num_bootstraps, binwidth=binwidth, neurons_per_orientation=neurons_per_orientation)
        peakshift_distribution_naive = bootstrap_peakshift_fromstimulus(activity_naive_neuronstrials, stim, num_bootstraps=num_bootstraps, binwidth=binwidth, neurons_per_orientation=neurons_per_orientation)
        
        print(f"Peak shift statistics:")
        print(f"  Bootstrap peak shift from stimulus (trained): "
              f"{np.mean(peakshift_distribution):.2f}° ± {np.std(peakshift_distribution):.2f}° (mean ± std over bootstraps, n={len(peakshift_distribution)})")
        t_stat, p_value = ttest_1samp(peakshift_distribution, popmean=0)
        print(f"  t-test absolute diff.: |stimulus-peak| (H0: mean=0): t={t_stat:.4f}, p={p_value:.16f}")

        print(f"  Bootstrap peak shift from stimulus (naive): "
              f"{np.mean(peakshift_distribution_naive):.2f}° ± {np.std(peakshift_distribution_naive):.2f}° (mean ± std over bootstraps, n={len(peakshift_distribution_naive)})")
        t_stat_naive, p_value_naive = ttest_1samp(peakshift_distribution_naive, popmean=0)
        print(f"  t-test absolute diff. naive: |stimulus-peak| (H0: mean=0): t={t_stat_naive:.4f}, p={p_value_naive:.16f}")


        # Compare trained vs naive peak shift distributions
        t_stat_diff, p_value_diff = ttest_ind(peakshift_distribution, peakshift_distribution_naive)
        print(f"  t-test trained vs naive peak shift: t={t_stat_diff:.4f}, p={p_value_diff:.16f}, n,n={len(peakshift_distribution)},{len(peakshift_distribution_naive)}")


        arrowcorrection = 3 # degrees offset for arrow heads
        # asterices = dict(D3='∗∗', D4='∗∗∗∗', D5='∗∗∗∗', D6='∗')
        asterices = ['∗', '∗∗', '∗∗∗', '∗∗∗∗']
        if day in ["D3", "D4", "D5", "D6"]:
            # Add arrows showing peak shift for D4, D5, D6
            # Arrow colors match the plot colors (naive: COLOR_NAIVE, trained: COLOR_NOGO)
            # Both arrows point away from stimulus (left=dashed, right=solid)

            mean_shift_naive = np.mean(peakshift_distribution_naive)
            mean_shift_trained = np.mean(peakshift_distribution)

            # Y position for arrows (slightly above the profile for visibility)
            arrow_y_naive = 0.022

            # Naive arrows (left dashed, right solid)
            ax.annotate('', 
                        xy=(stim - mean_shift_naive - arrowcorrection, arrow_y_naive), 
                        xytext=(stim, arrow_y_naive),
                        arrowprops=dict(arrowstyle='->', color=COLOR_NAIVE, lw=0.5, linestyle='--', alpha=0.3))
            ax.annotate('', 
                        xy=(stim + mean_shift_naive + arrowcorrection, arrow_y_naive), 
                        xytext=(stim, arrow_y_naive),
                        arrowprops=dict(arrowstyle='->', color=COLOR_NAIVE, lw=0.5, linestyle='-'))

            # Trained arrows (left dashed, right solid)
            arrow_y_trained = 0.020
            ax.annotate('', 
                        xy=(stim - mean_shift_trained-arrowcorrection, arrow_y_trained), 
                        xytext=(stim, arrow_y_trained),
                        arrowprops=dict(arrowstyle='->', color=COLOR_NOGO, lw=0.5, linestyle='--', alpha=0.3))
            ax.annotate('', 
                        xy=(stim + mean_shift_trained+arrowcorrection, arrow_y_trained), 
                        xytext=(stim, arrow_y_trained),
                        arrowprops=dict(arrowstyle='->', color=COLOR_NOGO, lw=0.5, linestyle='-'))

            # Add horizontal bracket between naive and trained arrows
            bracket_y_offset = 0.001 #arrow_y_naive + 0.005  # Small offset above arrows
            bracket_x_left = stim + mean_shift_naive
            bracket_x_right = stim + mean_shift_trained

            # Draw bracket: left vertical, top horizontal, right vertical
            ax.plot([bracket_x_left, bracket_x_left], [arrow_y_naive + bracket_y_offset, arrow_y_naive + 2*bracket_y_offset], 
                color='black', lw=0.5, linestyle='-')
            ax.plot([bracket_x_left, bracket_x_right], [arrow_y_naive + 2*bracket_y_offset, arrow_y_naive + 2*bracket_y_offset], 
                color='black', lw=0.5, linestyle='-')
            ax.plot([bracket_x_right, bracket_x_right], [arrow_y_naive + 2*bracket_y_offset, arrow_y_naive + bracket_y_offset], 
                color='black', lw=0.5, linestyle='-')

            # Determine number of asterisks based on p-value
            if p_value_diff < 0.0001:
                asterix = asterices[3]  # ∗∗∗∗
            elif p_value_diff < 0.001:
                asterix = asterices[2]  # ∗∗∗
            elif p_value_diff < 0.01:
                asterix = asterices[1]  # ∗∗
            elif p_value_diff < 0.05:
                asterix = asterices[0]  # ∗
            else:
                asterix = 'n.s.'  # No significance
            ax.text((bracket_x_left+bracket_x_right)/2, arrow_y_naive + 2*bracket_y_offset + 0.001, asterix, ha='center', va='bottom', fontsize=3, color='black')



        ax.set_title(f"{day}, stimulus: {stim}°", fontsize=12)
        ax.set_xlim([0, 180])
        ax.set_xticks(np.arange(0, 181, 45))
        ax.set_xlabel("Orientation (°)")
        # ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # if stim is not None:
        #     ax.legend(frameon=False, loc='upper right')
    
    axs[0].set_ylabel("Population Activity")
    axs[0].legend(frameon=False, loc='lower center', fontsize=8)
    plt.tight_layout()
    # plt.suptitle("NoGo activity profiles across experimental days", fontsize=16, y=1.05)

    fn = "nogo_activity_profiles"
    plt.savefig(os.path.join(RESULTS_DIR, fn + ".png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS_DIR, PDF_DIR, "Fig3A_" + fn + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return fn




def width_characterization() -> str:

    pal = sns.color_palette([COLOR_NAIVE, 'darkorange'])

    # model 
    # Model produced widths 
    if False:  # this is simple 2dot plot not used anymore
        naive = 28.8
        task = 21.4
        plt.figure(figsize=(2.5, 3.5))
        plt.tight_layout()
        plt.subplots_adjust(left=0.20, bottom=0.10)
        plt.scatter(x=1, y=naive, s=100, color=pal[0], marker='o', zorder=5, linewidth=2)
        plt.scatter(x=2, y=task, s=100, color=pal[1], marker='o', zorder=5, linewidth=2)
        plt.xlim(0.75, 2.25)
        plt.ylim(10, 40)
        plt.xticks([1, 2], ['Naive', 'Task'])
        plt.yticks(np.arange(10, 41, 10))
        plt.ylabel('Width')
        sns.despine(top=True, right=True)
        plt.savefig('./results/iclr_figures_bpteam/width_characterization_model_dots.png')
        plt.savefig('./results/iclr_figures_bpteam/model-pdf/Fig2C_width_characterization_model_dots.pdf')


    plt.figure(figsize=(2.5, 3.5))
    plt.subplots_adjust(left=0.25, bottom=0.15)
    df_frames_widths = pd.read_csv('./cache/widths_model.csv')
    pal = sns.color_palette([COLOR_NAIVE, 'darkorange'])
    sns.boxplot(x="Exp", hue="Exp", y="Width", palette=pal, data=df_frames_widths, width=0.6)
    # Statistical annotation (optional)
    try:
        from statannotations.Annotator import Annotator
        annot = Annotator(plt.gca(), [("Naive", "Task")], data=df_frames_widths, x="Exp", y="Width")
        annot.configure(test="Mann-Whitney", text_format="star", loc="outside", verbose=0)
        annot.apply_test()
        annot.annotate()
    except Exception as e:
        print(f"statannotations unavailable or failed ({e}); skipping significance annotation.")
    plt.xlabel("")
    sns.despine(top=True, right=True)
    plt.savefig('./results/iclr_figures_bpteam/width_characterization_model.png')
    plt.savefig('./results/iclr_figures_bpteam/model-pdf/Fig2C_width_characterization_model.pdf')



    # animal - Call frameautocorrelation which creates df_frames_activity internally
    # frameautocorrelation()  # Note: df_frames_activity is created inside the function
    nindepdatapoints_by_autocorr = 5  # from autocorr plot
    binsize_indepdatapoints_by_autocorr = 30 / nindepdatapoints_by_autocorr  # from autocorr plot


    plt.figure(figsize=(2.5, 3.5))
    plt.subplots_adjust(left=0.25, bottom=0.15)
    df_frames_widths = pd.read_csv('./data/df_frames_widths.csv')
    # Bin frames by autocorrelation-based independent data points
    df_frames_widths['Frame_bin'] = (df_frames_widths['Frame'] // binsize_indepdatapoints_by_autocorr).astype(int)
    
    # Group by Frame_bin AND all other identifying columns, averaging only Width
    grouping_cols = [col for col in df_frames_widths.columns if col not in ['Frame', 'Width']]
    df_frames_widths = df_frames_widths.groupby(grouping_cols, as_index=False).agg({'Width': 'mean'})
    df = df_frames_widths[df_frames_widths['Experimental condition'] == 'Task']
    df = df[df['Stimulus class'] == 'Go']
    pal = sns.color_palette([COLOR_NAIVE, 'darkorange'])
    sns.boxplot(x="Animal", hue="Animal", y="Width", palette=pal, data=df, width=0.6)
    print(df)

    # Statistical annotation (optional)
    try:
        from statannotations.Annotator import Annotator
        annot = Annotator(plt.gca(), [("Naive", "Training")], data=df, x="Animal", y="Width")
        annot.configure(test="Mann-Whitney", text_format="star", loc="outside", verbose=0)
        annot.apply_test()
        annot.annotate()
    except Exception as e:
        print(f"statannotations unavailable or failed ({e}); skipping significance annotation.")
    plt.xlabel("")
    # plt.ylabel("Width")

    sns.despine(top=True, right=True)
    plt.savefig('./results/iclr_figures_bpteam/width_characterization_mouse.png')
    plt.savefig('./results/iclr_figures_bpteam/mouse-pdf/Fig2G_width_characterization_mouse.pdf')








def baseline_comparisons() -> None:
    """Task baseline comparisons between Naive and Trained using cached processed arrays.

    Replicates baseline_comparisons.ipynb: loads trial matrices, extracts
    stimulus window for selected orientations, computes baselines, and plots a
    boxplot with optional Mann-Whitney annotation.
    """
    # Orientation indices as in notebook
    orientation_range = (
        # list(range(0, 31)) + list(range(149, 211)) + list(range(329, 360))                    # 0-30, 149-210, 329-359 all above 180
        list(range(0, 31)) + list(range(149, 180))  # 0-30, 149-179                         (0-180 only)
    )

    # Cache keys for processed matrices (stim-window sliced)
    naive_key = "baseline_naive_task_trial_matrices_proc"
    trained_key = "baseline_trained_task_trial_matrices_proc"

    # Load and process Naive matrices
    if cache_exists(naive_key):
        naive_task_trial_matrices = cache_load(naive_key)
    else:
        naive_path = os.path.join(_PROJ_ROOT, "data", "Naive_numpy_arrays", "task_aggregated_matrices.npy")
        if not os.path.exists(naive_path):
            raise FileNotFoundError(f"Missing data file: {naive_path}")
        naive_task_trial_matrices = np.load(naive_path)
        whenStim = two_second_stimulus_interval(naive_task_trial_matrices.shape[-1])
        naive_task_trial_matrices = naive_task_trial_matrices[:, orientation_range, :]
        naive_task_trial_matrices = naive_task_trial_matrices[:, :, whenStim]
        cache_save(naive_key, naive_task_trial_matrices)

    # Load and process Trained matrices
    if cache_exists(trained_key):
        trained_task_trial_matrices = cache_load(trained_key)
    else:
        trained_path = os.path.join(_PROJ_ROOT, "data", "DATASET1_numpy_arrays", "task_aggregated_matrices.npy")
        if not os.path.exists(trained_path):
            raise FileNotFoundError(f"Missing data file: {trained_path}")
        trained_task_trial_matrices = np.load(trained_path)
        whenStim = two_second_stimulus_interval(trained_task_trial_matrices.shape[-1])
        trained_task_trial_matrices = trained_task_trial_matrices[:, orientation_range, :]
        trained_task_trial_matrices = trained_task_trial_matrices[:, :, whenStim]
        cache_save(trained_key, trained_task_trial_matrices)

    # Compute baselines
    naive_task_trial_baselines = np.mean(naive_task_trial_matrices, axis=(1, 2))
    trained_task_trial_baselines = np.mean(trained_task_trial_matrices, axis=(1, 2))

    # Plot with seaborn; annotate if statannotations is available
    sns.set(style="whitegrid", context="paper", font_scale=1.2)

    data = pd.DataFrame({
        "Baseline": np.concatenate([naive_task_trial_baselines, trained_task_trial_baselines]),
        "Group": ["Naive"] * len(naive_task_trial_baselines) + ["Trained"] * len(trained_task_trial_baselines)
    })

    fig, ax = plt.subplots(figsize=(2.5, 3.5))
    pal = sns.color_palette([COLOR_NAIVE, 'darkorange'])
    sns.boxplot(x="Group", y="Baseline", data=data, hue="Group", palette=pal, width=0.6, ax=ax)

    # Statistical annotation (optional)
    try:
        from statannotations.Annotator import Annotator
        annot = Annotator(ax, [("Naive", "Trained")], data=data, x="Group", y="Baseline")
        annot.configure(test="Mann-Whitney", text_format="star", loc="outside", verbose=0)
        annot.apply_test()
        annot.annotate()
    except Exception as e:
        print(f"statannotations unavailable or failed ({e}); skipping significance annotation.")

    ax.set_xlabel("")
    ax.set_ylabel("Mean Baseline Activity", labelpad=6)
    ax.set_title("")
    sns.despine(trim=True)
    plt.tight_layout()

    fn = "baseline_comparisons"
    plt.savefig(os.path.join(RESULTS_DIR, fn + ".png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS_DIR, PDF_DIR, "Fig2H_" + fn + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return fn







def naive_vs_trained_profiles() -> str:
    """Compare Naive vs Trained Go D1 orientation profiles in 2D (orientation on y-axis).

    Replicates naive_vs_trained_profiles_2D.ipynb with dataset-aware caching.
    """
    # Colors and widths via global constants

    # Tables per dataset
    tables_naive = load_tables(DATASET=NAIVE)
    tables_trained = load_tables(DATASET=DATASET1)

    # Helper specifically for this panel with dataset + condition handling
    def activity_map_compilation_dataset(
        recording_day: str,
        visual_stimulus: int,
        exp_range: Tuple[float, float],
        behavior: Iterable[int],
        dataset: Dict[str, Any],
        tables,
        condition: str,
    ) -> np.ndarray:
        relevant_experiment_ids = retrieve_recording_day_experiment_ids(recording_day, tables)
        trial_bank = construct_trial_bank_by_percentage_behavior(
            relevant_experiment_ids,
            visual_stimulus,
            recording_day,
            exp_range,
            list(behavior),
            tables,
        )
        if condition == "trained":
            admissible_neurons = retrieve_admissible_neurons(relevant_experiment_ids, tables)
        elif condition == "naive":
            admissible_neurons = retrieve_admissible_neurons(
                relevant_experiment_ids,
                tables,
                fit_key="Best_Fit_spikes_1",
                orientation_key="Pref_Orientation_spikes_1",
            )
        else:
            raise ValueError("condition must be 'trained' or 'naive'")
        experiment_data = load_all_experiments_into_memory(trial_bank, DATASET=dataset, tqdm_disable=False)
        activity_map = construct_orientation_representation(
            admissible_neurons,
            experiment_data,
            TRIAL_LENGTH,
        )
        return activity_map

    def activity_profile_compilation(activity_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ap_mean = np.nanmean(activity_map[:, STIMULUS_ACTIVITY[0] : STIMULUS_ACTIVITY[1]], axis=-1)
        ap_std = np.nanstd(activity_map[:, STIMULUS_ACTIVITY[0] : STIMULUS_ACTIVITY[1]], axis=-1)
        return ap_mean, ap_std

    # Cache keys
    # Include behavior code in keys so caches are distinct for lick vs both
    behc = behavior_code()
    key_naive_mean = f"naive_vs_trained_2D_naive_D1_45_mean_beh-{behc}"
    key_naive_std = f"naive_vs_trained_2D_naive_D1_45_std_beh-{behc}"
    key_trained_mean = f"naive_vs_trained_2D_trained_D1_45_mean_beh-{behc}"
    key_trained_std = f"naive_vs_trained_2D_trained_D1_45_std_beh-{behc}"

    if cache_exists(key_naive_mean) and cache_exists(key_naive_std):
        naive_profile_mean = cache_load(key_naive_mean)
        naive_profile_std = cache_load(key_naive_std)
    else:
        naive_map = activity_map_compilation_dataset(
            "D1", 45, (0.0, 1.0), behavior_list_from_mode(), NAIVE, tables_naive, condition="naive"
        )
        naive_profile_mean, naive_profile_std = activity_profile_compilation(naive_map)
        cache_save(key_naive_mean, naive_profile_mean)
        cache_save(key_naive_std, naive_profile_std)

    if cache_exists(key_trained_mean) and cache_exists(key_trained_std):
        trained_profile_mean = cache_load(key_trained_mean)
        trained_profile_std = cache_load(key_trained_std)
    else:
        trained_map = activity_map_compilation_dataset(
            "D1", 45, (0.0, 1.0), behavior_list_from_mode(), DATASET1, tables_trained, condition="trained"
        )
        trained_profile_mean, trained_profile_std = activity_profile_compilation(trained_map)
        cache_save(key_trained_mean, trained_profile_mean)
        cache_save(key_trained_std, trained_profile_std)

    # Plot 2D profiles (orientation on y-axis) with optional folding
    if ORIENTATION_180_MODE == "fold":
        y_vals = ORIENTATIONS[:180]
        naive_x = 0.5 * (naive_profile_mean[:180] + naive_profile_mean[180:360])
        naive_x_std = 0.5 * (naive_profile_std[:180] + naive_profile_std[180:360])
        trained_x = 0.5 * (trained_profile_mean[:180] + trained_profile_mean[180:360])
        trained_x_std = 0.5 * (trained_profile_std[:180] + trained_profile_std[180:360])
    else:  # "cut"
        y_vals = ORIENTATIONS[:180]
        naive_x = naive_profile_mean[:180]
        naive_x_std = naive_profile_std[:180]
        trained_x = trained_profile_mean[:180]
        trained_x_std = trained_profile_std[:180]
    fig, ax = plt.subplots(figsize=(2.9, 3.48))
    ax.plot(naive_x, y_vals, lw=LW_PROFILE, color=COLOR_NAIVE, label="Naive")
    ax.fill_betweenx(
        y_vals,
        naive_x - naive_x_std,
        naive_x + naive_x_std,
    color=COLOR_NAIVE,
        alpha=0.3,
    )
    ax.plot(trained_x, y_vals, lw=LW_PROFILE, color=COLOR_GO, label="Trained")
    ax.fill_betweenx(
        y_vals,
        trained_x - trained_x_std,
        trained_x + trained_x_std,
    color=COLOR_GO,
        alpha=0.3,
    )
    ax.axhline(y=45, lw=LW_STIMULUS, color=COLOR_STIM_GO, ls=LS_DASHED)
    ax.set_ylim([180, 0])
    ax.set_yticks(np.arange(0, 181, 30))
    ax.set_ylabel("Orientation (°)")
    ax.set_xlabel("Population Activity")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    fn = "naive_vs_trained"
    plt.savefig(os.path.join(RESULTS_DIR, fn + ".png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS_DIR, PDF_DIR, "Fig1D_" + fn + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return fn


def nogo_profile_D2() -> str:
    """Generate a 1x3 figure for D2 NoGo: 4B, 4D, 4E.

    4B: Full-session profile (0.0–1.0)
    4D: Q1 vs Q5 profiles overlay
    4E: Shoulder activity (45°, 135°) across Q1..Q5 with error bars
    """
    tables = load_tables()

    def _prep(profile: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if ORIENTATION_180_MODE == "fold":
            x = ORIENTATIONS[:180]
            y = 0.5 * (profile[:180] + profile[180:360])
            ystd = 0.5 * (std[:180] + std[180:360])
        else:
            x = ORIENTATIONS[:180]
            y = profile[:180]
            ystd = std[:180]
        return x, y, ystd

    # 4B: Full-session D2 NoGo profile (0.0–1.0)
    ap_mean_full, ap_std_full = get_activity_profile_cached(
        tables, "D2", 90, (0.0, 1.0), behavior_list_from_mode()
    )

    # Quintile bins for 4D and 4E
    quintile_bins: List[Tuple[float, float]] = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0),
    ]

    quintile_profiles: List[Tuple[np.ndarray, np.ndarray]] = []
    for qbin in quintile_bins:
        q_mean, q_std = get_activity_profile_cached(
            tables, "D2", 90, qbin, behavior_list_from_mode()
        )
        quintile_profiles.append((q_mean, q_std))

    # Prepare figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"wspace": 0.3})

    # Panel 4B
    ax = axs[0]
    x, y, ystd = _prep(ap_mean_full, ap_std_full)
    ax.axvline(x=90, color=COLOR_STIM_NOGO, lw=LW_STIMULUS, ls=LS_DASHED)
    ax.plot(x, y, color=COLOR_NOGO, lw=LW_PROFILE)
    ax.fill_between(x, y - ystd, y + ystd, color=COLOR_NOGO, alpha=0.3)
    ax.set_xlim([0, 180])
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel("Population Activity")
    # ax.set_title("D2 NoGo")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 4D: Q1 vs Q5 overlay
    ax = axs[1]
    (q1_mean, q1_std) = quintile_profiles[0]
    (q5_mean, q5_std) = quintile_profiles[4]
    x1, y1, y1std = _prep(q1_mean, q1_std)
    x5, y5, y5std = _prep(q5_mean, q5_std)
    ax.axvline(x=90, color=COLOR_STIM_NOGO, lw=LW_STIMULUS, ls=LS_DASHED)
    # Q1 with main NoGo color; Q5 with lighter stimulus color for contrast
    ax.plot(x1, y1, color=COLOR_NOGO, lw=LW_PROFILE, label="Q1 (Early)")
    ax.fill_between(x1, y1 - y1std, y1 + y1std, color=COLOR_NOGO, alpha=0.25)
    ax.plot(x5, y5, color=COLOR_STIM_NOGO, lw=LW_PROFILE, label="Q5 (Late)")
    ax.fill_between(x5, y5 - y5std, y5 + y5std, color=COLOR_STIM_NOGO, alpha=0.25)
    ax.set_xlim([0, 180])
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel("Orientation (°)")
    ax.set_title("D2 nogo activity")    #     "Q1 vs Q5 (D2 NoGo)")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 4E: Shoulder activity across quintiles
    ax = axs[2]
    quintile_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    xq = np.arange(1, 6)
    # Extract from prepped profiles according to current mode
    def _shoulder(series_mean: np.ndarray, series_std: np.ndarray, deg: int) -> Tuple[float, float]:
        # deg is in degrees (1..180 meaningful here)
        idx = deg - 1  # 45° -> 44, 135° -> 134
        if ORIENTATION_180_MODE == "fold":
            # use folded profile
            mean_fold = 0.5 * (series_mean[:180] + series_mean[180:360])
            std_fold = 0.5 * (series_std[:180] + series_std[180:360])
            return float(mean_fold[idx]), float(std_fold[idx])
        else:
            return float(series_mean[idx]), float(series_std[idx])

    go_means, go_stds = [], []
    nogo_means, nogo_stds = [], []
    for (qm, qs) in quintile_profiles:
        m, s = _shoulder(qm, qs, 45)
        go_means.append(m)
        go_stds.append(s)
        m, s = _shoulder(qm, qs, 135)
        nogo_means.append(m)
        nogo_stds.append(s)
    # Plot errorbars
    ax.errorbar(xq, go_means, yerr=go_stds, label="Go shoulder (45°)",
                color=COLOR_GO, lw=LW_PROFILE, capsize=4, marker='o', linestyle='-')
    ax.errorbar(xq, nogo_means, yerr=nogo_stds, label="NoGo shoulder (135°)",
                color=COLOR_NOGO, lw=LW_PROFILE, capsize=4, marker='o', linestyle='-')
    ax.set_xticks(xq)
    ax.set_xticklabels(quintile_labels)
    ax.set_xlabel("Quintile")
    ax.set_ylabel("Mean Activity at Shoulder")
    # ax.set_title("Shoulder activity across quintiles")
    ax.set_xlim([0.8, 5.2])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fn = "nogo_profile_D2"
    plt.savefig(os.path.join(RESULTS_DIR, fn + ".png"), bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS_DIR, PDF_DIR, "Fig3BDE_" + fn + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return fn


# -----------------------------
# Figure: behavior_characterization.ipynb (modified)
# -----------------------------

def behaviour_characterization_acrossdays() -> str:
    """Compute and plot NoGo false alarm rates over days.

    Replicates the behavior_characterization notebook's data collation but plots
    only FA rate: mean across mice with a light error band (std).

    Returns the saved figure path.
    """
    # Cache key for the FA rate matrix + metadata
    cache_key = "behavior_fa_rates_over_days"

    if cache_exists(cache_key):
        data = cache_load(cache_key)
        fa_rate_matrix = data["fa_rate_matrix"]
        conditions = data["conditions"]
        mice = data["mice"]
    else:
        # Load tables
        tables = load_tables()
        _, df_trialtable = tables

        # Visual block trials and mouse labels
        experimental_trials = df_trialtable[df_trialtable["Block"] == "Visual"].copy()
        experimental_trials["mouse"] = experimental_trials["Experiment"].str[:3]

        # Recording days and mice
        conditions = np.sort(experimental_trials["Behav_Cond"].unique())
        mice = np.sort(experimental_trials["mouse"].unique())

        # Initialize FA rate matrix: rows=days, cols=mice
        fa_rate_matrix = np.zeros((len(conditions), len(mice)), dtype=float)

        # Compute per (day, mouse) FA rate among NoGo outcomes (CR, FA)
        for i, day in enumerate(conditions):
            for j, mouse in enumerate(mice):
                df_filtered = experimental_trials[(experimental_trials["Behav_Cond"] == day) & (experimental_trials["mouse"] == mouse)]
                # Count NoGo outcomes
                counts = df_filtered["Outcome"].value_counts()
                cr = counts.get("CR", 0)
                fa = counts.get("FA", 0)
                total_nogo = cr + fa
                fa_rate = (fa / total_nogo) if total_nogo > 0 else 0.0
                fa_rate_matrix[i, j] = fa_rate

        # Save cache as a dict for clarity
        cache_save(cache_key, {"fa_rate_matrix": fa_rate_matrix, "conditions": conditions, "mice": mice})

    # Prepare plot data
    x = np.arange(1, fa_rate_matrix.shape[0] + 1)
    mean = fa_rate_matrix.mean(axis=1)
    std = fa_rate_matrix.std(axis=1)

    # Plot FA rate with mean line and shaded std band
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, mean, color=COLOR_NOGO, linewidth=LW_PROFILE, label=None)
    ax.fill_between(x, mean - std, mean + std, color=COLOR_NOGO, alpha=0.2)
    ax.set_ylim([-0.05, 1.05])
    # Labels and styling
    ax.set_xlabel("Recording Day")
    ax.set_ylabel("False Alarm Rate")
    ax.set_title("NoGo False Alarm Rate Over Days")
    # Use day labels (e.g., D1..D6) on x-axis for clarity
    ax.set_xticks(x)
    try:
        ax.set_xticklabels(conditions)
    except Exception:
        pass
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()

    fn = "behavior_false_alarm_rate_acrossdays"
    plt.savefig(os.path.join(RESULTS_DIR, fn + ".png"), bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS_DIR, PDF_DIR, "SuppFig1_" + fn + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return fn


def behaviour_characterization_withinD2() -> str:
    """Plot NoGo false alarm rate within D2 across 5 quintiles of trials (0.0–1.0).

    Aggregates Visual-block D2 trials per mouse into 5 equal trial-percentage bins,
    computes FA rate per bin, and plots the mean with a shaded std band.
    """
    cache_key = "behavior_fa_rates_withinD2_quintiles"
    if cache_exists(cache_key):
        data = cache_load(cache_key)
        fa_quintiles = data["fa_quintiles"]
    else:
        tables = load_tables()
        _, df_trialtable = tables
        # Visual block only and D2 trials
        trials = df_trialtable[(df_trialtable["Block"] == "Visual") & (df_trialtable["Behav_Cond"] == "D2")].copy()
        trials["mouse"] = trials["Experiment"].str[:3]
        # Compute per-experiment trial percentage and define quintiles
        trials = compute_trial_percentages(trials)
        bins = np.linspace(0.0, 1.0, 6)
        trials["quintile"] = np.digitize(trials["Trial_Percentage"], bins, right=True)
        trials["quintile"].clip(1, 5, inplace=True)

        # For each quintile and mouse, compute FA rate among NoGo outcomes (CR, FA)
        mice = np.sort(trials["mouse"].unique())
        fa_quintiles = np.zeros((5, len(mice)), dtype=float)
        for q in range(1, 6):
            df_q = trials[trials["quintile"] == q]
            for j, mouse in enumerate(mice):
                df_m = df_q[df_q["mouse"] == mouse]
                counts = df_m["Outcome"].value_counts()
                cr = counts.get("CR", 0)
                fa = counts.get("FA", 0)
                tot = cr + fa
                fa_rate = (fa / tot) if tot > 0 else 0.0
                fa_quintiles[q - 1, j] = fa_rate

        cache_save(cache_key, {"fa_quintiles": fa_quintiles})

    # Plot
    x = np.arange(1, 6)
    mean = fa_quintiles.mean(axis=1)
    std = fa_quintiles.std(axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, mean, color=COLOR_NOGO, linewidth=LW_PROFILE, label=None)
    ax.fill_between(x, mean - std, mean + std, color=COLOR_NOGO, alpha=0.2)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Session progress")
    ax.set_ylabel("False Alarm Rate")
    ax.set_title("NoGo False Alarm Rate Within D2")
    ax.set_xticks(x)
    ax.set_xticklabels(["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fn = "behavior_false_alarm_rate_withinD2_quintiles"
    plt.savefig(os.path.join(RESULTS_DIR, fn + ".png"), bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS_DIR, PDF_DIR, "SuppFig2_" + fn + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return fn


# -----------------------------
# Figure: ghost_activity_comparison.ipynb (subset: full 360 panels for Go/NoGo)
# -----------------------------

def ghost_activity_go_nogo() -> str:
    """Replicate the full-360 Go and NoGo panels (displayed as 0–180) as a 1×2 figure.

    - Left: Go (Naive vs Trained), full 360 activity profiles, shaded std, stimulus lines at 45°, 135°
    - Right: NoGo (Naive vs Trained), same styling

    Data source: data/*/task_aggregated_matrices.npy; mean/std computed over stimulus window.
    Caches processed 360-length mean/std arrays per (dataset, condition).
    """
    # Cache keys for processed means/stds
    key_naive_go_mean = "ghost_naive_go_mean"
    key_naive_go_std = "ghost_naive_go_std"
    key_naive_nogo_mean = "ghost_naive_nogo_mean"
    key_naive_nogo_std = "ghost_naive_nogo_std"
    key_trained_go_mean = "ghost_trained_go_mean"
    key_trained_go_std = "ghost_trained_go_std"
    key_trained_nogo_mean = "ghost_trained_nogo_mean"
    key_trained_nogo_std = "ghost_trained_nogo_std"

    def _load_and_compute(dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (go_mean, go_std, nogo_mean, nogo_std) for given dataset id 'naive' or 'trained'."""
        if dataset == "naive":
            path = os.path.join(_PROJ_ROOT, "data", "Naive_numpy_arrays", "task_aggregated_matrices.npy")
        elif dataset == "trained":
            path = os.path.join(_PROJ_ROOT, "data", "DATASET1_numpy_arrays", "task_aggregated_matrices.npy")
        else:
            raise ValueError("dataset must be 'naive' or 'trained'")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing data file: {path}")
        mats = np.load(path)
        # Expect shape [condition, orientation, time]; condition 0=Go D1, 1=NoGo D1 (as per notebook)
        go_map = mats[0, :, :]
        nogo_map = mats[1, :, :]
        go_mean, go_std = compile_activity_profile(go_map, stimulus_activity=STIMULUS_ACTIVITY)
        nogo_mean, nogo_std = compile_activity_profile(nogo_map, stimulus_activity=STIMULUS_ACTIVITY)
        return go_mean, go_std, nogo_mean, nogo_std

    # Retrieve or compute caches
    if cache_exists(key_naive_go_mean) and cache_exists(key_naive_go_std) and \
       cache_exists(key_naive_nogo_mean) and cache_exists(key_naive_nogo_std):
        naive_go_mean = cache_load(key_naive_go_mean)
        naive_go_std = cache_load(key_naive_go_std)
        naive_nogo_mean = cache_load(key_naive_nogo_mean)
        naive_nogo_std = cache_load(key_naive_nogo_std)
    else:
        ngm, ngs, nngm, nngs = _load_and_compute("naive")
        naive_go_mean, naive_go_std = ngm, ngs
        naive_nogo_mean, naive_nogo_std = nngm, nngs
        cache_save(key_naive_go_mean, naive_go_mean)
        cache_save(key_naive_go_std, naive_go_std)
        cache_save(key_naive_nogo_mean, naive_nogo_mean)
        cache_save(key_naive_nogo_std, naive_nogo_std)

    if cache_exists(key_trained_go_mean) and cache_exists(key_trained_go_std) and \
       cache_exists(key_trained_nogo_mean) and cache_exists(key_trained_nogo_std):
        trained_go_mean = cache_load(key_trained_go_mean)
        trained_go_std = cache_load(key_trained_go_std)
        trained_nogo_mean = cache_load(key_trained_nogo_mean)
        trained_nogo_std = cache_load(key_trained_nogo_std)
    else:
        tgm, tgs, tngm, tngs = _load_and_compute("trained")
        trained_go_mean, trained_go_std = tgm, tgs
        trained_nogo_mean, trained_nogo_std = tngm, tngs
        cache_save(key_trained_go_mean, trained_go_mean)
        cache_save(key_trained_go_std, trained_go_std)
        cache_save(key_trained_nogo_mean, trained_nogo_mean)
        cache_save(key_trained_nogo_std, trained_nogo_std)


    # build error bars from neurons and trials raw data
    activity_go_trained_neuronstrials = get_activity_neuronstrials_cached(load_tables(), "D1", 45, (0.0, 1.0), [0,1])
    activity_go_naive_neuronstrials = get_activity_neuronstrials_cached(load_tables(DATASET=NAIVE), "D1", 45, (0.0, 1.0), [0,1], dataset_tag="naive")
    activity_nogo_trained_neuronstrials = get_activity_neuronstrials_cached(load_tables(), "D1", 135, (0.0, 1.0), [0,1])
    activity_nogo_naive_neuronstrials = get_activity_neuronstrials_cached(load_tables(DATASET=NAIVE), "D1", 135, (0.0, 1.0), [0,1], dataset_tag="naive")
    binwidth = 6
    sems_go_trained_nt = (compile_std_neuronstrials(activity_go_trained_neuronstrials, ORIENTATIONS[:180], binwidth=binwidth)) * 2
    sems_go_naive_nt = (compile_std_neuronstrials(activity_go_naive_neuronstrials, ORIENTATIONS[:180], binwidth=binwidth)) * 2
    sems_nogo_trained_nt = (compile_std_neuronstrials(activity_nogo_trained_neuronstrials, ORIENTATIONS[:180], binwidth=binwidth)) * 2
    sems_nogo_naive_nt = (compile_std_neuronstrials(activity_nogo_naive_neuronstrials, ORIENTATIONS[:180], binwidth=binwidth)) * 2



    x = ORIENTATIONS[:180]

    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=True)

    # Panel: Go
    ax = axs[0]
    ax.plot(x, naive_go_mean[:180], color=COLOR_NAIVE, lw=LW_PROFILE, label="Go, Naive")
    ax.fill_between(x, naive_go_mean[:180] - sems_go_naive_nt, naive_go_mean[:180] + sems_go_naive_nt,
                    color=COLOR_NAIVE, alpha=0.3)
    ax.plot(x, trained_go_mean[:180], color=COLOR_GO, lw=LW_PROFILE, label="Go, Trained")
    ax.fill_between(x, trained_go_mean[:180] - sems_go_trained_nt, trained_go_mean[:180] + sems_go_trained_nt,
                    color=COLOR_GO, alpha=0.3)
    ax.axvline(x=45, color=COLOR_STIM_GO, lw=LW_STIMULUS, ls=LS_DASHED)
    # ax.axvline(x=135, color=COLOR_STIM_NOGO, lw=LW_STIMULUS, ls=LS_DASHED)
    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel("Population Activity")
    ax.set_xlim([1, 180])
    ax.set_xticks(np.arange(0, 181, 30))
    # ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel: NoGo
    ax = axs[1]
    ax.plot(x, naive_nogo_mean[:180], color=COLOR_NAIVE, lw=LW_PROFILE, label="NoGo, Naive")
    ax.fill_between(x, naive_nogo_mean[:180] - sems_nogo_naive_nt, naive_nogo_mean[:180] + sems_nogo_naive_nt,
                    color=COLOR_NAIVE, alpha=0.3)
    ax.plot(x, trained_nogo_mean[:180], color=COLOR_NOGO, lw=LW_PROFILE, label="NoGo, Trained")
    ax.fill_between(x, trained_nogo_mean[:180] - sems_nogo_trained_nt, trained_nogo_mean[:180] + sems_nogo_trained_nt,
                    color=COLOR_NOGO, alpha=0.3)
    # ax.axvline(x=45, color=COLOR_STIM_GO, lw=LW_STIMULUS, ls=LS_DASHED)
    ax.axvline(x=135, color=COLOR_STIM_NOGO, lw=LW_STIMULUS, ls=LS_DASHED)
    ax.set_xlabel("Orientation (°)")
    # ax.set_ylabel("Population Activity")
    ax.set_xlim([1, 180])
    ax.set_xticks(np.arange(0, 181, 30))
    # ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # plt.tight_layout()
    fn = "ghost_activity_go_nogo"
    plt.savefig(os.path.join(RESULTS_DIR, fn + ".png"), bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, PDF_DIR, "Fig2EF_" + fn + ".pdf"), bbox_inches='tight')
    plt.close(fig)
    return fn



# -----------------------------
# Figure: shoulders_D2_5.ipynb (panels 1-3) – D2 NoGo Q1/Full, Q5/Full, shoulders across quintiles
# -----------------------------

def shoulders_d2_phenomenon() -> str:
    """Generate a 1×3 figure illustrating the D2 NoGo shoulder phenomenon.

    Activity profile comes from the NoGo trials in D2 sessions, split into quintiles.

    Panels:
    - (1) Q1 vs Q5 (D2 NoGo)
    - (2) Shoulder activity across quintiles (45° and 135°)
    """
    tables = load_tables()
    beh = behavior_list_from_mode()

    # Profiles for Full, Q1, Q5
    # ap_full_mean, ap_full_std = get_activity_profile_cached(tables, "D2", 90, (0.0, 1.0), beh)
    ap_q1_mean, ap_q1_std = get_activity_profile_cached(tables, "D2", 90, (0.0, 0.2), beh)
    ap_q5_mean, ap_q5_std = get_activity_profile_cached(tables, "D2", 90, (0.8, 1.0), beh)

    # Quintiles for panel 3
    quintile_bins: List[Tuple[float, float]] = [
        (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)
    ]
    quintile_profiles: List[Tuple[np.ndarray, np.ndarray]] = []
    activities_nt: List[np.ndarray] = []
    quintile_sems_nt: List[np.ndarray] = []
    binwidth = 6
    # collect NoGo trial profiles and activities
    for qbin in quintile_bins:
        q_mean, q_std = get_activity_profile_cached(tables, "D2", 90, qbin, beh)
        quintile_profiles.append((q_mean, q_std))
        activity_nt = get_activity_neuronstrials_cached(tables, "D2", 90, qbin, beh) # pull neuron-trial data
        activities_nt.append(activity_nt)
        q_sem_nt = compile_std_neuronstrials(activity_nt, ORIENTATIONS, binwidth=binwidth) * 2 # 95% CI
        quintile_sems_nt.append(q_sem_nt)



    # Statistical test for shoulder shift difference between Q1 and Q5
    # Test if (45°_Q5 - 135°_Q5) - (45°_Q1 - 135°_Q1) > 0
    # This tests whether the shoulder asymmetry increases from Q1 to Q5

    # Extract activity for both shoulders in Q1 and Q5
    activity_q1_nt = activities_nt[0]  # Q1 (0.0-0.2)
    activity_q5_nt = activities_nt[4]  # Q5 (0.8-1.0)

    # Get activities for each shoulder in each quintile
    activities_45_q1 = concatenate_orientation_activities(activity_q1_nt, 45, binwidth=1)
    activities_135_q1 = concatenate_orientation_activities(activity_q1_nt, 135, binwidth=1)
    activities_45_q5 = concatenate_orientation_activities(activity_q5_nt, 45, binwidth=1)
    activities_135_q5 = concatenate_orientation_activities(activity_q5_nt, 135, binwidth=1)

    print(f"\nShoulder shift statistical test:")
    print(f"  n(45° Q1)={len(activities_45_q1)}, n(135° Q1)={len(activities_135_q1)}")
    print(f"  n(45° Q5)={len(activities_45_q5)}, n(135° Q5)={len(activities_135_q5)}")

    n_q1 = min(len(activities_45_q1), len(activities_135_q1))
    n_q5 = min(len(activities_45_q5), len(activities_135_q5))
    diff_q1 = activities_45_q1[:n_q1] - activities_135_q1[:n_q1]  # Shoulder asymmetry in Q1
    diff_q5 = activities_45_q5[:n_q5] - activities_135_q5[:n_q5]  # Shoulder asymmetry in Q5
    n_min = min(len(diff_q1), len(diff_q5))
    diff_of_diffs = diff_q5[:n_min] - diff_q1[:n_min]

    diff_of_diffs = diff_of_diffs[~np.isnan(diff_of_diffs)]

    print(f"  Mean difference of differences: {np.mean(diff_of_diffs):.6f} ± {np.std(diff_of_diffs)/np.sqrt(len(diff_of_diffs)):.6f}")
    print(f"  n={len(diff_of_diffs)}")

    # Test if the difference of differences is significantly greater than 0
    t_stat_shoulders, p_value_shoulders = ttest_1samp(diff_of_diffs, popmean=0, alternative='greater')

    print(f"  t-test (difference of differences vs 0): t={t_stat_shoulders:.4f}, p={p_value_shoulders:.16f}")
    print(f"  H0: mean((45°_Q5 - 135°_Q5) - (45°_Q1 - 135°_Q1)) = 0")

    # One-tailed p-value for testing if difference of differences > 0
    if t_stat_shoulders > 0:
        p_value_one_tailed = p_value_shoulders / 2
        print(f"  One-tailed p-value (increase in asymmetry): {p_value_one_tailed:.6f}")
    else:
        p_value_one_tailed = 1 - (p_value_shoulders / 2)
        print(f"  One-tailed p-value (increase in asymmetry): {p_value_one_tailed:.6f} (opposite direction)")


    def _prep(profile: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if ORIENTATION_180_MODE == "fold":
            x = ORIENTATIONS[:180]
            y = 0.5 * (profile[:180] + profile[180:360])
            ystd = 0.5 * (std[:180] + std[180:360])
        else:
            x = ORIENTATIONS[:180]
            y = profile[:180]
            ystd = std[:180]
        return x, y, ystd

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))


    # Panel 1: Q1 Q5
    ax = axs[0]
    x, y_q1, ystd_q1 = _prep(ap_q1_mean, ap_q1_std)
    x, y_q5, ystd_q5 = _prep(ap_q5_mean, ap_q5_std)
    ystd_q1 = quintile_sems_nt[0][:180]
    ystd_q5 = quintile_sems_nt[4][:180]
    ax.axvline(x=45, color='black', lw=LW_STIMULUS, ls=LS_DASHED)
    ax.axvline(x=135, color='black', lw=LW_STIMULUS, ls=LS_DASHED)
    ax.plot(x, y_q1, color=COLOR_NOGO, alpha=0.4, lw=LW_PROFILE, label="Q1")
    ax.fill_between(x, y_q1 - ystd_q1, y_q1 + ystd_q1, color=COLOR_NOGO, alpha=0.25)
    ax.plot(x, y_q5, color=COLOR_NOGO, lw=LW_PROFILE, alpha=1.0, label="Q5")
    ax.fill_between(x, y_q5 - ystd_q5, y_q5 + ystd_q5, color=COLOR_NOGO, alpha=0.6)
    ax.set_xlim([0, 180])
    ax.set_ylim([-0.01, 0.025])
    ax.set_yticks([-0.01, 0.0, 0.01, 0.02])
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel("Population Activity")
    # ax.set_title("D2 NoGo: Q1 vs Q5")
    ax.legend(frameon=False)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: Shoulders across quintiles
    ax = axs[1]
    quintile_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    xq = np.arange(1, 6)
    def _shoulder(series_mean: np.ndarray, series_std: np.ndarray, deg: int) -> Tuple[float, float]:
        idx = deg - 1
        if ORIENTATION_180_MODE == "fold":
            mean_fold = 0.5 * (series_mean[:180] + series_mean[180:360])
            std_fold = 0.5 * (series_std[:180] + series_std[180:360])
            return float(mean_fold[idx]), float(std_fold[idx])
        else:
            return float(series_mean[idx]), float(series_std[idx])
    go_means, go_stds, nogo_means, nogo_stds = [], [], [], []
    for ((qm, qs), qen) in zip(quintile_profiles, quintile_sems_nt):
        m, s = _shoulder(qm, qs, 45)
        go_means.append(m); go_stds.append(qen[44])
        m, s = _shoulder(qm, qs, 135)
        nogo_means.append(m); nogo_stds.append(qen[134])
    ax.errorbar(xq, go_means, yerr=go_stds, label="Go shoulder (45°)",
                color=COLOR_GO, lw=LW_PROFILE, capsize=4, marker='o', linestyle='-')
    ax.errorbar(xq, nogo_means, yerr=nogo_stds, label="NoGo shoulder (135°)",
                color=COLOR_NOGO, lw=LW_PROFILE, capsize=4, marker='o', linestyle='-')
    ax.set_xticks(xq)
    ax.set_xticklabels(quintile_labels)
    ax.set_xlabel("Quintile")
    ax.set_yticks([-0.01, 0.0, 0.01])
    # ax.set_title("Shoulder activity across quintiles (D2 NoGo)")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fn = "shoulders_D2_phenomenon"
    plt.savefig(os.path.join(RESULTS_DIR, fn + ".png"), bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, PDF_DIR, "Fig4CD_" + fn + ".pdf"), bbox_inches='tight')
    plt.close(fig)
    return fn


# -----------------------------
# Figure: shoulders_D2_5.ipynb (panels 4-5) – Naive vs Likelihood, Estimated Task Prior (1×3 row)
# Panel 1: Trained Go profile (mean±std)
# -----------------------------

def shouldershift_d2_naive_likelihood_prior() -> str:
    """Generate a 1×3 figure: Trained Go profile, Naive vs Likelihood, Estimated Task Prior.

    Likelihood and Task Prior are inferred from D2 profiles:
    - r via peak positions in D2 NoGo Q1 (near 45°, 90°, 135°)
    - sigma_post via width of D2 Go Q1 around the 45° peak
    - sigma_llh = (sqrt(1+r^2)/r) * sigma_post
    - sigma_prior = r * sigma_llh
    """
    # Load tables for trained and naive datasets
    tables_trained = load_tables(DATASET=DATASET1)
    # tables_naive = load_tables(DATASET=NAIVE)
    beh = behavior_list_from_mode()

    # Profiles used for parameter estimation (D2 Q1)
    nogo_D2_q1_mean, _ = get_activity_profile_cached(tables_trained, "D2", 90, (0.0, 0.2), beh, dataset_tag="trained")
    # no need for go profile here
    # go_D2_q1_mean, _ = get_activity_profile_cached(tables_trained, "D2", 45, (0.0, 0.2), beh, dataset_tag="trained")

    # --- Notebook-equivalent estimation of r and sigma_llh ---
    # Find peaks in NoGo Q1 mean profile across all orientations
    peaks, _ = find_peaks(nogo_D2_q1_mean, distance=20)
    peak_locs = ORIENTATIONS[peaks]  # convert indices to degrees (1..360)

    # Nearest peaks to 45°, 90°, 135°
    peak_45 = closest_peak(45, peak_locs)
    peak_90 = closest_peak(90, peak_locs)
    peak_135 = closest_peak(135, peak_locs)
    # print("peaks", peak_45, peak_90, peak_135)
    
    # Compute r from left/right shoulders relative to mu_likelihood=90°
    mu_likelihood = 90.0
    mu_prior = 45.0
    r_left = np.sqrt((mu_prior - peak_45) / (peak_45 - mu_likelihood))
    mu_prior = 135.0
    r_right = np.sqrt((mu_prior - peak_135) / (peak_135 - mu_likelihood))
    r = 0.5 * (r_left + r_right)
    # print("r", r_left, r_right, r)

    # Measure FWHM around the 45° peak for NoGo Q1 and convert to sigma_post
    width, _, _ = measure_width(nogo_D2_q1_mean)
    sigma_post = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_llh = (np.sqrt(1+r**2)/r) * sigma_post
    sigma_prior_task = sigma_llh * r
    # print("width, sigma_post, sigma_llh, sigma_prior_task", width, sigma_post, sigma_llh, sigma_prior_task)

    # # Panel 0: Trained Go profile
    # go_D1_mean, go_D1_std = get_activity_profile_cached(tables_trained, "D1", 45, (0.0, 1.0), beh, dataset_tag="trained")

    # Panel 2: Naive profile (D1 full) vs Likelihood
    # ax = axs[0]
    # naive_go_mean, naive_go_std = get_activity_profile_cached(tables_naive, "D1", 45, (0.0, 1.0), beh, dataset_tag="naive")

    # Likelihood Gaussian centered at 90°
    domain = np.linspace(1, 180, 360)
    def _gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / max(sigma, 1e-12)) ** 2)
    likelihood_curve = _gauss(domain, 90.0, sigma_llh)
    likelihood_curve /= np.max(likelihood_curve) if np.max(likelihood_curve) > 0 else 1.0

    # Panel 3: Estimated Task Prior (comb of two Gaussians at 45° and 135°)
    prior_curve = 0.5 * (_gauss(domain, 45.0, sigma_prior_task) + _gauss(domain, 135.0, sigma_prior_task))
    prior_curve /= np.max(prior_curve) if np.max(prior_curve) > 0 else 1.0

    # Plot 1×2
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    # # Panel 0: Trained Go profile
    # ax = axs[0]
    # x = ORIENTATIONS[:180]
    # ax.plot(x, go_D1_mean[:180], color=COLOR_GO, lw=LW_PROFILE, label="Trained Go (D1)")
    # ax.fill_between(x, go_D1_mean[:180] - go_D1_std[:180], go_D1_mean[:180] + go_D1_std[:180],
    #                 color=COLOR_GO, alpha=0.3)
    # ax.axvline(x=45, color=COLOR_STIM_GO, lw=LW_STIMULUS, ls=LS_DASHED)
    # ax.set_xlim([0, 180])
    # ax.set_xticks(np.arange(0, 181, 30))
    # ax.set_xlabel("Orientation (°)")
    # ax.set_ylabel("Population Activity")
    # ax.set_title("Trained Go profile (D1)")
    # ax.grid(True, linestyle='--', alpha=0.3)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    # Panel 2: Naive (widest tuning) vs Likelihood (scaled), mimicking shoulders_D2_5.ipynb
    # Load naive tuning aggregated matrices and compute the widest naive profile
    ax = axs[0]
    naive_tuning_path = os.path.join(_PROJ_ROOT, "data", "Naive_numpy_arrays", "tuning_aggregated_matrices.npy")
    if not os.path.exists(naive_tuning_path):
        raise FileNotFoundError(f"Missing data file: {naive_tuning_path}")
    naive_activity_profiles = np.load(naive_tuning_path)

    # Construct tuning day/stim arrays as in notebook
    tuning_days = [f"D{i}" for i in range(1, 7) for _ in range(12)]
    tuning_stimulus = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
    tuning_stimuli = np.tile(tuning_stimulus, 6)

    # 1-second stimulus window for tuning (matches notebook)
    whenStim = two_second_stimulus_interval(naive_activity_profiles.shape[-1], starttime=0, stoptime=1)

    widths = []
    shifted_activity_profiles = np.zeros((naive_activity_profiles.shape[0], 360))
    for i in range(naive_activity_profiles.shape[0]):
        stim = tuning_stimuli[i]
        activity_profile = np.mean(naive_activity_profiles[i, :, whenStim].T, axis=-1)
        shifted_response = shift_signal_to_center(activity_profile, stim)
        shifted_activity_profiles[i, :] = shifted_response
        widths.append(measure_width(activity_profile)[0])

    max_idx = int(np.argmax(widths))
    sigma_naive = widths[max_idx] / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Create centered domain: -180 to 180 degrees
    domain_centered = np.linspace(-180, 180, num=720, endpoint=True)
    likelihood_scaled = 3 * norm.pdf(domain_centered, loc=0, scale=sigma_llh)

    # Center the naive profile: shift ORIENTATIONS to be centered at 0
    orientations_centered = ORIENTATIONS - 181  # Convert 1..360 to -180..179

    # Plot likelihood first, then widest naive profile, labels/titles as in notebook
    ax.plot(domain_centered, likelihood_scaled, color='black', lw=LW_PROFILE,
        label=f"Fitted likelihood σ={sigma_llh:.3}")
    ax.plot(orientations_centered, shifted_activity_profiles[max_idx], color=COLOR_NAIVE, lw=LW_PROFILE,
        label=f"Naive data σ={sigma_naive:.3}")
    ax.set_xlim(-120, 120)
    ax.set_xlabel("Relative Orientation (°)")
    # ax.set_title('Max-width naive activity profile vs. computed likelihood')
    ax.set_ylabel("Population Activity")

    # ax.grid(True)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    
    F_stat = (sigma_llh ** 2) / (sigma_prior_task ** 2)
    df_trained = 150 # 150 * 2 * 20 * 7 * 5  # orientations, stimuli, trials from Q1 D2, frames autocorr.corrected, mice
    df_prior_task = df_trained  # D2 + D1,D3-D6 and 5 x more trials above Q1-Q5
    p_value_f = 1 - f_dist.cdf(F_stat, df_trained, df_prior_task)
    print(f"\nLikelihood vs Task Prior width comparison F-test: F={F_stat:.4f}, p={p_value_f:.16f}, \n with df1={df_trained}, df2={df_prior_task}")
    
    F_stat = (sigma_llh ** 2) / (sigma_naive ** 2)
    df_trained = 150 # * 2 * 20 * 7 * 5  # orientations, stimuli, trials from Q1 D2, frames autocorr.corrected, mice
    df_naive = df_trained #* 6 * 5  # D2 + D1,D3-D6 and 5 x more trials above Q1-Q5
    p_value_f = 1 - f_dist.cdf(F_stat, df_trained, df_naive)
    print(f"\nNaive vs Likelihood width comparison F-test: F={F_stat:.4f}, p={p_value_f:.16f}, \n with df trained={df_trained}, df naive={df_naive}")


    # Panel 3: Estimated Task Prior
    ax = axs[1]
    ax.plot(domain, prior_curve, color=COLOR_PRIOR, lw=LW_PRIOR_PROFILE, label="Estimated\nPrior")          #"Estimated Task Prior")
    ax.axvline(x=45, color=COLOR_STIM_GO, lw=LW_STIMULUS, ls=LS_DASHED)
    ax.axvline(x=135, color=COLOR_PRIOR, lw=LW_STIMULUS, ls=LS_DASHED)
    ax.set_xlim([0, 180])
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel("Orientation (°)")
    # ax.set_title("Estimated Task Prior")
    # ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(frameon=False, fontsize=10, loc='upper center')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    fn = "shouldershift_D2_likelihood_prior"
    # plt.savefig(os.path.join(RESULTS_DIR, fn + ".png"), bbox_inches='tight')
    # plt.savefig(os.path.join(RESULTS_DIR, PDF_DIR, "Fig4EF_" + fn + ".pdf"), bbox_inches='tight')
    plt.close(fig)
    return fn






# -----------------------------
# Figure: brain model mapping
# -----------------------------

def brain_model_mapping() -> str:
    """Collect data for each stimulus: mean over neurons in orientation bins, and frames raw.

    Returns the saved figure path.
    """
    tables = load_tables()
    tables_naive = load_tables(DATASET=NAIVE)

    days = ["D1", "D1", "D2", "D2", "D3", "D3", "D4", "D4", "D5", "D5", "D6", "D6"]
    stim_means = [45, 135, 45, 90, 45, 75, 45, 70, 45, 65, 45, 60]
    exp_ranges = [(0.0, 1.0) for k in range(len(days))]
    behaviors = behavior_days_from_mode(len(days))

    all_binspertrials_trained = []
    all_binspertrials_naive = []

    # DATA COLLECTION LOOP
    for day, stim, rng, beh in zip(days, stim_means, exp_ranges, behaviors):
        cache_key_trained = f"binspertrials_trained_day-{day}_stim-{stim}"
        cache_key_naive = f"binspertrials_naive_day-{day}_stim-{stim}"
        
        if False: #cache_exists(cache_key_trained) and cache_exists(cache_key_naive):
            print(f"Loading cached data for {day}, {stim}")
            binspertrials_trained = cache_load(cache_key_trained)
            binspertrials_naive = cache_load(cache_key_naive)
        else:
            print(f"Computing data for {day}, {stim}")
            activity_trained_neuronstrials = get_activity_neuronstrials_cached(tables, day, stim, rng, beh)
            activity_naive_neuronstrials = get_activity_neuronstrials_cached(tables_naive, day, stim, rng, beh, dataset_tag="naive")

            binspertrials_trained = compile_activity_profile_neuronstrials(activity_trained_neuronstrials)
            binspertrials_naive = compile_activity_profile_neuronstrials(activity_naive_neuronstrials)

            cache_save(cache_key_trained, binspertrials_trained)
            cache_save(cache_key_naive, binspertrials_naive)

        print(day, stim, "trained", binspertrials_trained.shape)
        print(day, stim, "naive", binspertrials_naive.shape)
        
        all_binspertrials_trained.append(binspertrials_trained[:180])
        all_binspertrials_naive.append(binspertrials_naive[:180])


    binsize = 1
    ticks = np.arange(1, 181, binsize)
    # PLOTTING SECTION (moved outside the data collection loop)
    fig, axs = plt.subplots(6, 2, figsize=(8, 18), sharex=True)
    COLOR_TRAINED = 'red'
    for idx in range(0, 12, 2):
        row_idx = idx // 2
        day = days[idx]
        
        # Go stimulus
        stim_go = stim_means[idx]
        trained_go = all_binspertrials_trained[idx].reshape(-1, binsize, all_binspertrials_trained[idx].shape[1]).mean(axis=2).mean(axis=1)
        naive_go = all_binspertrials_naive[idx].reshape(-1, binsize, all_binspertrials_naive[idx].shape[1]).mean(axis=2).mean(axis=1)
        
        # NoGo stimulus
        stim_nogo = stim_means[idx + 1]
        trained_nogo = all_binspertrials_trained[idx + 1].reshape(-1, binsize, all_binspertrials_trained[idx + 1].shape[1]).mean(axis=2).mean(axis=1)
        naive_nogo = all_binspertrials_naive[idx + 1].reshape(-1, binsize, all_binspertrials_naive[idx + 1].shape[1]).mean(axis=2).mean(axis=1)
        
        # Plot Go
        ax_go = axs[row_idx, 0]
        ax_go.plot(ticks, trained_go, color=COLOR_TRAINED,label="trained")
        ax_go.plot(ticks, naive_go, color=COLOR_NAIVE,label="naive")
        ax_go.set_title(f"{day}, {stim_go}°", fontsize=10)
        ax_go.set_ylim([-0.01, 0.03])
        ax_go.set_xticks(ticks)
        if row_idx == 5:
            ax_go.set_xlabel("orientation bin")
            ax_go.set_ylabel("activity")
        
        # Plot NoGo
        ax_nogo = axs[row_idx, 1]
        ax_nogo.plot(ticks, trained_nogo, color=COLOR_TRAINED,label="trained")
        ax_nogo.plot(ticks,naive_nogo, color=COLOR_NAIVE,label="naive")
        ax_nogo.set_title(f"{day}, {stim_nogo}°", fontsize=10)
        ax_nogo.set_xticks(ticks)
        ax_nogo.set_ylim([-0.01, 0.03])
        if row_idx == 5:
            ax_nogo.set_xlabel("orientation bin")
    
    plt.tight_layout()
    fn = f"brain_model_mapping_b{binsize}"
    # plt.savefig(os.path.join(RESULTS_DIR, fn + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fn









# -----------------------------
# command line interface
# -----------------------------


KNOWN_TARGETS = {
    "activity_map": lambda: print("run this notebook end: notebooks/random_visualizations/orientation_activity_map_construction.ipynb"),
    "naive_vs_trained_profiles": naive_vs_trained_profiles,
    "ghost_activity_go_nogo": ghost_activity_go_nogo,
    "width_characterization": width_characterization,
    "baseline_comparisons": baseline_comparisons,
    "orientation_response_profiles": orientation_response_profiles,
    "behaviour_characterization_acrossdays": behaviour_characterization_acrossdays,
    "behaviour_characterization_withinD2": behaviour_characterization_withinD2,
    "shoulders_d2_phenomenon": shoulders_d2_phenomenon,
    "shouldershift_d2_naive_likelihood_prior": shouldershift_d2_naive_likelihood_prior,
    "brain_model_mapping": brain_model_mapping,
}


def _run_all() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, func in KNOWN_TARGETS.items():
        try:
            print(f"Running {name}...")
            res = func()
            results[name] = res
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results[name] = {"error": str(e)}
    return results


if __name__ == "__main__":
    if len(sys.argv) == 1:
        summary = _run_all()
        print(json.dumps(summary, indent=2, default=str))
    else:
        target = sys.argv[1]
        if target not in KNOWN_TARGETS:
            print(f"Unknown target '{target}'. Known targets: {', '.join(KNOWN_TARGETS)}")
            sys.exit(1)
        res = KNOWN_TARGETS[target]()
        if isinstance(res, dict):
            print(json.dumps(res, indent=2, default=str))
        else:
            print(str(res))
