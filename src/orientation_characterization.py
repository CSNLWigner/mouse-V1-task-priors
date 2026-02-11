import os
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.integrate import simpson as simps

def shift_signal_to_center(signal, presented_orientation):
    # Calculate the shift amount to center the presented orientation at index 179 (for 180°)
    shift_amount = 179 - (presented_orientation - 1)
    return np.roll(signal, shift_amount)

def shift_signal_to_orientation(shifted_signal, target_orientation):
    # Calculate the difference in orientations and the corresponding shift
    shift_amount = (target_orientation - 180)
    
    # Apply the shift (adjust for the difference between orientation and index)
    return np.roll(shifted_signal, shift_amount)

def create_pdf(array, dx=1):
    # Shift the array so the minimum value is 0
    shifted_array = array - np.min(array)
    
    # Compute the area under the array
    area = np.sum(shifted_array) * dx
    
    # Normalize the array so the area under it is 1
    normalized_array = shifted_array / area
    
    return normalized_array

def gain_measure(orientation_activity,):
    """How large and where is the response?"""
    assert orientation_activity.shape == (360,), "orientation_activity is not an array of shape (360,)"
    max_response = np.max(orientation_activity).item()
    max_response_orientation = np.argmax(orientation_activity).item() + 1 # Since orientation is 1-index
    return max_response, max_response_orientation

def width_measure(orientation_activity, presented_stim=None, height_percent=0.5, return_indices=False, gain_min=0.005):
    """How many degrees wide is the response?"""
    # orientation_pmf = create_pdf(orientation_activity)
    gain = gain_measure(orientation_activity)
    activity_normalized = orientation_activity / gain[0]
    peak_index = gain[1] - 1  # Since orientation is 1-indexed

    # Check if peak_index is too far from presented_stim
    if presented_stim:
        if abs(presented_stim-gain[1]) > 30:
            raise ValueError(f'Presented stimulus of {presented_stim} and max activity orientation of {gain[1]}')

    # Check if the signal is strong enough
    if gain[0] < gain_min:
        raise ValueError(f'Activity profile had max response of {gain}')

    # Roll the array so that the peak is centered at index 180
    center_index = 179
    rolled_activity = shift_signal_to_center(activity_normalized, peak_index)
    peak_index = center_index  # Update the peak index after rolling

    # Search for the left index where the value crosses half-max before the peak
    left_index = peak_index
    while left_index > 0 and rolled_activity[left_index] > height_percent:
        left_index -= 1

    # Search for the right index where the value crosses half-max after the peak
    right_index = peak_index
    while right_index < len(rolled_activity) - 1 and rolled_activity[right_index] > height_percent:
        right_index += 1

    # Width in degrees
    width = right_index - left_index

    if return_indices:
        rolled_left_index = left_index - (180-gain[1])
        rolled_right_index = right_index - (180-gain[1])
        return width, (rolled_left_index, rolled_right_index)
    else:
        return width

def estimate_width_distribution(trial_matrix, presented_stim):
    """Estimate the mean and standard deviation for activity widths in an experiment."""
    assert trial_matrix.shape[-1] == 360, "For some reason I made this function with orientation as the columns"
    
    frame_widths = []
    frames = []
    for i in range(trial_matrix.shape[0]):
        activity_profile_frame = trial_matrix[i,:]
        try:
            frame_widths.append(width_measure(activity_profile_frame, presented_stim=presented_stim))
            frames.append(i)
        except ValueError:
            pass

    return np.array(frame_widths), frames

def compute_mean_and_variance(pmf, values):
    """
    Compute the mean and variance of a PMF.

    Parameters:
        pmf (array-like): Array of probabilities (must sum to 1).
        values (array-like): Array of corresponding values for the random variable.

    Returns:
        tuple: (mean, variance)
    """
    # Ensure PMF sums to 1
    pmf = np.array(pmf)
    pmf = pmf / np.sum(pmf)  # Normalize if not already

    # Compute the mean
    mean = np.sum(values * pmf)

    # Compute the variance using E[x^2] - E[x]^2
    variance = np.sum(pmf * (values-mean)**2)

    return mean, variance

def variance_based_width(activity, presented_orientation):
    orientations = np.arange(1, 361)
    shifted_activity = shift_signal_to_center(activity, presented_orientation)
    shifted_pmf = create_pdf(shifted_activity)
    mean, variance = compute_mean_and_variance(
        shifted_pmf[89:270],
        orientations[89:270]
    )
    return variance

def compute_area_between_curves(orientations, naive_profile, trained_profile, boundaries):
    """
    Used to assess the strength of the ghost activity in the activity profiles.

    Note:
        boundaries (tuple): A tuple specifying the range (min_x, max_x) for the integration.
    """
    # Extract the indices for the specified boundary range
    min_x, max_x = boundaries
    indices = (orientations >= min_x) & (orientations <= max_x)
    
    # Select the relevant portions of the curves
    x_subset = orientations[indices]
    naive_subset = naive_profile[indices]
    trained_subset = trained_profile[indices]
    
    # Compute the absolute difference between the curves
    diff = trained_subset - naive_subset
    
    # Compute the area using numerical integration (Simpson's rule)
    area = simps(diff, x=x_subset)
    return area.item()

def estimate_ghost_area_distribution(orientations, naive_matrix, trained_matrix, presented_stim):
    """Estimate the mean and standard deviation for ghost area."""
    assert naive_matrix.shape[-1] == 360, "For some reason I made this function with orientation as the columns"
    assert trained_matrix.shape[-1] == 360, "For some reason I made this function with orientation as the columns"

    boundaries = (presented_stim+45, presented_stim+135)
    
    ghost_areas = []
    frames = []
    for i in range(naive_matrix.shape[0]):
        naive_profile = naive_matrix[i,:]
        trained_profile = trained_matrix[i,:]
        
        try:
            # These widths are used to filter out low-amplitude signals
            naive_width = width_measure(naive_profile, presented_stim=presented_stim)
            trained_width = width_measure(trained_profile, presented_stim=presented_stim)
            
            ghost_areas.append(compute_area_between_curves(orientations, naive_profile, trained_profile, boundaries))
            frames.append(i)
            
        except ValueError:
            pass

    return np.array(ghost_areas), frames