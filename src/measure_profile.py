import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import vonmises
from scipy.signal import find_peaks

from src.orientation_characterization import shift_signal_to_center, gain_measure

def measure_width(orientation_activity, height_percent=0.5, gain_min=0.005):
    """How many degrees wide is the response?"""
    # orientation_pmf = create_pdf(orientation_activity)
    gain = gain_measure(orientation_activity)
    activity_normalized = orientation_activity / gain[0]
    peak_index = gain[1] - 1  # Since orientation is 1-indexed

    # Check if the signal is strong enough
    if gain[0] < gain_min:
        raise ValueError(f'Activity profile had max response of {gain}')

    # Roll the array so that the peak is centered at index 180
    center_index = 179
    rolled_activity = shift_signal_to_center(activity_normalized, peak_index)
    peak_index = center_index  # Update the peak index after rolling

    # Search for left crossing
    left_index = peak_index
    while left_index > 0 and rolled_activity[left_index] > height_percent:
        left_index -= 1
    if left_index > 0:
        y0, y1 = rolled_activity[left_index], rolled_activity[left_index + 1]
        x0 = left_index
        left_interp = x0 + (height_percent - y0) / (y1 - y0)
    else:
        left_interp = left_index  # fallback
    
    # Search for right crossing
    right_index = peak_index
    while right_index < len(rolled_activity) - 1 and rolled_activity[right_index] > height_percent:
        right_index += 1
    if right_index < len(rolled_activity) - 1:
        y0, y1 = rolled_activity[right_index - 1], rolled_activity[right_index]
        x0 = right_index - 1
        right_interp = x0 + (height_percent - y0) / (y1 - y0)
    else:
        right_interp = right_index  # fallback
    
    # Width as a float
    width = right_interp - left_interp

    # Roll indices back
    rolled_left_index = left_index - (180-gain[1])
    rolled_right_index = right_index - (180-gain[1])
    return width, rolled_right_index, rolled_left_index

def closest_peak(target, peak_locs):
    return peak_locs[np.argmin(np.abs(peak_locs - target))]