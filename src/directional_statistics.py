import numpy as np
from scipy.special import i0  # Modified Bessel function of the first kind
from scipy.stats import vonmises

from .orientation_characterization import create_pdf


def circular_mean(activity):
    """
    Compute the circular mean of a circular distribution given its activity or PDF.

    Parameters:
        activity (array-like): Pointwise activity values or PDF over 1° to 360°.

    Returns:
        float: Circular mean in degrees.
    """
    # Transform the activity into a proper PDF
    pdf = create_pdf(activity)  # Normalize the activity to sum to 1

    # Define the angles (1° to 360°) in radians
    angles = np.linspace(1, 360, 360)
    angles_rad = np.deg2rad(angles)

    # Compute the circular components
    C = np.sum(pdf * np.cos(angles_rad))
    S = np.sum(pdf * np.sin(angles_rad))

    # Compute the mean angle in radians
    mean_angle_rad = np.arctan2(S, C)

    # Convert the mean angle back to degrees
    mean_angle_deg = np.rad2deg(mean_angle_rad)

    # Ensure the result is in the range [0, 360)
    return mean_angle_deg % 360


def circular_variance(activity):
    """
    Compute the circular variance of a circular distribution given its PDF.

    Parameters:
        pdf (array-like): Pointwise PDF values over 1° to 360°.

    Returns:
        float: The circular variance (range: 0 to 1).
    """
    # Transform representation activity into PDF
    pdf = create_pdf(activity)

    # Define the angles (1° to 360°) in radians
    angles = np.linspace(1, 360, 360)
    angles_rad = np.deg2rad(angles)

    # Compute the circular components
    C = np.sum(pdf * np.cos(angles_rad))
    S = np.sum(pdf * np.sin(angles_rad))

    # Compute the mean resultant length
    R = np.sqrt(C**2 + S**2)

    # Compute the circular variance
    V = 1 - R
    return V


def von_mises_pdf(theta, mu, kappa):
    """
    Compute the von Mises PDF.

    Parameters:
        theta (array-like): Angles in radians.
        mu (float): Circular mean (radians).
        kappa (float): Concentration parameter.

    Returns:
        array-like: Probability density values.
    """
    return np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * i0(kappa))


def compute_von_mises(activity):
    """
    Compute parameters and generate an idealized von Mises distribution.

    Parameters:
        activity (array-like): Activities of orientation-selective neurons (360 units long).

    Returns:
        tuple: (angles, idealized_pdf)
    """
    # Normalize activity into a PDF
    pdf = create_pdf(activity)

    # Angles in radians
    angles = np.deg2rad(np.linspace(0, 359, 360))

    # Circular mean and variance
    mu = np.deg2rad(circular_mean(activity))  # Circular mean in radians
    V = circular_variance(activity)          # Circular variance

    # Estimate kappa (concentration parameter)
    kappa = (1 - V) / V if V > 0 else 0  # Handle edge cases for uniform distributions

    # Generate von Mises PDF
    idealized_pdf = von_mises_pdf(angles, mu, kappa)

    return angles, idealized_pdf


def von_mises(x, kappa, mu_fixed):
    """
    Compute von Mises probability density function with scipy.
    """
    return vonmises.pdf(x, kappa, loc=mu_fixed)



def find_local_max_between(profile, std, start_deg, end_deg):
    """Find local maximum between start_deg and end_deg (inclusive)."""
    start_idx = start_deg - 1  # Convert to 0-based index
    end_idx = end_deg - 1
    segment = profile[start_idx:end_idx+1]
    if len(segment) == 0:
        return np.nan, np.nan, np.nan
    max_idx_in_segment = np.argmax(segment)
    max_idx_global = start_idx + max_idx_in_segment
    max_deg = max_idx_global + 1  # Convert back to 1-based degree
    return max_deg, profile[max_idx_global], std[max_idx_global]
