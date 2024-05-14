import numpy as np


def estimate_mean_variance_robot_location(
    particles: np.ndarray, 
    w: np.ndarray
):
    pos = particles[:, :2]
    mean = np.average(pos, axis=0, weights=w)
    var = np.average(np.square(pos - mean), axis=0, weights=w)
    
    return mean, var