from typing import Optional

import numpy as np


def init_particles(
    num: int, 
    init_x: Optional[np.ndarray] = None, 
    **kwargs, 
):
    if init_x is None:
        particles = init_particles_uniform(num=num, **kwargs)
    else:
        particles = init_particles_gaussian(mean=init_x, num=num, **kwargs)
    return particles


def init_particles_uniform(
    x_range: np.ndarray,  
    y_range: np.ndarray,  
    hd_range: np.ndarray,  
    num: int, 
):
    """
    Initialising particles uniformly, we assume only three latent variables (x and y location, and head direction)
    """
    particles = np.zeros((num, 3))
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=(num, ))
    particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=(num, ))
    particles[:, 2] = np.random.uniform(hd_range[0], hd_range[1], size=(num, ))
    
    particles[:, 2] %= 2 * np.pi
    
    return particles


def init_particles_gaussian(
    mean: np.ndarray, 
    std: np.ndarray, 
    num: int
):
    particles = np.zeros((num, 3))
    particles[:, 0] = mean[0] + np.random.randn(num) * std[0]
    particles[:, 1] = mean[1] + np.random.randn(num) * std[1]
    particles[:, 2] = mean[2] + np.random.randn(num) * std[2]
    
    particles[:, 2] %= 2 * np.pi
    
    return particles

