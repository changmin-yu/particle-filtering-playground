from typing import Optional, List

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
    lims: List[np.ndarray], 
    num: int, 
):
    D = len(lims)
    particles = np.zeros((num, D))
    for d in range(D):
        particles[:, d] = np.random.uniform(lims[d][0], lims[d][1], size=(num, ))
    
    return particles


def init_particles_gaussian(
    mean: np.ndarray, 
    cov: np.ndarray, 
    num: int
):
    particles = np.random.multivariate_normal(mean, cov, size=(num, ))
    
    return particles

