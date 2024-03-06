import numpy as np


def resampling(
    particles: np.ndarray, 
    w: np.ndarray, 
    resampling_method: str = "systematic", 
):
    if resampling_method == "systematic":
        inds = systematic_resampling(w)
    elif resampling_method == "stratified":
        inds = stratified_resampling(w)
    else:
        raise NotImplementedError
    
    return resample_from_inds(particles, w, inds)


def systematic_resampling(w: np.ndarray):
    """
    Systematic resampling (Algorithm 2 in https://ieeexplore.ieee.org/document/978374)
    """
    
    N = len(w)
    
    pos = (np.random.uniform() + np.arange(N)) / N
    inds = np.zeros((N, ), dtype=np.int32)
    cumulative_sum = np.cumsum(w)
    
    i, j = 0, 0
    while i < N:
        if pos[i] < cumulative_sum[j]:
            inds[i] = j
            i += 1
        else:
            j += 1
    
    return inds


def stratified_resampling(w: np.ndarray):
    
    N = len(w)
    
    pos = (np.random.uniform(size=(N, )) + np.arange(N)) / N
    inds = np.zeros((N, ), dtype=np.int32)
    cumulative_sum = np.cumsum(w)
    i, j = 0, 0
    while i < N:
        if pos[i] < cumulative_sum[j]:
            inds[i] = j
            i += 1
        else:
            j += 1
    
    return inds


def effective_size(w: np.ndarray):
    return 1. / np.sum(np.square(w))


def resample_from_inds(
    particles: np.ndarray, 
    w: np.ndarray, 
    inds: np.ndarray, 
):
    particles = particles[inds]
    w = np.ones_like(w) * (1.0 / len(w))
    
    return particles, w
