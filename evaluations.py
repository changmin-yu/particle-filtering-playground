from typing import Optional
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def gaussian_cdf_evaluation_KF(
    mu: np.ndarray, 
    cov: np.ndarray, 
    z: np.ndarray, 
    dim: Optional[int] = None, 
    ax=None,
    color: str = "red", 
):
    N, T, D = z.shape
    cdf_arr = np.zeros((N, T))

    for n in range(N):
        for t in range(T):
            if dim is not None:
                cdf_arr[n, t] = stats.norm(mu[n, t, dim], np.sqrt(cov[n, t, dim, dim])).cdf(z[n, t, dim])
            else:
                for d in range(D):
                    cdf_arr[n, t] += stats.norm(mu[n, t, d], np.sqrt(cov[n, t, d, d])).cdf(z[n, t, d])
                cdf_arr[n, t] /= D

    sorted_cdf = np.sort(cdf_arr.flatten())
    ecdf = np.cumsum(np.ones_like(sorted_cdf)) / len(sorted_cdf)
    
    if ax is None:
        ax = plt.gca()
    
    ax.plot(sorted_cdf, ecdf, label="Kalman filtering", color=color)
    
    return cdf_arr


def gaussian_cdf_evaluation_PF(
    particles: np.ndarray, 
    z: np.ndarray, 
    w: np.ndarray, 
    ax = None, 
    dim: Optional[int] = None, 
    label: Optional[str] = None, 
    color: str = "red"
):
    """
    Evaluating particle filteirng performance (see, e.g., Extended Fig. 2 in https://www.nature.com/articles/s41586-021-04129-3)
    """
    T, N, D = particles.shape
    
    cdf_arr = np.zeros((len(z), ))
    
    for i in range(T):
        if dim is not None:
            cdf_arr[i] = np.sum(w[i][particles[i, :, dim] <= z[i, dim]])
        else:
            cdf_arr[i] = np.sum(w[i][np.sum(np.less_equal(particles[i], z[i]), axis=-1) == D])
    
    sorted_cdf = np.sort(cdf_arr)
    ecdf = np.cumsum(np.ones_like(cdf_arr)) / len(cdf_arr)
    
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(sorted_cdf, ecdf, label=label, color=color)
    
    ax.set_xlabel("q")
    ax.set_ylabel("cumulative distribution")
    
    ax.set_xlim(0.0, 1.0)
    
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--")
