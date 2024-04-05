from typing import Optional
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def gaussian_cdf_evaluation_KF(
    mu: np.ndarray, 
    cov: np.ndarray, 
    z: np.ndarray, 
    ax = None, 
    dim: Optional[int] = None
):
    N = len(z)
    cdf_arr = np.zeros((N, ))
    
    for i in range(N):
        # cdf_arr[i] = stats.multivariate_normal(mean=mu[i], cov=cov[i]).cdf(z[i])
        if dim is not None:
            cdf_arr[i] = stats.norm(mu[i, dim], cov[i, dim, dim]).cdf(z[i, dim])
        else:
            cdf_arr[i] = stats.multivariate_normal(mean=mu[i], cov=cov[i]).cdf(z[i])
    
    return cdf_arr


def gaussian_cdf_evaluation_PF(
    particles: np.ndarray, 
    z: np.ndarray, 
    w: np.ndarray, 
    ax = None, 
    n_bins: int = 100, 
    dim: Optional[int] = None, 
):
    """
    Evaluating particle filteirng performance (see, e.g., Extended Fig. 2 in https://www.nature.com/articles/s41586-021-04129-3)
    """
    N, D = particles[0].shape
    
    cdf_arr = np.zeros((len(z), ))
    
    for i in range(len(z)):
        # cdf_arr[i] = np.sum(w[i][np.sum(np.less_equal(particles[i], z[i]), axis=-1) == D])
        # cdf_arr[i] = np.sum(w[i][particles[i, :, 1] <= z[i, 1]])
        if dim is not None:
            cdf_arr[i] = np.sum(w[i][particles[i, :, dim] <= z[i, dim]])
        else:
            cdf_arr[i] = np.sum(w[i][np.sum(np.less_equal(particles[i], z[i]), axis=-1) == D])
        # cdf_arr[i] = np.sum(np.sum(np.less_equal(particles[i], z[i]), axis=-1) == D) / N
    
    # cdf_hist, bin_edges = np.histogram(cdf_arr, bins=n_bins)
    # cumulative = np.cumsum(cdf_hist) / np.sum(cdf_hist)
    
    # bin_edges = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    
    sorted_cdf = np.sort(cdf_arr)
    ecdf = np.cumsum(np.ones_like(cdf_arr)) / len(cdf_arr)
    cumulative = np.cumsum(sorted_cdf) / np.sum(sorted_cdf)
    
    if ax is None:
        fig, ax = plt.subplots()

    # for i in range(D):
    # ax.plot(sorted_cdf, cumulative, label="particle-filtering")
    ax.plot(sorted_cdf, ecdf, label="particle-filtering")
    # ax.plot(bin_edges, cumulative)
    
    # ax.plot(np.arange(len(particles)), np.arange(len(particles)) / D, "k--")
    
    ax.set_xlabel("q")
    ax.set_ylabel("cumulative distribution")
    ax.legend()
    
    ax.set_xlim(0.0, 1.0)
    
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--")
    
    plt.show()