import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def gaussian_cdf_evaluation(
    particles: np.ndarray, 
    z: np.ndarray, 
    w: np.ndarray, 
    ax = None
):
    """
    Evaluating particle filteirng performance (see, e.g., Extended Fig. 2 in https://www.nature.com/articles/s41586-021-04129-3)
    """
    N, D = particles[0].shape
    
    cdf_arr = np.zeros((len(z), ))
    
    for i in range(len(z)):
        cdf_arr[i] = np.sum(w[i][np.sum(np.less_equal(particles[i], z[i]), axis=-1) == D])
    
    sorted_cdf = np.sort(cdf_arr)
    cumulative = np.cumsum(sorted_cdf) / np.sum(sorted_cdf)
    
    if ax is None:
        fig, ax = plt.subplots()

    for i in range(D):
        ax.plot(sorted_cdf, cumulative)
    
    # ax.plot(np.arange(len(particles)), np.arange(len(particles)) / D, "k--")
    
    ax.set_xlabel("q")
    ax.set_ylabel("cumulative distribution")
    ax.legend()
    
    plt.show()