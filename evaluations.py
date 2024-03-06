import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def gaussian_cdf_evaluation(
    particles: np.ndarray, 
    ax = None
):
    """
    Evaluating particle filteirng performance (see, e.g., Extended Fig. 2 in https://www.nature.com/articles/s41586-021-04129-3)
    """
    particles = (particles - np.mean(particles, axis=0, keepdims=True)) / np.std(particles, axis=0, keepdims=True)
    
    particles = np.reshape(particles, (len(particles), -1))
    
    if ax is None:
        fig, ax = plt.subplots()

    for i in range(particles.shape[-1]):
        ax.plot(stats.norm(0, 1).cdf(np.sort(particles[:, i])), label=str(i))
    
    ax.plot(np.arange(len(particles)), np.arange(len(particles)) / len(particles), "k--")
    
    ax.legend()
    
    plt.show()