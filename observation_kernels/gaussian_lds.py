import numpy as np
from scipy import stats

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from observation_kernels.base import ObservationKernel


class LDSObservationKernel(ObservationKernel):
    def __init__(
        self, 
        C: np.ndarray, 
        Sigma: np.ndarray, 
    ):
        super().__init__()
        
        self.C = C
        self.Sigma = Sigma
    
    def likelihood(self, x: np.ndarray, obs: np.ndarray):
        N = x.shape[0]
        reconstructed_obs_mean = np.dot(self.C, x.T).T
        likelihood = np.array([
            stats.multivariate_normal(reconstructed_obs_mean[i], self.Sigma).pdf(obs)
            for i in range(N)
        ])
        
        return likelihood
