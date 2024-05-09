import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from transition_kernels.base import TransitionKernel


class LDSTransitionKernel(TransitionKernel):
    def __init__(self, A: np.ndarray, Gamma: np.ndarray):
        super().__init__()
        
        self.A = A
        self.Gamma = Gamma
        
    def step(self, x: np.ndarray):
        N, D = x.shape

        noise = np.random.multivariate_normal(np.zeros((D, )), self.Gamma, size=(N, ))
        x = np.matmul(self.A, x.T).T + noise
        
        return x
    