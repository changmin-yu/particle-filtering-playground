import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from transition_kernels.base import TransitionKernel


class RobotGaussianTransitionKernel(TransitionKernel):
    def __init__(self, control_noise: np.ndarray, dt: float):
        super().__init__()
        self.control_noise = control_noise
        self.dt = dt
    
    def step(self, x: np.ndarray, u: np.ndarray):
        N, D = x.shape
        L = u.shape[-1]
        
        assert D == 3, f"input particles must be three-dimensional, received {D}-dimensional inputs"
        assert L == 2, f"input control signal must be two-dimensional, received {L}-dimensional control"
        
        d_hd = np.random.normal(u[0], self.control_noise[0], size=(N, ))
        x[:, 2] += d_hd
        x[:, 2] %= (2 * np.pi)
        
        dist = np.random.randn(N) * self.control_noise[1] + u[1] * self.dt
        x[:, 0] += np.cos(x[:, 2]) * dist
        x[:, 1] += np.sin(x[:, 2]) * dist
        
        return x