import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from environment.base import BaseEnv


class RobotNavigationEnvSimple(BaseEnv):
    def __init__(
        self, 
        init_state: np.ndarray, 
        obs_std: float, 
        landmarks: np.ndarray
    ):
        self.state = init_state
        self.obs_std = obs_std
        self.landmarks = landmarks
        
    def step(self):
        self.state += np.array([1., 1.])
        
        control_signal = np.array([0.0, np.sqrt(2)])
        
        return control_signal
    
    def obs_state(self):
        return np.linalg.norm(self.landmarks - self.state, axis=-1) + np.random.randn(len(self.landmarks)) * self.obs_std
    