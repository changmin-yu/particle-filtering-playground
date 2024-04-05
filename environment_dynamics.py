from typing import Optional

import numpy as np 
from scipy.linalg import solve_discrete_lyapunov

class BaseEnv:
    def __init__(self):
        pass
    
    def step(self):
        return
    
    def obs_state(self):
        return
    

class RobotNavigationEnvSimple(BaseEnv):
    def __init__(
        self, 
        init_state: np.ndarray, 
        obs_std: float, 
        landmarks: np.ndarray
    ):
        self.init_state = init_state
        self.obs_std = obs_std
        self.landmarks = landmarks
        
    def step(self):
        self.state += np.array([1., 1.])
        
        control_signal = np.array([0.0, np.sqrt(2)])
        
        return control_signal
    
    def obs_state(self):
        return np.linalg.norm(self.landmarks - self.state, axis=-1) + np.random.randn(len(self.landmarks)) * self.obs_std
    
    
class LDSSimple(BaseEnv):
    def __init__(
        self, 
        A: np.ndarray, 
        C: np.ndarray, 
        Gamma: np.ndarray, 
        Sigma: np.ndarray, 
        mu0: Optional[np.ndarray] = None, 
        Gamma0: Optional[np.ndarray] = None, 
    ):
        super().__init__()
        
        self.A = A
        self.C = C
        self.Gamma = Gamma
        self.Sigma = Sigma
        if mu0 is None or Gamma0 is None:
            mu0 = np.array([0., 0.])
            Gamma0 = solve_discrete_lyapunov(A, Gamma)
        self.mu0 = mu0
        self.Gamma0 = Gamma0
    
        self.init_state = np.random.multivariate_normal(self.mu0, self.Gamma0)
        self.curr_state = self.init_state
    
    def step(self):
        self.curr_state = np.random.multivariate_normal(np.dot(self.A, self.curr_state), self.Gamma)
        
        return self.curr_state

    def obs_state(self):
        return np.random.multivariate_normal(np.dot(self.C, self.curr_state), self.Gamma)
