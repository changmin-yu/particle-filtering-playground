import numpy as np


class TransitionKernel:
    def __init__(self):
        pass
    
    def step(self, x: np.ndarray, **kwargs):
        return
    

class RobotGaussianTransitionKernel(TransitionKernel):
    def __init__(self):
        super().__init__()
    
    def step(self, x: np.ndarray, u: np.ndarray, u_std: np.ndarray, dt: float):
        N, D = x.shape
        L = u.shape[-1]
        
        assert D == 3, f"input particles must be three-dimensional, received {D}-dimensional inputs"
        assert L == 2, f"input control signal must be two-dimensional, received {L}-dimensional control"
        
        d_hd = np.random.normal(u[0], u_std[0], size=(N, ))
        x[:, 2] += d_hd
        x[:, 2] %= (2 * np.pi)
        
        dist = np.random.randn(N) * u_std[1] + u[1] * dt
        x[:, 0] += np.cos(x[:, 2]) * dist
        x[:, 1] += np.sin(x[:, 2]) * dist
        
        return x
    

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
