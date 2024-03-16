import numpy as np
from scipy import stats


class ObservationKernel:
    def __init__(self):
        pass
    
    def likelihood(self, x: np.ndarray):
        return
    

class RobotGaussianLandmarkObservationKernel(ObservationKernel):
    def __init__(self, landmarks: np.ndarray, obs_std: float):
        super().__init__()
        
        self.landmarks = landmarks
        self.obs_std = obs_std
    
    def likelihood(self, x: np.ndarray, obs: np.ndarray):
        likelihood = 1.
        for i, landmark in enumerate(self.landmarks):
            distance = np.linalg.norm(x[:, :2] - landmark, axis=-1)
            likelihood *= stats.norm(distance, self.obs_std).pdf(obs[i])
        
        return likelihood


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
