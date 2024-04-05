from typing import Callable, Dict, Any
import numpy as np
from scipy.linalg import solve_discrete_lyapunov


class KalmanFilter:
    """
    Assume LDS of the following form:
    z_{n} = Az_{n-1} + w_{n}
    x_{n} = Cz_{n} + v_{n}
    where w_{n} ~ N(0, \Gamma), i.i.d., and v_{n} ~ N(0, \Sigma), i.i.d.
    
    The problem is finding the posterior distribution of z_{n} given observations up to timestep n, x_{1:n}
    """
    def __init__(
        self,
        A: np.ndarray, 
        C: np.ndarray, 
        Gamma: np.ndarray, 
        Sigma: np.ndarray, 
    ):
        D = A.shape[0]
        self.A = A
        self.C = C
        self.Gamma = Gamma
        self.Sigma = Sigma
        self.mu0 = np.zeros((D, ))
        # self.Gamma0 = Gamma0
        # self.Gamma0 = np.matmul(np.linalg.inv(np.eye(D) - A), np.matmul(Gamma, np.linalg.inv(np.eye(D) - A).T))
        self.Gamma0 = solve_discrete_lyapunov(A, Gamma)
        
    def filtering(self, x: np.ndarray):
        N, D = x.shape
        
        mu, V = np.zeros((N, D)), np.zeros((N, D, D))
        
        for n in range(N):
            if n == 0:
                K = np.matmul(self.Gamma0, np.matmul(self.C.T, np.linalg.inv(
                    np.matmul(self.C, np.matmul(self.Gamma0, self.C.T)) + self.Sigma
                )))

                mu_temp = self.mu0 + np.matmul(K, x[n] - np.dot(self.C, self.mu0))
                V_temp = np.matmul(np.eye(D) - np.matmul(K, self.C), self.Gamma0)
            else:
                P = np.matmul(self.A, np.matmul(V[n-1], self.A.T)) + self.Gamma
                K = np.matmul(P, np.matmul(self.C.T, np.linalg.inv(
                    np.matmul(self.C, np.matmul(P, self.C.T)) + self.Sigma
                )))
                mu_temp = np.dot(self.A, mu[n-1]) + np.matmul(K, x[n] - np.dot(self.C, np.dot(self.A, mu[n-1])))
                V_temp = np.matmul(np.eye(D) - np.matmul(K, self.C), P)
            
            mu[n] = mu_temp
            V[n] = V_temp
        
        return mu, V


def kalman_filtering_without_control(
    num_iters: int, 
    env_cls: Callable,
    init_env_kwargs: Dict[str, Any] = {},  
    init_KF_kwargs: Dict[str, Any] = {}, 
):
    env = env_cls(**init_env_kwargs)
    kalman_filter = KalmanFilter(**init_KF_kwargs)
    
    true_state_history = []
    
    for i in range(num_iters):
        if i > 0:
            _ = env.step()
        obs = env.obs_state()
        
        true_state_history.append(env.curr_state.copy())
    
    true_state_history = np.array(true_state_history)
    
    mu, V = kalman_filter.filtering(true_state_history)
    
    return mu, V, true_state_history
