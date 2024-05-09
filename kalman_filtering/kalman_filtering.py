from typing import Callable, Dict, Any, Optional
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
        mu0: Optional[np.ndarray] = None, 
        Gamma0: Optional[np.ndarray] = None, 
    ):
        D = A.shape[0]
        self.A = A
        self.C = C
        self.Gamma = Gamma
        self.Sigma = Sigma
        
        if mu0 is None:
            mu0 = np.zeros((D, ))
        if Gamma0 is None:
            Gamma0 = solve_discrete_lyapunov(A, Gamma)
        self.mu0 = mu0
        self.Gamma0 = Gamma0
        
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
    

class KalmanFilter_v2:
    """
    Computing filtering posterior with Kalman filtering without explicitly computing the Kalman gain.
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
        self.Gamma0 = solve_discrete_lyapunov(A, Gamma)
        
    def filtering(self, x: np.ndarray):
        N, D = x.shape
        
        mu, V = np.zeros((N+1, D)), np.zeros((N+1, D, D))
        
        mu[0] = self.mu0.copy()
        V[0] = self.Gamma0.copy()
        
        Sigma_inv = np.linalg.inv(self.Sigma.copy())
        
        for n in range(N):
            mu_pred = np.dot(self.A, mu[n].copy())
            cov_pred = np.matmul(self.A, np.matmul(V[n].copy(), self.A.T)) + self.Gamma
            
            cov_filter = np.linalg.inv(
                np.linalg.inv(cov_pred) + 
                np.matmul(self.C.T, np.matmul(Sigma_inv, self.C))
            )
            mu_filter = np.matmul(
                cov_filter, 
                np.linalg.inv(cov_pred).dot(mu_pred) + np.dot(self.C.T, np.dot(Sigma_inv, x[n]))
                # np.dot(mu_pred, np.linalg.inv(cov_pred)) + np.dot(np.dot(x[n], self.C.T), Sigma_inv)
            )
            
            mu[n+1] = mu_filter.copy()
            V[n+1] = cov_filter.copy()
        
        return mu[1:], V[1:]


def kalman_filtering_without_control(
    num_iters: int, 
    env_cls: Callable,
    init_env_kwargs: Dict[str, Any] = {},  
    init_KF_kwargs: Dict[str, Any] = {}, 
    use_v2: bool = False, 
):
    env = env_cls(**init_env_kwargs)
    if use_v2:
        kalman_filter = KalmanFilter_v2(**init_KF_kwargs)
    else:
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
