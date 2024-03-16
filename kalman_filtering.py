import numpy as np


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
        mu0: np.ndarray, 
        Gamma0: np.ndarray, 
    ):
        self.A = A
        self.C = C
        self.Gamma = Gamma
        self.Sigma = Sigma
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
