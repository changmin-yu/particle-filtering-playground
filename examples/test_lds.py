from typing import Optional

import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from particle_filtering.particle_filtering import ParticleFiltering
from environment.gaussian_lds import LDSSimple
from transition_kernels.gaussian_lds import LDSTransitionKernel
from observation_kernels.gaussian_lds import LDSObservationKernel


def main(
    seed: Optional[int] = None, 
    init: str = "uniform", 
    save_particles: bool = False, 
    num_iters: int = 10, 
    num_particles: int = 5000, 
    mu0: Optional[np.ndarray] = None, 
    Gamma0: Optional[np.ndarray] = None,
    resampling_method = "systematic", 
):
    if seed is not None:
        np.random.seed(seed)
    
    theta = np.pi / 12
    A = 0.99 * np.array([
        [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)], 
    ])
    C = np.eye(2)
    
    Sigma = np.eye(2) * 1.0
    Gamma = np.eye(2) * 0.5
    # Gamma0 = np.matmul(np.linalg.inv(np.eye(D) - A), np.matmul(Gamma, np.linalg.inv(np.eye(D) - A).T))
    
    if init == "uniform":
        init_particles_kwargs = {
            "lims": [np.array([-20, 20]), np.array([-20, 20])]
        }
    elif init == "gaussian":
        init_particles_kwargs = {
            "init_x": np.array([0., 0.]), 
            # "cov": np.matmul(np.linalg.inv(np.eye(D) - A), np.matmul(Gamma, np.linalg.inv(np.eye(D) - A).T))
            "cov": solve_discrete_lyapunov(A, Gamma)
        }
    else:
        raise NotImplementedError
    
    init_env_kwargs = {
        "A": A, 
        "C": C, 
        "Sigma": Sigma, 
        "Gamma": Gamma, 
        "mu0": mu0, 
        "Gamma0": Gamma0, 
    }
    
    transition_kernel_kwargs = {
        "A": A, 
        "Gamma": Gamma, 
    }
    
    observation_kernel_kwargs = {
        "C": C, 
        "Sigma": Sigma, 
    }
    
    particle_filter = ParticleFiltering(
        num_particles=num_particles, 
        transition_kernel=LDSTransitionKernel, 
        observation_kernel=LDSObservationKernel, 
        resampling_method=resampling_method, 
        init_particles_kwargs=init_particles_kwargs, 
        transition_kernel_kwargs=transition_kernel_kwargs, 
        observation_kernel_kwargs=observation_kernel_kwargs, 
    )
    
    env = LDSSimple(**init_env_kwargs)
    
    true_state_history = np.zeros((num_iters, 2))
    particle_mean_history = np.zeros((num_iters, 2))
    particle_std_history = np.zeros((num_iters, 2))
    particle_history = np.zeros((num_iters, num_particles, 2))
    weight_history = np.zeros((num_iters, num_particles))
    obs_history = np.zeros((num_iters, 2))
    
    for i in range(num_iters):
        if i > 0:
            _ = env.step()
        obs = env.obs_state()
        
        particle_mean, particle_std = particle_filter.step(obs)
        
        true_state_history[i] = env.curr_state
        particle_mean_history[i] = particle_mean
        particle_std_history[i] = particle_std
        obs_history[i] = obs
        if save_particles:
            particle_history[i] = particle_filter.particles
            weight_history[i] = particle_filter.w
    
    return (
        true_state_history, 
        particle_mean_history, 
        particle_std_history, 
        particle_history, 
        weight_history, 
        obs_history, 
    )


if __name__=="__main__":
    (
        true_state_history, 
        particle_mean_history, 
        particle_std_history, 
        particle_history, 
        weight_history, 
        obs_history, 
    ) = main(2, init="gaussian")