import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from particle_filtering_without_control import particle_filtering_without_control
from environment_dynamics import LDSSimple
from transition_kernels import LDSTransitionKernel
from observation_kernels import LDSObservationKernel
from estimate_state import estimate_mean_variance_robot_location


def main(
    seed: int, 
    init: str = "uniform", 
    save_particles: bool = False, 
    num_iters: int = 10, 
    num_particles: int = 5000, 
):
    np.random.seed(seed)
    
    resampling_method = "systematic"
    
    theta = np.pi / 12
    A = np.array([
        [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)], 
    ])
    C = np.eye(2)
    
    Sigma = np.eye(2) * 1.0
    Gamma = np.eye(2) * 0.5
    
    mu0 = np.array([10., 0.])
    Gamma0 = np.eye(2) * 1.0
    
    if init == "uniform":
        init_particles_kwargs = {
            "x_range": [np.array([-20, 20]), np.array([-20, 20])]
        }
    elif init == "gaussian":
        init_particles_kwargs = {
            "init_x": np.array([10., 0.]), 
            "std": 5.0, 
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
    
    true_state_history, estimate_state_history, particle_history, weight_history, obs_history = particle_filtering_without_control(
        num_particles=num_particles, 
        num_iters=num_iters, 
        env_cls=LDSSimple, 
        transition_kernel=LDSTransitionKernel, 
        observation_kernel=LDSObservationKernel, 
        estimate_state_fn=estimate_mean_variance_robot_location, 
        save_particles=save_particles, 
        resampling_method=resampling_method, 
        init_particles_kwargs=init_particles_kwargs, 
        init_env_kwargs=init_env_kwargs, 
        transition_kernel_kwargs=transition_kernel_kwargs, 
        observation_kernel_kwargs=observation_kernel_kwargs, 
    )
    
    return true_state_history, estimate_state_history, particle_history, weight_history, obs_history


if __name__=="__main__":
    true_state_history, estimate_state_history, particle_history, weight_history = main(2, init="uniform")