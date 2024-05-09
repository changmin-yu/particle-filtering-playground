import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from particle_filtering.particle_filtering_with_control import particle_filtering_SIR_with_control
from environment_dynamics import RobotNavigationEnvSimple
from particle_filtering.transition_kernels import RobotGaussianTransitionKernel
from particle_filtering.observation_kernels import RobotGaussianLandmarkObservationKernel
from estimate_state import estimate_mean_variance_robot_location
from evaluations import gaussian_cdf_evaluation_PF


def main(seed: int, init: str = "uniform", save_particles: bool = False, num_iters: int = 8, num_particles: int = 5000):
    np.random.seed(seed)
    
    landmarks = np.array([
        [-1, 2], 
        [5, 10], 
        [12, 14], 
        [18, 21], 
    ])
    
    resampling_method = "systematic"
    
    initial_state = np.array([0., 0.])
    
    if init == "uniform":
        init_particles_kwargs = {
            "x_range": np.array([0, 20]), 
            "y_range": np.array([0, 20]), 
            "hd_range": np.array([0, 2*np.pi]), 
        }
    elif init == "gaussian":
        init_particles_kwargs = {
            "init_x": np.concatenate([initial_state, np.array([np.pi / 4])]), 
            "std": np.array([5., 5., np.pi / 4])
        }
    else:
        raise NotImplementedError
    
    sensor_obs_std = 0.1
    init_env_kwargs = {
        "obs_std": sensor_obs_std, 
        "landmarks": landmarks, 
    }
    
    control_std = np.array([0.2, 0.05])
    prediction_kwargs = {
        "u_std": control_std, 
    }
    
    transition_kernel_kwargs = {}
    
    observation_kernel_kwargs = {
        "landmarks": landmarks, 
        "obs_std": sensor_obs_std, 
    }
    
    
    true_state_history, estimate_state_history, particle_history, weight_history = particle_filtering_SIR_with_control(
        num_particles=num_particles, 
        num_iters=num_iters, 
        init_true_state=initial_state, 
        env_cls=RobotNavigationEnvSimple, 
        transition_kernel=RobotGaussianTransitionKernel, 
        observation_kernel=RobotGaussianLandmarkObservationKernel, 
        estimate_state_fn=estimate_mean_variance_robot_location, 
        save_particles=save_particles, 
        resampling_method=resampling_method, 
        init_particles_kwargs=init_particles_kwargs, 
        init_env_kwargs=init_env_kwargs, 
        pred_kwargs=prediction_kwargs, 
        transition_kernel_kwargs=transition_kernel_kwargs, 
        observation_kernel_kwargs=observation_kernel_kwargs, 
    )
    
    return true_state_history, estimate_state_history, particle_history, weight_history


if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    true_state_history, estimate_state_history, particle_history, weight_history = main(seed=6, init="gaussian", save_particles=True, num_iters=200, num_particles=5000)
    
    particle_history_loc = np.array([arr[:, :-1] for arr in particle_history])
    true_state_history_loc = np.array(true_state_history)
    weight_history = np.array(weight_history)
    gaussian_cdf_evaluation_PF(particle_history_loc, true_state_history_loc, weight_history, dim=1)