from typing import Callable, Dict, Any, Optional, List

import numpy as np

from resampling import effective_size, resampling
from transition_kernels import TransitionKernel
from observation_kernels import ObservationKernel
from environment_dynamics import BaseEnv


def particle_filtering_without_control(
    num_particles: int, 
    num_iters: int, 
    env_cls: Callable, 
    transition_kernel: Callable, 
    observation_kernel: Callable, 
    estimate_state_fn: Callable, 
    save_particles: bool = False, 
    resampling_method: str = "systematic", 
    init_particles_kwargs: Dict[str, Any] = {},
    init_env_kwargs: Dict[str, Any] = {},  
    pred_kwargs: Dict[str, Any] = {},
    transition_kernel_kwargs: Dict[str, Any] = {}, 
    observation_kernel_kwargs: Dict[str, Any] = {}, 
):
    particles = init_particles(num=num_particles, **init_particles_kwargs)
    w = np.ones(num_particles) / num_particles
    
    true_state_history = []
    estimate_state_history = []
    particle_history = []
    weight_history = []
    obs_history = []
    
    env = env_cls(**init_env_kwargs)
    trans_kernel = transition_kernel(**transition_kernel_kwargs)
    obs_kernel = observation_kernel(**observation_kernel_kwargs)
    
    for i in range(num_iters):
        if i > 0:
            _ = env.step()
        obs = env.obs_state()
        
        particles = predict(particles, trans_kernel, transition_kwargs=pred_kwargs)
        w = update(particles, w, obs_kernel, obs)
        
        if effective_size(w) < num_particles / 2:
            particles, w = resampling(particles, w, resampling_method)
            assert np.allclose(w, 1. / num_particles)
        
        estimated_state, _ = estimate_state_fn(particles, w)
        
        true_state_history.append(env.curr_state.copy())
        estimate_state_history.append(estimated_state)
        weight_history.append(w.copy())
        obs_history.append(obs.copy())
        
        if save_particles:
            particle_history.append(particles.copy())
    
    return true_state_history, estimate_state_history, particle_history, weight_history, obs_history


def predict(
    particles: np.ndarray, 
    transition_kernel: TransitionKernel, 
    transition_kwargs: Dict[str, Any] = {}
):
    particles = transition_kernel.step(particles, **transition_kwargs)
    
    return particles


def update(
    particles: np.ndarray, 
    w: np.ndarray, 
    observation_kernel: ObservationKernel, 
    obs: np.ndarray, 
):
    obs_likelihood = observation_kernel.likelihood(particles, obs)
    
    w = obs_likelihood
    w += 1e-300
    w /= np.sum(w)
    
    return w


def estimate(
    particles: np.ndarray, 
    w: np.ndarray, 
    estimate_fn: Callable, 
):
    return estimate_fn(particles, w)


def init_particles(
    num: int, 
    init_x: Optional[np.ndarray] = None, 
    **kwargs, 
):
    if init_x is None:
        particles = init_particles_uniform(num=num, **kwargs)
    else:
        particles = init_particles_gaussian(mean=init_x, num=num, **kwargs)
    return particles


def init_particles_uniform(
    x_range: List[np.ndarray],  
    num: int, 
):
    """
    Initialising particles uniformly, we assume only three latent variables (x and y location, and head direction)
    """
    D = len(x_range)
    particles = np.zeros((num, D))
    for d in range(D):
        particles[:, d] = np.random.uniform(x_range[0][0], x_range[0][1], size=(num, ))
    
    return particles


def init_particles_gaussian(
    mean: np.ndarray, 
    cov: np.ndarray, 
    num: int
):
    D = mean.shape[0]
    particles = np.random.multivariate_normal(mean, cov, size=(num, ))

    return particles
