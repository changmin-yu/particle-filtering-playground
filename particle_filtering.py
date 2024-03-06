from typing import Callable, Dict, Any

import numpy as np

from transition_kernels import TransitionKernel
from observation_kernels import ObservationKernel
from environment_dynamics import BaseEnv
from initialisation import init_particles
from resampling import effective_size, resampling


def particle_filtering_SIR(
    num_particles: int, 
    num_iters: int, 
    init_true_state: np.ndarray, 
    env_cls: BaseEnv, 
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
    """
    Particle filtering based on Sampling-Importance-Resampling (SIR) framework.
    
    Parameters
    ----------
    
    num_particles: int
        number of particles
    num_iters: int
        number of iterations
    init_true_state: np.ndarray
        initial ground-truth latent state
    env_cls: BaseEnv
        environment
    transition_kernel: Callable
        transition kernel
    observation_kernel: Callable
        observation kernel
    estimate_state_fn: Callable
        estimate state (statistics) function
    save_particles: bool
        save particle history
    resampling_method: str
        resampling method
    init_particles_kwargs: Dict[str, Any]
        kwargs for initialising particles
    init_env_kwargs: Dict[str, Any]
        kwargs for initialising environment
    pred_kwargs: Dict[str, Any]
        kwargs for prediction step
    transition_kernel_kwargs: Dict[str, Any]
        kwargs for transition kernel
    observation_kernel_kwargs: Dict[str, Any]
        kwargs for observation kernel
    """
    
    particles = init_particles(num=num_particles, **init_particles_kwargs)
    w = np.ones(num_particles) / num_particles
    
    true_state_history = []
    estimate_state_history = []
    particle_history = []
    
    env = env_cls(init_true_state, **init_env_kwargs)
    trans_kernel = transition_kernel(**transition_kernel_kwargs)
    obs_kernel = observation_kernel(**observation_kernel_kwargs)
    
    for i in range(num_iters):
        u = env.step()
        obs = env.obs_state().copy()
        
        particles = predict(particles, trans_kernel, u=u, dt=1, transition_kwargs=pred_kwargs)
        w = update(particles, w, obs_kernel, obs)
        
        if effective_size(w) < num_particles / 2:
            particles, w = resampling(particles, w, resampling_method)
            assert np.allclose(w, 1. / num_particles)
        
        estimated_state = estimate_state_fn(particles, w)
        
        true_state_history.append(env.state.copy())
        estimate_state_history.append(estimated_state)
        
        if save_particles:
            particle_history.append(particles.copy())
    
    return true_state_history, estimate_state_history, particle_history
    

def predict(
    particles: np.ndarray, 
    transition_kernel: TransitionKernel, 
    u: np.ndarray, 
    dt: float=1., 
    transition_kwargs: Dict[str, Any] = {}, 
):
    particles = transition_kernel.step(particles, u, dt=dt, **transition_kwargs)
    
    return particles


def update(
    particles: np.ndarray, 
    w: np.ndarray, 
    observation_kernel: ObservationKernel, 
    obs: np.ndarray, 
):
    obs_likelihood = observation_kernel.likelihood(particles, obs, w)
    
    w *= obs_likelihood
    w += 1e-300
    w /= np.sum(w)
    
    return w


def estimate(
    particles: np.ndarray, 
    w: np.ndarray, 
    estimate_fn: Callable, 
):
    return estimate_fn(particles, w)
