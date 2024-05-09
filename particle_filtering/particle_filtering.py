from typing import Dict, Any, Optional
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from particle_filtering.initialisation import init_particles
from utils.general_utils import estimate_mean_variance_robot_location
from particle_filtering.resampling import (
    systematic_resampling, 
    stratified_resampling, 
    resample_from_inds, 
    effective_size, 
)

JITTER = 1e-20


class ParticleFiltering:
    def __init__(
        self, 
        num_particles: int, 
        transition_kernel: Any, 
        observation_kernel: Any, 
        resampling_method: str = "systematic", 
        resampling_strategy: str = "SIR", 
        effective_size_threshold: float = 0.5, 
        init_particles_kwargs: Dict[str, Any] = {}, 
        transition_kernel_kwargs: Dict[str, Any] = {}, 
        observation_kernel_kwargs: Dict[str, Any] = {}, 
    ):
        self.num_particles = num_particles
        self.transition_kernel = transition_kernel(**transition_kernel_kwargs)
        self.observation_kernel = observation_kernel(**observation_kernel_kwargs)
        
        self.resampling_method = resampling_method
        self.resampling_strategy = resampling_strategy
        self.effective_size_threshold = effective_size_threshold
        
        self.particles = init_particles(num_particles, **init_particles_kwargs)
        self.w = np.ones((num_particles, )) / num_particles

    def step(self, obs: np.ndarray, u: Optional[np.ndarray]=None):
        self.predict(u)
        self.update(obs)
        
        if self.resampling_strategy == "SIR":
            if effective_size(self.w) < self.effective_size_threshold * self.num_particles:
                self.resampling()
        else:
            self.resampling()
        
        particle_mean, particle_std = self.estimate()
        
        return particle_mean, particle_std
        
    def predict(self, u: Optional[np.ndarray] = None):
        if u is not None:
            self.particles = self.transition_kernel.step(self.particles, u)
        else:
            self.particles = self.transition_kernel.step(self.particles)
            
    def update(self, obs: np.ndarray):
        obs_likelihood = self.observation_kernel.likelihood(self.particles, obs)
        
        self.w *= obs_likelihood
        self.w += JITTER
        self.w /= np.sum(self.w)
        
    def resampling(self):
        if self.resampling_method == "systematic":
            inds = systematic_resampling(self.w)
        elif self.resampling_method == "stratified":
            inds = stratified_resampling(self.w)
        else:
            raise NotImplementedError(f"{self.resampling_method} resampling is not implemented!")
        
        self.particles = self.particles[inds]
        self.w = np.ones_like(self.w) / len(self.w)
        
    def estimate(self):
        return estimate_mean_variance_robot_location(self.particles, self.w)
    