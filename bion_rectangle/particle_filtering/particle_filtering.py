from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.distributions as distributions

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from bion_rectangle.particle_filtering.initialisation import init_particles
from bion_rectangle.particle_filtering.resampling import (
    systematic_resampling, 
    stratified_resampling, 
    effective_size, 
)


class ParticleFiltering:
    def __init__(
        self, 
        num_particles: int, 
        transition_kernel: Any, 
        observation_kernel: Any, 
        generative_model, 
        resampling_method: str = "systematic", 
        resampling_strategy: str = "SIR", 
        effective_size_threshold: float = 0.5, 
        init_method: str = "uniform", 
        transition_kernel_kwargs: Dict[str, Any] = {}, 
        observation_kernel_kwargs: Dict[str, Any] = {}, 
        resampling_loc_std: float = 0.0,
        resampling_hd_std: float = 0.0, 
    ):
        self.num_particles = num_particles
        self.transition_kernel = transition_kernel(**transition_kernel_kwargs)
        self.observation_kernel = observation_kernel(**observation_kernel_kwargs)
        
        self.resampling_method = resampling_method
        self.resampling_strategy = resampling_strategy
        self.effective_size_threshold = effective_size_threshold
        self.init_method = init_method
        
        self.resampling_loc_std = resampling_loc_std
        self.resampling_hd_std = resampling_hd_std
        
        self.init_particles(generative_model)
        
    def step(
        self, 
        generative_model, 
        control, 
        noise_control_rotate_pconc, 
        noise_control_shift_per_speed, 
        i_states_belief: Optional[torch.Tensor] = None, 
    ):
        self.predict(control, noise_control_rotate_pconc, noise_control_shift_per_speed)
        self.update(generative_model, i_states_belief)
        
        if self.resampling_strategy == "SIR":
            if effective_size(self.w) < self.effective_size_threshold * self.num_particles:
                self.resampling()
        else:
            self.resampling()
        
        particle_mean, particle_var = self.estimate_posterior()
        
        return particle_mean, particle_var
    
    def init_particles(self, generative_model):
        if self.init_method == "uniform":
            uniform_dist = distributions.uniform.Uniform(torch.tensor([-generative_model.env.x_max, -generative_model.env.y_max, -180]),
                                                            torch.tensor([generative_model.env.x_max, generative_model.env.y_max, 180]))
            samples = uniform_dist.rsample(torch.Size([self.num_particles]))
            accepted_samples = samples[generative_model.is_inside(samples.numpy()[:, :2])]
            while accepted_samples.shape[0] < self.num_particles:
                samples = uniform_dist.sample(torch.Size([self.num_particles - accepted_samples.shape[0]]))
                accepted_samples = torch.cat(
                    [accepted_samples,
                        samples[generative_model.is_inside(samples.numpy()[:, :2])]],
                    0)
            
            self.particles = accepted_samples
            self.w = torch.full([self.num_particles, ], 1 / self.num_particles)
        else:
            raise NotImplementedError
    
    def predict(self, control, noise_control_rotate_pconc, noise_control_shift_per_speed):
        locs, headings = self.transition_kernel.predict(
            self.particles, 
            control, 
            noise_control_rotate_pconc, 
            noise_control_shift_per_speed, 
        )
        self.particles = torch.cat([locs, headings[:, None]], dim=1)
    
    def update(self, generative_model, i_states_belief: Optional[torch.Tensor] = None):
        ll = self.observation_kernel.log_likelihood(self.particles, generative_model, i_states_belief)
        
        w_log = torch.log(self.w)
        w_log_new = w_log + ll
        w_log_new = w_log_new - torch.max(w_log_new)
        w_new = torch.exp(w_log_new)
        self.w = w_new / torch.sum(w_new)
    
    def resampling(self):
        if self.resampling_method == "systematic":
            inds = systematic_resampling(self.w)
        elif self.resampling_method == "stratified":
            inds = stratified_resampling(self.w)
        else:
            raise NotImplementedError(f"{self.resampling_method} resampling is not implemented!")

        # inds = torch.from_numpy(inds)
        
        self.particles = self.particles[inds]
        
        if self.resampling_loc_std > 0:
            loc = self.particles[..., :-1]
            loc_jitter = loc + torch.normal(0, self.resampling_loc_std, size=loc.size())
            
            jitter_trajectory = torch.stack([loc, loc_jitter], dim=-1)
            intersections = self.transition_kernel.intersection_lines_vvec(jitter_trajectory)
            loc = torch.where(
                torch.isinf(intersections).any(-1, keepdim=True),
                loc_jitter,
                0.05 * loc + 0.95 * intersections)
            self.particles[..., :-1] = loc
        if self.resampling_hd_std > 0:
            self.particles[..., -1] += torch.normal(0, std=self.resampling_hd_std, size=self.particles[..., -1].size())
        
        self.w = torch.ones_like(self.w) / len(self.w)
    
    def estimate_posterior(self):
        loc = self.particles[:, :-1]
        mean = torch.mean(loc * self.w[:, None], axis=0)
        var = torch.mean(torch.square(loc - mean) * self.w[:, None], axis=0)
        
        return mean, var
