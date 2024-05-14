import torch
from torch import distributions

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from bion_rectangle.utils import np2


class Particles(np2.MonitorChange):

    key_states = ('loc', 'weight')

    def __init__(self, loc: torch.Tensor, weight: torch.Tensor):
        assert loc.shape[0] == weight.shape[0]
        self.loc = loc
        self.weight = weight
        super().__init__(self.key_states)

    @property
    def n_particle(self):
        return self.loc.size()[0]

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value
        return

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, value):
        self._loc = value
        return

    def resample(self, update=True, MH=False):
        """
        Resample the particles in the continuous state space
        :return:
        """
        def sample(normalize=True):
            sample_indices = torch.distributions.Categorical(probs=self.weight
                                                             ).sample(torch.Size([self.n_particle]))
            loc = self.loc[sample_indices, :]
            weight = self.weight[sample_indices]
            return loc, weight

        if MH:
            res_loc1, res_weight1 = sample()
            res_loc2, res_weight2 = sample()
            condition = (distributions.Uniform(0, 1).sample(res_weight1.size()) >=
                         torch.minimum(torch.ones_like(res_weight1), res_weight1 / res_weight2))
            loc = torch.where(torch.stack(3*[condition], dim=-1), res_loc1, res_loc2)
        else:
            loc = sample()[0]

        weight = torch.full([self.n_particle, ], 1 / self.n_particle)
        if update:
            self.loc = loc
            self.weight = weight
        return loc, weight

    def downsample(self, n_particle):
        """
        Downsample the particles
        :param n_particle:
        :return:
        """
        if n_particle >= self.n_particle:
            return
        downsample_indices = torch.sort(self.weight, dim=0, descending=True)[1][:n_particle]
        self.loc = self.loc[downsample_indices, :]
        self.weight = self.weight[downsample_indices]
        return

    def jitter(self):
        """
        Add small noise to the particles
        :return:
        """
        new_loc = self.loc.clone()
        new_loc[:, :2] += torch.normal(mean=0, std=0.03, size=self.loc[:, :2].size())
        new_loc[:, 2] += torch.normal(mean=0, std=2, size=self.loc[:, 2].size())
        return new_loc


"""
class ParticlesDiscrete(Particle):

    def __init__(self, n_particle):
        super().__init__(n_particle)


class ParticlesDiscreteHist(ParticleDiscrete):

    def __init__(self, statetab: StateTable):
        n_particle = statetab.n_state**2 * statetab.n_heading
        super().__init__(n_particle)
"""