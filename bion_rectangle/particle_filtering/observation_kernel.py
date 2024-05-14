from typing import Optional

import numpy as np
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from bion_rectangle.utils.numpytorch import indexshape


class BionObservationKernel:
    def __init__(
        self, 
        duration,  
    ):
        self.duration = duration
        
    def log_likelihood(self, x, generative_model, i_states_belief: Optional[torch.Tensor] = None):
        ll = self.get_log_likelihood_retina_vectorised(x, generative_model, i_states_belief)[0]
        return ll
    
    def get_log_likelihood_retina_vectorised(
        self, 
        x, 
        generative_model, 
        i_states_belief: Optional[torch.Tensor] = None
    ):
        # NOTE: potential over/underflow with duration >= 4 ?
        # assert duration < 4, 'potential over/underflow with duration >= 4'
        rate_imgs = generative_model.measure_retinal_image(generative_model.agent_state)[None, :]

        if i_states_belief is None:
            return (
                    torch.sum(
                        # [state_true_batch, (state_belief), x, y, chn]
                        rate_imgs.unsqueeze(1)
                        #
                        # [1, state_belief, x, y, chn]
                        * (generative_model.log_img_given_state(x)[None, :]
                           + np.log(self.duration)),
                        [-3, -2, -1]
                    ) - generative_model.get_sum_img_given_state(x)[None, :]  # [1, state_belief]
            ) * self.duration
        elif i_states_belief.ndim == 1:
            return (
                    torch.sum(
                        # [state_true_batch, x, y, chn]
                        rate_imgs
                        #
                        # [state_beliefs_to_consider, x, y, chn]
                        * (generative_model.log_img_given_state(x)[i_states_belief]
                           + np.log(self.duration)),
                        [-3, -2, -1]
                    ) - generative_model.get_sum_img_given_state(x)[i_states_belief]  # [state_belief]
            ) * self.duration
        elif i_states_belief.ndim == 2:
            return (
                    torch.sum(
                        # [state_true_batch, x, y, chn]
                        rate_imgs[:, None, :]
                        #
                        # [state_true_batch, state_beliefs_to_consider, x, y, chn]
                        * (indexshape(generative_model.log_img_given_state(x), i_states_belief)
                           + np.log(self.duration)),
                        [-3, -2, -1]
                    ) - indexshape(generative_model.get_sum_img_given_state(x), i_states_belief)
            ) * self.duration
        else:
            raise ValueError()
