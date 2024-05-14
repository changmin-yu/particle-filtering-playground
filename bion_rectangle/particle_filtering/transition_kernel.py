import numpy as np
import torch
from torch.distributions import VonMises, Gamma, Normal

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from bion_rectangle.utils.np2 import (
    pconc2conc, 
    deg2rad, 
)


class BionTransitionKernel:
    def __init__(self, dt: float, corners: np.ndarray):
        self.dt = dt
        self.corners = corners
    
    def predict(
        self, 
        x: torch.Tensor, 
        control, 
        noise_control_rotate_pconc, 
        noise_control_shift_per_speed, 
    ):
        n_particles = len(x) # TODO: check this
        locs = x[..., :-1]
        headings = x[..., -1]
        
        control_heading = control.dheading_deg[..., -1]
        
        headings_new = headings + torch.rad2deg(
            VonMises(
                torch.deg2rad(control_heading * self.dt), 
                pconc2conc(noise_control_rotate_pconc) / (torch.abs(torch.deg2rad(control_heading)) * self.dt)
            ).sample(torch.Size([n_particles]))
        )
        
        rate = control.velocity_ego[0] / (
            (self.dt * control.velocity_ego[0] * noise_control_shift_per_speed[0]) ** 2
        )
        conc = control.velocity_ego[0] * rate
        
        rotation = torch.stack([
            torch.stack([torch.cos(deg2rad(headings_new)), torch.sin(deg2rad(headings_new))], -1),
            torch.stack([-torch.sin(deg2rad(headings_new)), torch.cos(deg2rad(headings_new))], -1)
        ], -1)
        
        locs_new = locs + (rotation @ torch.stack([
            Gamma(conc, rate).rsample([n_particles]), 
            Normal(0, (self.dt * control.velocity_ego[0] * noise_control_shift_per_speed[1]) ** 2).rsample([n_particles])
        ], -1)[..., None])[..., 0]
        
        trajectories = torch.stack([locs, locs_new], 1)
        intersections = self.intersection_lines_vvec(trajectories)
        locs = torch.where(
            torch.isinf(intersections).any(-1, keepdim=True), 
            locs_new, 
            0.05 * locs + 0.95 * intersections
        )
        
        return locs, headings_new
        
    def intersection_lines_vvec(self, lines: torch.Tensor):
        num_walls = self.corners.shape[0]
        num_lines = lines.size()[0]
        corners = torch.tensor(self.corners)
        # Contains the start and end points of each wall
        # The matrix in the first pos in torch.tensordot transforms
        # all the corners into the start and end points of the walls
        walls = torch.tensordot(torch.stack([torch.eye(num_walls),
                                             torch.roll(torch.eye(num_walls), 1, dims=1)], -1),
                                corners[:, :2], dims=([1], [0]))
        # --- Implementation of Cramer's rule for an intersecting line
        diff = torch.tensor([1., -1.])
        # ['num_lines', 'num_walls'] lots of [2, 2] matrices containing the difference between
        # the start and end points of the line (col 0) and the walls (col 1)
        full_diff = torch.stack([torch.stack(walls.size()[0] * [diff @ lines], dim=1),
                                 torch.stack(num_lines * [diff @ walls])], -1)
        # Determinants of the 'num_walls' [2, 2] difference matrices
        # If == 0 then the lines are parallel and do not intersect
        # div = torch.det(full_diff)
        # ['num_lines', 'num_walls', 2] matrix containing the determinant of the line and a wall
        d = torch.stack([torch.stack(walls.size()[0] * [torch.det(lines)], dim=1),
                         torch.stack(num_lines * [torch.det(walls)])], -1)
        # Contains the intersection points. If no intersection, the point is (-inf, -inf)
        final = (torch.stack([torch.det(torch.stack([d, full_diff[:, :, 0, :]], -1)),
                              torch.det(torch.stack([d, full_diff[:, :, 1, :]], -1))], -1) /
                 torch.stack(2 * [torch.det(full_diff)], -1))
        # Conditions that ensure the intersection point on the line segment and the wall segment;
        # Cramer's rule is for infinite lines, so we need to check this.
        # Condition: (1) line segments maximum point, (2) line segments minimum point,
        # (3) wall segments maximum point, (4) wall segments minimum point
        condition = ((final <= torch.stack(walls.size()[0] * [torch.maximum(lines[:, 0], lines[:, 1])], dim=1)) &
                     (final >= torch.stack(walls.size()[0] * [torch.minimum(lines[:, 0], lines[:, 1])], dim=1)) &
                     (final <= torch.maximum(walls[:, 0, :], walls[:, 1, :]) + 1e-6) &
                     (final >= torch.minimum(walls[:, 0, :], walls[:, 1, :]) - 1e-6))
        condition = condition[:, :, 0] & condition[:, :, 1]
        final = torch.where(torch.stack([condition, condition], -1), final, torch.inf)
        final_diff = final - torch.stack(walls.size()[0] * [lines[:, 0]], dim=1)

        return final.reshape(-1, 2)[torch.argmin(torch.norm(final_diff, dim=2), dim=1) + walls.size()[0]
                                    * torch.arange(0, num_lines)]
    