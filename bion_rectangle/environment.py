"""
Use a continuous representation of the self location & orientation,
and pixel (bitmap) representation of the retina
to model boundaries
"""

import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from typing import Union, Dict, Optional, Any, Tuple
from scipy import ndimage

import torch
from torch import distributions, Tensor
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from bion_rectangle.utils.np2 import (
    permute2st, 
    shorten_dict, 
    npy, 
    sumto1, 
    pconc2conc, 
    issimilar, 
    joinformat, 
    deg2rad, 
    dictlist2listdict, 
)
from bion_rectangle.utils.np2 import AliasStr as Short
from bion_rectangle.utils.plt2 import (
    GridAxes, 
    box_off, 
    savefig, 
)
from bion_rectangle.utils.yktorch import BoundedParameter
from bion_rectangle.utils.numpytorch import tensor as npt_tensor
from bion_rectangle.utils.numpytorch import ones as npt_ones
from bion_rectangle.utils.numpytorch import zeros as npt_zeros
from bion_rectangle.utils.numpytorch import zeros_like as npt_zeros_like
from bion_rectangle.utils.numpytorch import eye as npt_eye
from bion_rectangle.utils.numpytorch import arange as npt_arange
from bion_rectangle.utils.numpytorch import rand as npt_rand
from bion_rectangle.utils.numpytorch import randint as npt_randint
from bion_rectangle.utils.numpytorch import (
    sumto1, 
    vmpdf_prad_pconc, 
    set_device, 
    deg2rad, 
    rad2deg, 
    expand_batch, 
    mvnrnd, 
    normrnd, 
    permute2en, 
    permute2st, 
    vmpdf_a_given_b, 
    issimilar, 
    mvnpdf_log, 
    nan2v, 
    pconc2var, 
    log_normpdf, 
    categrnd, 
    empty_like, 
    gamma_logpdf_ms, 
    prod_sumto1, 
    aggregate, 
    indexshape, 
    npy, 
    npys, 
    p2st, 
)
from bion_rectangle.utils.localfile import (
    LocalFile, 
    mkdir4file, 
    Cache, 
)
from bion_rectangle.utils.mplstyle import kw_legend

from bion_rectangle.behav.projection_3D import ProjectiveGeometry3D
from bion_rectangle.behav.geometry import (
    rotation_matrix, 
    get_unit_vec, 
)
from bion_rectangle.behav.env_boundary import EnvBoundary, EnvSpace

from bion_rectangle.behav.slam_pixel_grid import (
    AgentState, 
    TactileSensor, 
    StateTable, 
    GenerativeModel, 
    Control, 
    Measurement, 
    Observer, 
    prad2unitvec, 
    render_walls, 
    Retina, 
    TactileSensorTabularBinary, 
)
from bion_rectangle.behav.particles import Particles


# from nav_uncertainty_util.behav.particles import Particles

device0 = torch.device(
    # 'cuda:0' if torch.cuda.is_available() else
    'cpu'
)
set_device(device0)

geom = ProjectiveGeometry3D()
eps = 1e-6
locfile = LocalFile('../Data_SLAM/slam', kind2subdir=True)


class EnvTable(StateTable, EnvSpace):
    """
   Summary:
   ------------
   StateTable + access to attribute of environment.

   Description:
   ------------
   A StateTable that also allows one to access the environment attributes. The addition of the abstract class EnvSpace
   is present as a requirement for ObserverNav to utilise the continuous state space.

   Attributes:
   ------------
   - Attributes of StateTable
   - Attributes of env:
         - x_max
         - dx
         - dy
         - ddeg
         - to_use_inner
         - kwargs

   Methods:
   ------------
   - __getattr__ -> Attributes of the env

   """

    def __init__(self, env, **kwargs):
        StateTable.__init__(self, env, **kwargs)

    def __getattr__(self, item):
        if item == 'env':
            return None
        else:
            return getattr(self.env, item)


class TactileSensorBinary(TactileSensor):
    # TODO: convert to continuous case and ensure it works as a placeholder (Not required)).
    # Currently has not been tested. Furthermore will require revisit once the particles structure is determined
    """
    Summary:
    -----------
    Operates TactileSensor on the current environment.

    Description:
    ------------
    Inherits the TactileSensor class and accesses the EnvTable in order to carry out sense walls for particles and
    the resulting changes to their beliefs.

    Attributes:
    -----------
    - envtab
    - touch_range
    - touch_reliability
    - tactile_input
    - p_touch

    Methods:
    -------------
    - clone
    - get_dict_param
    - sense_vec
    - log_prob
    """

    def __init__(
            self,
            env: EnvBoundary, particles: torch.Tensor,
            touch_range=0., touch_reliability=0.,
    ):
        """
        :param env:
        :param particles:
        :param touch_range: defaults to 0 = disabled
        :param touch_reliability: between 0 (default: totally unreliable)
            and 1. (completely reliable)
        """
        super().__init__()
        self.env = env
        self.touch_range = touch_range
        self.touch_reliability = touch_reliability

        assert touch_reliability == 0.

        self.tactile_input = npt_zeros(particles.shape[0])

        i_states = npt_arange(particles.shape[0])

        # xy_src[state, 1, xy]
        xy_src = particles[:, :2]

        # heading_src[state]
        heading_src = particles[:, 2]

        # dheadings[side]
        dheadings = particles[:, 2]

        # heading_dst[state, side]
        heading_dst = heading_src[:, None] + dheadings[None, :]

        # xy_dst[state, side, xy]
        xy_dst = xy_src + torch.stack([
            torch.cos(deg2rad(heading_dst)),
            torch.sin(deg2rad(heading_dst))
        ], -1) * self.touch_range

        # in_touch[state, side]
        in_touch = npt_tensor(self.env.is_crossing_boundary(
            *npys(xy_src, xy_dst))).float()

        # p_touch1[state, side]
        p_touch1 = 1 / 2 + (self.touch_reliability * (in_touch - 0.5))

        # p_touch[state, side, touched]
        self.p_touch = torch.stack([1. - p_touch1, p_touch1], -1)

    def clone(self, **kwargs) -> 'Union[TactileSensorBinary, TactileSensorTabularBinary]':
        return type(self)(**{**self.get_kw_params(), **kwargs})

    def get_dict_param(self) -> Dict[str, Union[str, float]]:
        """For saving parameters to a human-readable table"""
        return {
            'touch range (m)': self.touch_range,
            'touch reliability (probability)': self.touch_reliability,
        }

    def get_kw_params(self) -> dict:
        """For cloning."""
        return {
            'env': self.env,
            'touch_range': self.touch_range,
            'touch_reliability': self.touch_reliability}

    @staticmethod
    def sense_vec(
            particles: torch.Tensor
    ) -> torch.Tensor:
        """
        :param particles:
        :return: touched[batch, side]
        """
        return npt_zeros(particles.shape[0], particles.shape[0])

    @staticmethod
    def log_prob(
            particles: torch.Tensor,
            return_by_side=False,
    ) -> torch.Tensor:
        """
        ...: e.g., [batch, particle_belief]
        :param particles:
        :param return_by_side
        :return: log_prob[...]
        """
        # loglik_touch[..., side]
        loglik_touch = npt_ones(particles.shape[0], particles.shape[0]).log()

        if return_by_side:
            return loglik_touch
        else:
            # pool across sides
            return loglik_touch.sum(-1)


class GenerativeModelSingleEnv(GenerativeModel):
    """
    Summary:
    -----------
    Generative model for a given environment.

    Description:
    -------------
    This class implements the  GenerativeModel for a given environment, with some parameters identical to those used
    in experiments

    Attributes:
    --------------
    - dt   PAPERPARAM
    - geom
    - n_dim
    - gain_retina
    - env
    - radius_corner
    - tactile_sensor
    - self.img_given_state
    - inertia_velocity_ego
    - inertia_heading
    - Transition & observation noise
        - noise_heading
        - noise_velocity_ego
        - noise_tangential
        - blur_retina
        - noise_pixel
        - noise_obs_dloc
        - noise_obs_dheading
        - noise_obs_dvelocity_ego
    - Observations
        - render_type
        - render_noiseless_retina
    - meas  Measurement class from slam_pixel_grid.py
    - dcorners_deg
    - dist
    - apparent_radius_deg

    Methods:
    -------------
    - fill_control_heading  @StaticMethod
    - transition
    - measure
    - measure_retinal_mage
    """

    def __init__(
            self,
            env: EnvBoundary,
            radius_corner=1.,
            self_loc_xy0=(0., 0.),
            heading0=(1., 0.),
            velocity_ego0=5.,
            self_height0=None,  # 1.8,  # Bellmund et al. 2019
            inertia_velocity_ego=0.,
            inertia_heading=0.,
            noise_heading=0.,
            noise_velocity_ego=0.,
            noise_control_shift=0.,
            blur_retina=1.,
            gain_retina=1.,
            noise_pixel=0.,
            noise_obs_dloc=.1,
            noise_obs_dheading=.1,
            noise_obs_dvelocity_ego=.1,
            retina=None,
            render_type='walls',
            render_noiseless_retina=True,  # for visualization
            state: Union[type, AgentState] = AgentState,
            tactile_sensor: Union[TactileSensorBinary, TactileSensorTabularBinary] = None,
            dt=1.,
            **kwargs  # to ignore extraneous kwargs
    ):
        """

        Renamed from 'Agent' in slam_ori_speed_clamp,
        since this model is independent of the policy, which is assumed
        observed.

        :type env: EnvBoundary
        :param radius_corner:
        :param self_loc_xy0:
        :param heading0:
        :param velocity_ego0:
        :param self_height0:
        :param inertia_velocity_ego:
        :param inertia_heading:
        :param noise_heading:
        :param noise_velocity_ego:
        :param noise_control_shift:
        :param blur_retina:
        :param noise_obs_dloc:
        :param noise_obs_dheading:
        :param noise_obs_dvelocity_ego:
        :type retina: Retina
        :param render_type:
        """

        GenerativeModel.__init__(self)

        self.dt = dt  # PAPERPARAM

        self.geom = geom
        self.n_dim = 3
        self.gain_retina = gain_retina

        # if env is None:
        #     env = expr.behav.trapezoid_Bellmund2020.env_bellmund2020.Trapezoid()
        self.env = env
        self.radius_corner = radius_corner

        self.tactile_sensor = tactile_sensor

        # self.img_given_state = None

        self.inertia_velocity_ego = inertia_velocity_ego
        self.inertia_heading = inertia_heading

        if self_height0 is None:
            self_height0 = env.self_height

        # Initial self location
        if type(state) is type:
            state = state(
                loc=torch.cat([
                    npt_tensor(self_loc_xy0),
                    npt_tensor([self_height0])
                ], -1),
                velocity_ego=npt_tensor(velocity_ego0),
                heading=torch.cat([
                    npt_tensor(heading0),
                    npt_tensor([0.])
                ], -1)
            )
        self.agent_state = state  # type: AgentState

        # Transition & observation noise
        self.noise_heading = npt_tensor(noise_heading)
        self.noise_velocity_ego = npt_tensor(noise_velocity_ego)
        self.noise_tangential = npt_tensor(noise_control_shift)

        self.blur_retina = npt_tensor(blur_retina)  # in degree
        self.noise_pixel = npt_tensor(noise_pixel)
        self.noise_obs_dloc = npt_tensor(noise_obs_dloc)
        self.noise_obs_dheading = npt_tensor(noise_obs_dheading)
        self.noise_obs_dvelocity_ego = npt_tensor(noise_obs_dvelocity_ego)

        # Observations
        self.render_type = render_type
        self.render_noiseless_retina = render_noiseless_retina
        if render_type == 'walls':
            n_chn = 3
        elif render_type == 'corners':
            n_chn = 1
        else:
            raise ValueError()
        if retina is None:
            retina = Retina(n_channel=n_chn)
        if self.render_noiseless_retina:
            self.retina_noiseless = retina.get_copy()
        else:
            self.retina_noiseless = None

        self.meas = Measurement(
            retina=retina,
            dloc=npt_zeros_like(self.agent_state.dloc),
            dheading=npt_zeros_like(self.agent_state.dheading),
            tactile_sensor=tactile_sensor,
            # dvelocity_ego=npt_zeros_like(self.agent_state.dvelocity_ego),
        )

        if render_type == 'walls':
            # add meshes here from env, and only move cameras in
            # observe_retinal_image()

            # NOTE: might as well render here, to help pickling env.
            #   but the best way is to avoid pickling complex objects
            #   to begin with.

            env = self.env
            self.meas.retina.fun_render_init = render_walls
            self.meas.retina.fun_render_init_args = (
                env, self.agent_state.loc, self.agent_state.heading_deg)

            #  = lambda fig: (
            #     render_walls(
            #         fig,
            #         env,
            #         self.agent_state.loc,
            #         self.agent_state.heading_deg
            #     )
            # )

            if self.render_noiseless_retina:
                env_highcontrast = deepcopy(env)
                c, cw = env.contrast, env._contrast_btw_walls  # TODO Is this supposed to be protected?
                # env_highcontrast.set_contrast(1., 1.)
                self.retina_noiseless.fun_render_init = render_walls
                self.retina_noiseless.fun_render_init_args = (
                    env_highcontrast,
                    self.agent_state.loc,
                    self.agent_state.heading_deg)
                # self.retina_noiseless.fun_render_init = lambda fig: (
                #     render_walls(
                #         fig,
                #         env_highcontrast,
                #         self.agent_state.loc,
                #         self.agent_state.heading_deg
                #     )
                # )
                env.contrast = c
                env._contrast_btw_walls = cw

        self.dcorners_deg = None
        self.dist = None
        self.apparent_radius_deg = None

        # NOTE: consider adding corners / edges here as well to mayavi figure,
        #   and only move cameras in observe_retinal_image() to save time

    @staticmethod
    def fill_control_heading(rot_yx=0., rot_zy=0., rot_xz=0.):
        """
        :param rot_yx: deg or rad
        :param rot_zy: deg or rad
        :param rot_xz: deg or rad
        :return:
        """

        rot_yx = npt_tensor(rot_yx, min_ndim=0)
        rot_zy = npt_tensor(rot_zy, min_ndim=0)
        rot_xz = npt_tensor(rot_xz, min_ndim=0)

        rot_zy, rot_xz, rot_yx = expand_batch(
            rot_zy, rot_xz, rot_yx
        )
        return torch.stack([rot_zy, rot_xz, rot_yx], -1)

    def transition(self, control: Control,
                   state: AgentState = None) -> AgentState:
        """
        :param control:
        :param state:
        :return: AgentState
        """
        if self.dt != 1:
            raise NotImplementedError(
                'yet to scale noise with dt in gen.transition()'
                'when dt != 1')

        # Heading
        if state is None:
            s = self.agent_state
        else:
            s = state
        heading_prev = s.heading.clone()

        if torch.all(~self.noise_heading.bool()):
            noise_heading = npt_zeros(self.n_dim)
        else:
            noise_heading = torch.cat([mvnrnd(
                npt_zeros(self.n_dim - 1),
                self.noise_heading * npt_eye(self.n_dim - 1)
            ), npt_tensor([0.])], 0)
        v = npt_tensor([self.inertia_heading, 0., 0.]) \
            + geom.rad2ori(control.dheading_deg / 180. * np.pi) + \
            noise_heading
        s.heading = get_unit_vec(geom.h2e(geom.rotate_to_align(
            geom.e2h(v),
            geom.e2h(s.heading),
            add_dst=True
        )))
        s.dheading = s.heading - heading_prev

        # Velocity (vector, new)
        velocity_ego_prev = s.velocity_ego.clone()
        noise_velocity_ego = normrnd((0., 0., 0.), self.noise_velocity_ego,
                                         sample_shape=control.velocity_ego.shape[:-1])
        s.velocity_ego = (
                s.velocity_ego * self.inertia_velocity_ego + control.velocity_ego + noise_velocity_ego
        )
        s.dvelocity_ego = s.velocity_ego - velocity_ego_prev

        velocity_RtFwUp = permute2en(
            # s.velocity_ego is originally FwRtUp
            permute2st(s.velocity_ego)
            * npt_tensor([1., -1., 1.])
        )

        direction = get_unit_vec(geom.h2e(geom.rotate_to_align(
            geom.e2h(velocity_RtFwUp),
            geom.e2h(s.heading),
            add_dst=True
        )))
        speed = (s.velocity_ego ** 2).sum(-1).sqrt()

        # Position
        loc_prev = s.loc.clone()

        loc = (
                loc_prev
                + direction * speed
        )
        if self.env.is_inside(npy(p2st(loc, 1).unsqueeze(1).flatten(1)
                                  [:2, :]).T):
            s.loc = loc
        else:
            # Revert to prevent going through the wall
            s.loc = loc_prev.clone()

        s.dloc = s.loc - loc_prev

        # Return state, which is necessary and sufficient to feed
        # observation(), and determines meas.
        return s

    def measure(
            self, state: AgentState = None, **kwargs
    ) -> Measurement:
        """
        :type state: AgentState
        :rtype: Measurement
        """
        if state is None:
            state = self.agent_state
        meas = self.meas

        # NOTE: consider re-enabling self motion input
        # # -- self motion
        # noise_obs_dloc = torch.cat([
        #     npt.mvnrnd(
        #         npt_zeros(self.n_dim - 1),
        #         self.noise_obs_dloc * npt_eye(self.n_dim - 1)
        #     ), npt_zeros(1) # Assume zero motion along height
        # ], -1)
        # meas.dloc = geom.h2e(geom.rotate_to_align(
        #     geom.e2h(state.dloc),
        #     geom.e2h(state.heading)
        # )) + noise_obs_dloc
        #
        # noise_obs_dheading = torch.cat([npt.mvnrnd(
        #     npt_zeros(self.n_dim - 1),
        #     self.noise_obs_dheading * npt_eye(self.n_dim - 1)
        # ), npt_zeros(1)], -1)
        # meas.dheading = geom.h2e(geom.rotate_to_align(
        #     geom.e2h(state.dheading),
        #     geom.e2h(state.heading)
        # )) + noise_obs_dheading

        # noise_obs_dvelocity_ego = npt.normrnd(
        #     sigma=self.noise_obs_dvelocity_ego
        # )
        # meas.dvelocity_ego = state.dvelocity_ego + noise_obs_dvelocity_ego

        meas.retina.image = self.measure_retinal_image(state, **kwargs)

        if self.tactile_sensor is not None:
            meas.tactile_sensor.tactile_input = self.tactile_sensor.sense(state)

        return meas

    def measure_retinal_image(
            self, state=None, retina=None,
            blur_retina=None,
            noise_pixel=None,
            render_noiseless_retina=None
    ):
        """
        :param blur_retina:
        :param noise_pixel:
        :param render_noiseless_retina:
        :type state: AgentState
        :type retina: Retina
        :return: image[x, y, color]
        """
        if state is None:
            state = self.agent_state
        if retina is None:
            retina = self.meas.retina
        if blur_retina is None:
            blur_retina = self.blur_retina
        if noise_pixel is None:
            noise_pixel = self.noise_pixel

        # -- retinal_image
        # First obtain retinal coordinates of the corners
        # Then compute the distance of the corners from each retinal
        # pixel position
        if self.render_type == 'corners':
            corners = npt_tensor(self.env.corners)
            dcorners = corners.clone()
            dcorners[:, :2] = dcorners[:, :2] - state.loc[None, :2]
            dcorners = geom.h2e(geom.rotate_to_align(
                geom.e2h(dcorners),
                geom.e2h(state.heading)
            ))
            # Consider height
            dcorners[:, 2] = corners[:, 2] - state.loc[2]
            dcorners_deg = rad2deg(geom.ori2rad(dcorners, n_dim=3))

            dist = torch.sqrt((dcorners ** 2).sum(1))
            # noinspection PyTypeChecker
            apparent_radius_deg = rad2deg(
                torch.asin(npt_tensor(self.radius_corner / dist,
                                      min_ndim=0)))
            apparent_radius_deg[self.radius_corner >= dist] = 1e4  # np.inf

            # DEBUGGED - logging
            self.dcorners_deg = dcorners_deg
            self.dist = dist
            self.apparent_radius_deg = apparent_radius_deg

            retina.image.zero_()
            x, y = torch.meshgrid(retina.xs)
            for d, r in zip(dcorners_deg, apparent_radius_deg):
                # retina.image[:,:,0] = retina.image[:, :, 0] \
                #     + torch.exp(
                #         -torch.clamp_min(
                #             (x + d[2]) ** 2
                #             + (y - d[1]) ** 2
                #             - r ** 2
                #         , 0.)
                #         / self.noise_retina ** 2
                #         - torch.log(self.noise_retina)
                #     )
                # retina.image[:,:,0] = torch.clamp_max(
                #     retina.image[:, :, 0]
                #     , 1. / self.noise_retina
                # )

                retina.image[:, :, 0] = retina.image[:, :, 0] + torch.exp(
                    distributions.Normal(
                        npt_tensor(0.),
                        blur_retina).log_prob(
                        torch.sqrt(torch.clamp_min(
                            (x + d[2]) ** 2 + (y - d[1]) ** 2 - r ** 2,
                            0.
                        ))))
        elif self.render_type == 'walls':
            retina.view(
                loc=npy(state.loc),
                heading_deg=npy(state.heading_deg)
            )
            img = retina.render_image() / 255

            if render_noiseless_retina is None:
                render_noiseless_retina = self.render_noiseless_retina
            if render_noiseless_retina:
                self.retina_noiseless.view(
                    loc=npy(state.loc),
                    heading_deg=npy(state.heading_deg)
                )
                self.retina_noiseless.image = \
                    self.retina_noiseless.render_image() / 255

            # # CHECKED
            # print('id(state, retina, retina.fig): %d, %d, %d'
            #       % (id(state), id(retina), id(retina.fig)))
            # print((npy(state.loc), npy(state.heading_deg)))
            # plt.imshow(npy(img.transpose([1, 0, 2])), origin='lower')
            # plt.show()

            if blur_retina > 0:
                blur_pix = float(npy(blur_retina)) / retina.deg_per_pix
                img = np.stack([
                    ndimage.gaussian_filter(
                        img11, sigma=blur_pix,
                        truncate=3 + 1 / blur_pix,  # kernel_size / 2 / kernel_stdev
                    ) for img11 in permute2st(img)
                ], -1)

                # # CHECKED: comparing ndimage and cv2.GaussianBlur
                # img1 = np.stack([
                #     ndimage.gaussian_filter(
                #         img11, sigma=blur_pix,
                #         truncate=3 + 1 / blur_pix,  # kernel_size / 2 / kernel_stdev
                #     ) for img11 in np2.permute2st(img)
                # ], -1)
                # import cv2
                # img2 = cv2.GaussianBlur(
                #     src=img,
                #     ksize=(
                #               int(np.ceil(blur_pix * 3 + 1)) // 2 * 2 + 1,
                #           ) * 2,
                #     sigmaX=blur_pix,
                # )
                # axs = plt2.GridAxes(
                #     2, 4, widths=2, heights=2,
                #     hspace=0.5, top=1
                # )
                # for axs1, img11 in zip(axs.axs.T, [
                #     img, img1, img2, img2 - img1
                # ]):
                #     plt.sca(axs1[0])
                #     plt.imshow(
                #         npy(img11).transpose([1, 0, 2]),
                #         origin='lower'
                #     )
                #
                #     plt.title(f'max: {np.amax(img11):1.3g}, '
                #               f'min: {np.amin(img11):1.3g}')
                #
                #     plt.sca(axs1[1])
                #     mid = img11.shape[0] // 2
                #     plt.plot(img11[:, mid])
                # axs.suptitle(f'blur: {blur_pix} pix')
                # plt.show()
                # print('--')  # CHECKED

            if noise_pixel > 0:
                img = np.random.poisson(img * npy(noise_pixel)) / npy(
                    noise_pixel)
                # img = np.clip(
                #     img / np.amax(npy(noise_pixel)),
                #     a_min=0.,
                #     a_max=1.
                # )

            retina.image = npt_tensor(img, dtype=torch.double)

            # # CHECKED
            # retina.plot()
            # plt.show()

        else:
            raise ValueError()

        return retina.image


class GenerativeModelSingleEnvContTabular(GenerativeModelSingleEnv, EnvTable):
    """
    Description:
    -------------
    Inherits both the EnvTable class and GenerativeModelSingleEnv class allowing a generative model to be
    constructed for discrete and continuous state spacers.

    Attributes:
    ----------
    - env
    - tactile_sensor
    - cache_img_given_state
    - dict_img_given_state
    - Lazy pre-computation
        - _img_given_state
        - _log_img_given_state
        - _sum_img_given_state

    Methods:
    -----------
    - get_dict_file_img_given_state
    - state_template
    - measure_retinal_image
    - measure_retinal_image_vec
    - measure_vec
    - get_img_given_state
    - set_img_given_state
    - get_mean_img_given_state
    - log_img_given_state
    - get_sum_img_given_state
    """

    def __init__(
            self,
            env: Union[EnvBoundary, EnvTable],
            tactile_sensor: Union[TactileSensorBinary, TactileSensorTabularBinary] = None,
            **kwargs
    ):
        """

        :param env:
        :param kwargs:

        --- EnvTable
        - env:
        - x_max:
        - dx:
        - dy:
        - ddeg:
        - to_use_inner:
        - kwargs:

        --- GenerativeModelSingleEnv
        - radius_corner:
        - self_loc_xy0:
        - heading0:
        - velocity_ego0:
        - self_height0:
        - inertia_velocity_ego:
        - inertia_heading:
        - noise_heading:
        - noise_velocity_ego:
        - noise_control_shift:
        - blur_retina:
        - noise_obs_dloc:
        - noise_obs_dheading:
        - noise_obs_dvelocity_ego:
        - retina: Retina
        - render_type:
        """

        # Check if continuous or discrete state space.
        if self.is_binned_space(env):
            EnvTable.__init__(self, env=env, **kwargs)
        else:
            self.dx, self.dy, self.ddeg = 0, 0, 0

        GenerativeModelSingleEnv.__init__(
            self, env=env, tactile_sensor=tactile_sensor, **kwargs)

        self.cache_img_given_state: Optional[Cache] = None
        self.dict_img_given_state = 0

        # Lazy pre-computation
        self._img_given_state = None
        self._log_img_given_state = None
        self._sum_img_given_state = None

    @staticmethod
    def is_binned_space(env):
        return isinstance(env, EnvTable)

    def get_dict_file_img_given_state(self) -> Dict[str, str]:
        return shorten_dict({
            Short('ev', 'environment name'): self.env.name,
            Short('dx'): '%g' % self.dx,
            Short('dy'): '%g' % self.dy if self.dy != self.dx else None,
            Short('dd', 'ddegree'): '%g' % self.ddeg if self.ddeg != 90 else None,

            Short('bl', 'retina blur'): self.blur_retina,
            Short('gn', 'retinal gain'): None if self.gain_retina == 1
            else '%g' % self.gain_retina,
            # Short('dr', 'retina duration'): None if self.gain_retina == 1
            # else '%g' % self.gain_retina,
            Short('dp', 'degree per pixel'):
                None if self.meas.retina.deg_per_pix == 2
                else self.meas.retina.deg_per_pix,

            Short('vf', 'visual field'): (
                ','.join(
                    [f'{v:g}' for v in np.array(
                        self.meas.retina.aperture_deg).flatten()])
                if np.any(
                    np.abs(self.meas.retina.aperture_deg[0][0])
                    != np.abs(
                        np.array(
                            self.meas.retina.aperture_deg
                        ).flatten()))
                else '%g' % np.abs(self.meas.retina.aperture_deg[0][0])
            ),

            Short('ct', 'contrast'): ('%g' % self.env.contrast).lstrip(
                '0'),
            Short('cw', 'contrast between walls'): (
                    '%g' % self.env.contrast_btw_walls).lstrip('0'),

        }, shorten_zero=True)

    @property
    def state_template(self) -> AgentState:
        return self.agent_state

    def measure_retinal_image(
            self, state=None, retina=None,
            blur_retina=None,
            noise_pixel=None,
            to_force_render=False,
    ) -> torch.Tensor:
        if retina is None:
            retina = self.meas.retina

        if self._img_given_state is not None and not to_force_render and self.is_binned_space(self.env):
            i_state = self.get_i_state(loc=state.loc,
                                       heading_deg=state.heading_deg)
            image = self._img_given_state[i_state, :]

            if noise_pixel is not None and noise_pixel > 0:
                image = npt_tensor(
                    np.random.poisson(npy(image) * npy(noise_pixel))
                    / npy(noise_pixel), dtype=torch.double)
                retina.image = image
        else:
            image = super().measure_retinal_image(
                state, retina, blur_retina, noise_pixel
            )
        return image

    def measure_retinal_image_vec(
            self, i_state: torch.Tensor, gain_retina=None
    ) -> torch.Tensor:
        """
        :param i_state:
        :param gain_retina: when gain_retina is 2, it is as though
            there are two retinal cells when there were 1 when gain_retina=1.
            The image in expectation should be the same,
            but it is estimated with less noise thanks to more averaging.
        :return:
        """
        image = self.get_img_given_state()[i_state]
        if gain_retina is None:
            gain_retina = self.gain_retina
        if gain_retina is None or gain_retina == 0:
            pass
        else:
            image = distributions.poisson.Poisson(
                image * gain_retina * self.dt
            ).sample() / self.dt / gain_retina
        return image

    def measure_vec(
            self, i_states: torch.Tensor,
            gain=1.,
    ):
        """
        :param i_states: [batch]
        :param gain:
        :return: rate_retina[batch, x, y, channel],
            tactile_input[batch, dheading]
        """
        rate_retina = (distributions.Poisson(
            self.get_img_given_state()[i_states] * self.dt * gain
        ).sample() / self.dt / gain)

        tactile_input = self.tactile_sensor.sense_vec(i_states)

        return rate_retina, tactile_input

    def get_img_given_state(self, particles: Optional[Particles] = None) -> torch.Tensor:
        """
        :return: img_given_state[state, x, y, c]
        """
        # print('before computing img_given_state')  # CHECKED
        # if self._log_img_given_state is None:
        # Precompute spatial tuning curve of each pixel (img_given_state)
        # Note that noise is defined by the noise of the image.
        # Noise parameter is currently not defined inside this filter.
        # To include retinal noise, I'll have to give the renderer the
        # retinal noise parameter defined in this class.

        if self.is_binned_space(self.env):
            loaded = False
            cache = self.cache_img_given_state
            if cache is not None:
                try:
                    img_given_state = self.dict_img_given_state[cache.fullpath]
                    loaded = True
                    self._img_given_state = img_given_state
                except KeyError:
                    try:
                        img_given_state = cache.getdict(['img_given_state'])[0]
                        assert img_given_state is not None
                        loaded = True
                        self._img_given_state = img_given_state
                    except (KeyError, AssertionError):
                        pass

            if not loaded:
                if cache is not None:
                    print(f'cache not found for img_given_state at {cache.fullpath}'
                          '\ncomputing..')

                img_given_state = torch.stack([
                    self.get_mean_img_given_state((x1, y1), heading1)[0].clone()
                    for x1, y1, heading1 in zip(self.x, self.y, self.heading_deg)
                ], 0)  # PARAM: set breakpoint here to stop before importing mayavi, VTK, and PyQT

                if cache is not None:
                    cache.set({'img_given_state': img_given_state})
                    cache.save()

                self._img_given_state = img_given_state

            if self.dict_img_given_state is not None and cache is not None:
                self.dict_img_given_state[
                    cache.fullpath] = self._img_given_state
                cache.clear()  # to avoid memory leak
        else:
            self._img_given_state = torch.stack([
                self.get_mean_img_given_state((x1, y1), heading1)[0].clone()
                for x1, y1, heading1 in particles.loc
            ], 0)

        return self._img_given_state

    def set_img_given_state(self, v: torch.Tensor):
        self._img_given_state = npt_tensor(v)

    def get_mean_img_given_state(
            self,
            xy=(0., 0.),
            heading_deg=0.
    ):
        state = self.get_state(xy=xy, heading_deg=heading_deg)
        # # CHECKED
        # print('obs2d.get_mean_img_given_state(): gen id: %d' % id(self.gen))
        return self.measure_retinal_image(
            state,
            noise_pixel=0.
        ), state

    def log_img_given_state(self, particles: Optional[Particles] = None) -> torch.Tensor:
        # # CHECKED
        # print('Observer2D.log_img_given_state(): observer ID: %d' % id(self))
        #if self._log_img_given_state is None:
        self._log_img_given_state = torch.log(self.get_img_given_state(particles))
        return self._log_img_given_state

    def get_sum_img_given_state(self, particles: Optional[Particles] = None) -> torch.Tensor:
        #if self._sum_img_given_state is None:
        self._sum_img_given_state = torch.sum(
            self.get_img_given_state(particles),
            dim=[v for v in range(1, self.get_img_given_state(particles).ndim)]
        )
        return self._sum_img_given_state


class ObserverNav(Observer, EnvTable):
    """
    Summary:
    ------------
    Continuous version of Observer2D

    Description:
    ------------
    Where Observer2D operates in a discrete state space in both location and heading, ObserverNav replicated the
    methods in the aforementioned class, but in a continuous state space.

    Attributes:
    ------------
    - Attributes of EnvTable
    - gen
    - to_use_lapse
    - gain_retina
    Dense representation
        - p_state0
    Prior: uniform (sparse representation)
        - p_state_incl
    self-motion noise parameters: not used for now
        - noise_obs_shift
        - noise_obs_rotate
    Control noise parameters
        - noise_control_shift
        - noise_control_shift_per_speed
        - noise_control_rotate_pconc
        - noise_shift_kind
        - max_speed
        - max_noise_speed

    Methods:
    ------------
    - set_prior(self) -> torch.Tensor
    - update(self) -> torch.Tensor
    - prediction_step(self) -> torch.Tensor
    - measurement_step(self) -> torch.Tensor
    - get_dict_file(self) -> Dict[str, str]
    - merge_dict_file_true_bel(obs_bel: 'Observer', obs_tru: 'Observer')
    """

    def __init__(
            self,
            gen: Union[type, GenerativeModelSingleEnvContTabular],
            noise_control_shift,  # =(0., 0.),
            noise_control_rotate_pconc,  # =.999,
            noise_control_shift_per_speed,  # =(0.1, 0.),
            noise_shift_kind='g',
            max_speed=5,
            max_noise_speed=5,
            requires_grad=False,
            duration_retina=1.,
            to_use_lapse=False,
            init_num_particles=5000, 
            **kwargs  # to ignore extraneous kwargs
    ):
        """

        :param x_max:
        :param dx:
        :param ddeg:
        :param noise_obs_shift:
        :param noise_obs_rotate:
        :param noise_control_shift: [sigma_forward, sigma_sideways]
        :param noise_control_rotate_pconc:
        :param noise_shift_kind: 'g'amma, 'n'ormal
        """

        # DEBUGGED: should not deepcopy(gen) since the retina.fig will not be
        #  correctly initialized. To use gen with a different noise
        #  characteristic, etc., construct gen outside Observer2D before
        #  giving it as an argument.
        #  Check with:
        #       assert(gen.meas.retina.fig.scene is not None)
        #  or:
        #       print(gen.meas.retina.fig.scene)
        # CAVEAT: not sure if the following is still relevant
        #   - never got the error for a while
        # assert gen.meas.retina.fig.scene is not None, \
        #     'Cannot compute visual likelihood - do not deepcopy(gen) before ' \
        #     'feeding to Observer2D - construct separately instead!'

        Observer.__init__(self)
        self.samples_loc_heading = None
        if gen.is_binned_space(gen.env):
            EnvTable.__init__(self, env=gen.env)
            # gen should be assigned after StateTable.__init__()
            # to allow calling set_state_table(gen)
            self.gen = gen  # type:GenerativeModelSingleEnvContTabular
            # it's not removed by Module.__init__()
            self.set_state_table(self.gen)

            # Dense representation
            self.p_state0 = npt_zeros(
                self.nx, self.ny, self.n_heading
            )
            # Prior: uniform (sparse representation)
            self.p_state_incl = npt_ones(self.n_state_incl) / self.n_state_incl
        else:
            self.gen = gen  # type:GenerativeModelSingleEnvContTabular
            loc, weights = self.set_prior(set_p_state=True, n_particles=init_num_particles)
            self.particles = Particles(loc, weights)

        self.to_use_lapse = to_use_lapse

        self.gain_retina = duration_retina

        # # Self-motion noise parameters: not used for now
        # self.noise_obs_shift = nn.Parameter(npt_tensor([
        #     noise_obs_shift]), True)
        # self.noise_obs_rotate = nn.Parameter(npt_tensor([
        #     noise_obs_rotate]), True)

        # Control noise parameters
        self.noise_control_shift = BoundedParameter(
            npt_tensor(noise_control_shift).expand([2]),
            -1e-6, 40., requires_grad=requires_grad
        )

        self.noise_control_shift_per_speed = BoundedParameter(
            npt_tensor(noise_control_shift_per_speed).expand([2]),
            -1e-6, 40., requires_grad=requires_grad
        )

        self.noise_control_rotate_pconc = BoundedParameter(
            npt_tensor(noise_control_rotate_pconc),
            1e-6,  # not using 0 to avoid NaN
            1 - 1e-6,  # not using 1 to avoid NaN
            requires_grad=requires_grad
        )
        assert self.noise_control_rotate_pconc[:].numel() == 1

        self.noise_shift_kind = noise_shift_kind

        self.max_speed = max_speed
        self.max_noise_speed = max_noise_speed

    def get_dict_param(self) -> Dict[str, Union[str, float]]:
        """For saving parameters to a human-readable table"""
        return {
            'spatial bin size along x (m)': self.dx,
            'spatial bin size along y (m)': self.dy,
            'heading direction bin size (deg)': self.ddeg,
            'dt (sec)': self.dt,
            **self.tactile_sensor.get_dict_param(),
            **self.gen.meas.retina.get_dict_param(),
            'altitude of the camera (m)': self.env.self_height,
            'blur (pixel)': float(self.gen.blur_retina[0] / self.gen.meas.retina.deg_per_pix),
            'contrast (unitless)': self.env.contrast,
            'self motion noise (unitless)':
                float(npy(self.noise_control_shift_per_speed[0])),
        }

    def get_dict_file(self) -> Dict[str, str]:
        return shorten_dict({
            **self.gen.get_dict_file_img_given_state(),

            # Short('dt'): None if self.dt == 1 else self.dt,
            # Short('nl', 'noise control shift per step'): str_list(obs_bel.noise_control_shift[:]),
            # Short('ns', 'noise control shift per meter'): str_list(obs_bel.noise_control_shift_per_speed[:]),
            Short('nk', 'noise shift kind'):
                None if self.noise_shift_kind == 'g' else self.noise_shift_kind,

            Short('ns', 'noise shift per meter'): '%g' % npy(
                self.noise_control_shift_per_speed[0]),
            Short('no', 'noise rotation pconc'): (
                    '%g' % self.noise_control_rotate_pconc[:]).lstrip(
                '0' if self.noise_control_rotate_pconc[:] != 0.999
                else None
            ),
            # # 'rc': rectangular_coord,
            Short('tp', 'touch reliability'): (
                f'{self.gen.meas.tactile_sensor.touch_reliability:g}'.lstrip(
                    '0')
                if (
                        (self.gen.meas.tactile_sensor is not None)
                        and (self.gen.meas.tactile_sensor != 0.999)
                ) else None),
            Short('tr', 'touch range'): (
                f'{self.gen.meas.tactile_sensor.touch_range:1.3g}'
                if self.gen.meas.tactile_sensor is not None
                else None),
            # Short('tt', 'set to true location on touch'):
            #     int(dict_cache_ideal_obs['tt']),
        }, shorten_zero=True)

    @property
    def dt(self) -> float:
        return self.gen.dt

    @property
    def tactile_sensor(self) -> Union[TactileSensorBinary, TactileSensorTabularBinary]:
        return self.gen.tactile_sensor

    @property
    def gain_retina(self):
        return self.gen.gain_retina

    @gain_retina.setter
    def gain_retina(self, v):
        self.gen.gain_retina = v

    @property
    def state_template(self) -> AgentState:
        return self.gen.agent_state

    def get_img_given_state(self) -> torch.Tensor:
        """
        :return: [state, x, y, color]
        """
        if self.gen.is_binned_space(self.gen.env):
            return self.gen.get_img_given_state()
        else:
            return self.gen.get_img_given_state(self.particles)

    def set_img_given_state(self, v: torch.Tensor):
        self.gen.set_img_given_state = v

    def log_img_given_state(self) -> torch.Tensor:
        if self.gen.is_binned_space(self.gen.env):
            return self.gen.log_img_given_state()
        else:
            return self.gen.log_img_given_state(self.particles)

    def get_sum_img_given_state(self) -> torch.Tensor:
        if self.gen.is_binned_space(self.gen.env):
            return self.gen.get_sum_img_given_state()
        else:
            return self.gen.get_sum_img_given_state(self.particles)

    def ____STATE____(self):
        pass

    @property
    def p_state_incl(self):
        return self.p_state0[self.state_incl]

    @p_state_incl.setter
    def p_state_incl(self, p):
        self.p_state0[self.state_incl] = p

    @property
    def p_state0_loc(self):
        return self.p_state0.sum(-1)

    @property
    def p_state_incl_loc(self):
        return self.p_state0_loc[self.state_loc_incl]

    # def p_state_incl_loc_heading2loc(self, p_state_incl):

    def set_prior(self, i_state=None, loc0=(0., 0.), sigma0=40.,
                  heading_deg0=0., heading_pconc0=0.95,
                  set_p_state=True, n_particles=4096*4
                  ) -> Union[tuple[Union[Tensor, Any], Tensor], Any]:
        """
        :param n_particles:
        :param i_state:
        :param loc0:
        :param sigma0:
        :param heading_deg0:
        :param heading_pconc0:
        :param set_p_state:
        :return:
        """
        if self.gen.is_binned_space(self.gen.env):
            if i_state is not None:
                loc0, heading_deg0 = self.get_loc_heading_by_i_state(i_state)
                sigma0 = eps
                heading_pconc0 = .99  # von mises parameter

            p_xy = sumto1(torch.exp(distributions.MultivariateNormal(
                loc=npt_tensor(loc0),
                covariance_matrix=npt_eye(2) * sigma0,
            ).log_prob(torch.stack([self.x0,
                                    self.y0], -1))))

            p_ori = vmpdf_prad_pconc(
                self.headings_deg / 360.,
                # DEBUGGED: convert deg to prad (0 to 1)
                npt_tensor([heading_deg0 / 360.]),
                npt_tensor([heading_pconc0]))

            p_state0 = p_xy * p_ori[None, None, :]
            p_state0[~self.state_incl] = 0.
            p_state0 = p_state0 / p_state0.sum()

            # plt.imshow(npy(p_xy[:, :, 0]).T)
            # plt.show()

            # plt.plot(npy(self.headings_deg), npy(p_ori))
            # plt.show()
            if set_p_state:
                self.p_state0 = p_state0

            return p_state0

        else:

            uniform_dist = distributions.uniform.Uniform(torch.tensor([-self.gen.env.x_max, -self.gen.env.y_max, -180]),
                                                         torch.tensor([self.gen.env.x_max, self.gen.env.y_max, 180]))
            samples = uniform_dist.rsample(torch.Size([n_particles]))
            accepted_samples = samples[self.gen.env.is_inside(samples.numpy()[:, :2])]
            while accepted_samples.shape[0] < n_particles:
                samples = uniform_dist.sample(torch.Size([n_particles - accepted_samples.shape[0]]))
                accepted_samples = torch.cat(
                    [accepted_samples,
                     samples[self.gen.env.is_inside(samples.numpy()[:, :2])]],
                    0)

            weights = torch.full([n_particles, ], 1 / n_particles)

            """
            # number of particles determined
            # sample from the multivariate normal distribution
            dist = distributions.MultivariateNormal(
                loc=npt_tensor(loc0),
                covariance_matrix=npt_eye(2) * sigma0,
            )

            n_particles = 5000
            samples = dist.sample(torch.Size([n_particles]))
            accepted_samples = samples[self.gen.env.is_inside(np.asarray(samples))]

            while accepted_samples.shape[0] < n_particles:
                samples = dist.sample(torch.Size([n_particles - accepted_samples.shape[0]]))
                accepted_samples = torch.cat(
                    [accepted_samples,
                     samples[self.gen.env.is_inside(np.asarray(samples))]],
                    0)
            self.headings_deg = distributions.uniform.Uniform(0., 360.).sample(
                torch.Size([n_particles]))
            self.n_heading = n_particles
            self.samples_loc_heading = torch.cat([accepted_samples, self.headings_deg.unsqueeze(-1)],
                                                 -1)
            p_xy = npt.sumto1(torch.exp(dist.log_prob(npt_tensor(accepted_samples))))
            p_ori = npt.vmpdf_prad_pconc(
                self.headings_deg / 360.,
                # DEBUGGED: convert deg to prad (0 to 1)
                npt_tensor([heading_deg0 / 360.]),
                npt_tensor([heading_pconc0]))

            p_state0 = (p_xy * p_ori)  """  # .reshape(-1, 1)

            # plt.scatter(npy(sample_headings), npy(p_ori))
            # plt.show()

            return accepted_samples, weights

    @staticmethod
    def get_p_s_true_given_s_belief(
            p_s_belief_given_s_true: np.ndarray, p_stationary: np.ndarray
    ) -> np.ndarray:
        return sumto1(p_s_belief_given_s_true * p_stationary[:, None], axis=1).T

    def ____FILTERING____(self):
        pass

    def update(
            self, meas, control,
            skip_prediction=False,
            skip_measurement=False,
            p_state0=None,
            update_state=True
    ):
        """
        :type meas: Measurement
        :type control: Control
        :type skip_prediction: bool
        :type skip_measurement: bool
        :type p_state0: torch.FloatTensor
        :param p_state0: [x, y, heading]
        :type update_state: bool
        :param update_state: if True (default), update the internal
        state of the object.
        :return: p_state0[x, y, heading]
        :rtype: torch.FloatTensor
        """

        if p_state0 is None:
            p_state0 = self.p_state0

        if not skip_prediction:
            p_state0 = self.prediction_step(
                control=control,
                p_state0=p_state0
            )

        if not skip_measurement:
            p_state0[self.state_incl] = sumto1(
                p_state0[self.state_incl] *
                self.measurement_step(
                    meas=meas,
                    control=control,
                    p_state_incl=p_state0[self.state_incl]
                )
            )

        if update_state:
            self.p_state0 = p_state0

        return p_state0

    def ____PREDICTION____(self):
        pass

    def transition_step(
            self,
            control: Union[torch.Tensor, Control],
            i_state_true: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        """

        :param i_state_true: [batch]
        :param control: [batch, (forwrad, rightward, dheading)]
        :return:
        """
        if self.gen.is_binned_space(self.gen.env):
            batch_shape = i_state_true.shape

            p_state_incl = npt_zeros(
                batch_shape + self.p_state_incl.shape)
            p_state_incl[..., i_state_true] = 1.

            assert np.prod(list(batch_shape)) == 1, \
                'not implemented for batch_shape != [1] ' \
                'in prediction_step()'
            control1 = Control(
                dheading_deg=control[0, 2],
                velocity_ego=control[0, :2])

            p_state_incl = self.prediction_step(
                control=control1,
                p_state_incl=p_state_incl,
                return_p_state_incl=True,
                **kwargs)

            # sample transition from prediction
            i_state_true1 = distributions.Categorical(
                probs=p_state_incl).sample()

            # print(p_state_incl.shape)
            # print(i_state_true1.shape)
            # print('--')
            return i_state_true1[None]  # scalar for now
        else:
            predicted_pose = self.prediction_step(control)
            self.particles.loc[..., :2] = predicted_pose[0]
            self.particles.loc[..., -1] = predicted_pose[1]

    def intersection_lines_vec(self, line: torch.Tensor) -> torch.Tensor:
        """
        Returns the coordinates of the intersection points of a line and the environment walls.
        from https://stackoverflow.com/a/20677983/2565317, but adapted for tensor operations.
        :param line:
        :return: x_intersect, y_intersect or None if lines do not intersect
        """
        num_corners = self.gen.env.corners.shape[0]
        corners = torch.tensor(self.gen.env.corners)
        # Contains the start and end points of each wall
        # The matrix in the first pos in torch.tensordot transforms
        # all the corners into the start and end points of the walls
        walls = torch.tensordot(torch.stack([torch.eye(num_corners),
                                             torch.roll(torch.eye(num_corners), 1, dims=1)], -1),
                                corners[:, :2], dims=([1], [0]))

        # --- Implementation of Cramer's rule for an intersecting line
        diff = torch.tensor([1., -1.])
        # 'num_corners' lots of [2, 2] matrices containing the difference between
        # the start and end points of the line (col 0) and the walls (col 1)
        full_diff = torch.stack([torch.stack(walls.size()[0] * [diff @ line]), diff @ walls], -1)
        # Determinants of the 'num_corners' [2, 2] difference matrices
        # If == 0 then the lines are parallel and do not intersect
        div = torch.det(full_diff)
        # ['num_corners', 2] matrix containing the determinant of the line and a wall
        d = torch.stack([torch.stack(walls.size()[0] * [torch.det(line)]), torch.det(walls)], -1)

        # Contains the intersection points. If no intersection, the point is (-inf, -inf)
        final = torch.full([num_corners, 2], -torch.inf)
        final[div != 0] = torch.stack([torch.det(torch.stack([d[div != 0], full_diff[div != 0, 0, :]],
                                                             -1)),
                                       torch.det(torch.stack([d[div != 0], full_diff[div != 0, 1, :]],
                                                             -1))], -1) / div[div != 0].reshape(-1, 1)
        # Conditions that ensure the intersection point on the line segment and the wall segment;
        # Cramer's rule is for infinite lines, so we need to check this.
        condition = ((final <= torch.maximum(line[0], line[1]) + 1e-6) &  # line segment maximum point
                     (final >= torch.minimum(line[0], line[1]) - 1e-6) &  # line segment minimum point
                     (final <= torch.maximum(walls[:, 0, :], walls[:, 1, :]) + 1e-6) &  # wall segments maximum point
                     (final >= torch.minimum(walls[:, 0, :], walls[:, 1, :]) - 1e-6))  # wall segments minimum point
        condition = condition[:, 0] & condition[:, 1]
        if True not in condition:
            return torch.tensor([[torch.inf, torch.inf]])

        return final[condition, :][torch.argmin(torch.norm(final[condition, :] - line[0], dim=1))]

    def intersection_lines_vvec(self, lines: torch.Tensor) -> torch.Tensor:
        """
        Returns the coordinates of the intersection points of a line and the environment walls.
        from https://stackoverflow.com/a/20677983/2565317, but adapted for tensor operations.
        :param lines:
        :return: [x_intersect, y_intersect] or [inf, inf] line segments do not intersect
        """
        num_walls = self.gen.env.corners.shape[0]
        num_lines = lines.size()[0]
        corners = torch.tensor(self.gen.env.corners)
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

    def prediction_step(
            self, control: Control,
            p_state0: torch.Tensor = None,
            p_state_incl: torch.Tensor = None,
            update_state=False,
            use_boundary_prior=True,
            noise_control_rotate_pconc=None,
            noise_control_shift=None,
            noise_control_shift_per_speed=None,
            # to_normalize=True,
            return_p_state_incl=False,
            to_plot=False,  # PARAM to plot prediction_step()
            verbose=False,
            dict_file_plot=(),
    ) -> Union[tuple[Tensor, Union[Tensor, Any]], Tensor]:
        """
        :param control:
        :param p_state0: [x, y, heading]
        :param p_state_incl: [i_state_incl]
        :param update_state: if True, update self.p_state0. Default: False
        :param use_boundary_prior: if True (default), p > 0 only inside the
            boundary.
        :param noise_control_rotate_pconc:
        :param noise_control_shift:
        :param noise_control_shift_per_speed:
        :param return_p_state_incl:  if True, only return included states.
            Defaults to False for backward compatibility.
        :param to_plot:
        :param verbose:
        :param dict_file_plot:
        :return: p_state0[x, y, heading]
        """

        if noise_control_rotate_pconc is None:
            noise_control_rotate_pconc = self.noise_control_rotate_pconc[0]
        else:
            noise_control_rotate_pconc = noise_control_rotate_pconc + \
                                         npt_zeros_like(
                                             self.noise_control_rotate_pconc[0])

        if noise_control_shift is None:
            noise_control_shift = self.noise_control_shift[:]
        else:
            noise_control_shift = noise_control_shift + npt_zeros_like(
                self.noise_control_shift[:])

        if noise_control_shift_per_speed is None:
            noise_control_shift_per_speed = \
                self.noise_control_shift_per_speed[:]
        else:
            noise_control_shift_per_speed = \
                noise_control_shift_per_speed + npt_zeros_like(
                    self.noise_control_shift_per_speed[:])

        if not self.gen.is_binned_space(self.gen.env):

            # --- Predict rotation
            # pred_rotation_dist_approx = distributions.VonMises(2 * torch.pi * control.dheading_deg[..., -1] / 360,
            #                                                   np2.pconc2conc(noise_control_rotate_pconc) /
            #                                                   (2 * torch.pi * control.dheading_deg[..., -1] / 360)
            #                                                   * self.dt
            #                                                   )

            # --- Update headings
            headings = self.particles.loc[..., -1] + torch.rad2deg(
                distributions.VonMises(torch.deg2rad(control.dheading_deg[..., -1]) * self.dt,
                                       pconc2conc(noise_control_rotate_pconc) /
                                       (torch.abs(torch.deg2rad(control.dheading_deg[..., -1]))
                                        * self.dt)
                                       ).sample(torch.Size([self.particles.n_particle])))
            """
            self.particles.loc[..., -1] += torch.rad2deg(
                distributions.VonMises(torch.deg2rad(control.dheading_deg[..., -1]),
                                       np2.pconc2conc(noise_control_rotate_pconc) /
                                       (torch.abs(torch.deg2rad(control.dheading_deg[..., -1]))
                                        * self.dt)
                                       ).sample(torch.Size([self.particles.n_particle]))
            )
            """

            # --- Predict forward shifts
            rate = (control.velocity_ego[0] /
                    (self.dt * control.velocity_ego[0] * noise_control_shift_per_speed[0]) ** 2)
            conc = control.velocity_ego[0] * rate
            # pred_f_gam_dist = distributions.Gamma(conc, rate)

            # --- Predict rightward shifts
            # pred_r_norm_dist = distributions.Normal(0,
            #                                        (self.dt * control.velocity_ego[0] * noise_control_shift[1]) ** 2)

            # --- Update locations (x, y)
            rot = torch.stack([torch.stack([torch.cos(deg2rad(headings)),
                                            torch.sin(deg2rad(headings))], -1),
                               torch.stack([-torch.sin(deg2rad(headings)),
                                            torch.cos(deg2rad(headings))], -1)],
                              -1)

            new_loc = self.particles.loc[..., :2] + (rot @ torch.stack(
                [distributions.Gamma(conc,
                                     rate).rsample([self.particles.n_particle]),
                 distributions.Normal(0, (self.dt *
                                          control.velocity_ego[0] *
                                          noise_control_shift_per_speed[1]) ** 2).rsample(
                     [self.particles.n_particle])], -1)[..., None])[..., 0]

            # --- Check new locations
            trajectories = torch.stack([self.particles.loc[..., :2], new_loc], 1)
            intersections = self.intersection_lines_vvec(trajectories)
            locs = torch.where(torch.isinf(intersections).any(-1, keepdim=True),
                               new_loc,
                               0.05 * self.particles.loc[..., :2] + 0.95 * intersections)

            return locs, headings

        else:

            if p_state0 is None:
                p_state0 = self.p_state0
                if p_state_incl is not None:
                    p_state0[self.state_incl] = p_state_incl

            # --- Predict rotation
            # DEBUGGED: rotation should happen before jumping in that direction
            p_heading1_given_heading0 = vmpdf_a_given_b(
                self.headings_deg / 360.,
                (self.headings_deg - control.dheading_deg[..., -1]) / 360.,
                noise_control_rotate_pconc
            )

            p_state = p_state0 @ p_heading1_given_heading0
            p_angle0 = p_state.sum([0, 1], keepdim=True)

            # --- Predict shift
            if self.noise_shift_kind == 'g':
                assert (noise_control_shift < 1e-6).all()
                assert (noise_control_shift_per_speed[1] < 1e-6)
                assert self.n_heading in [1, 2, 4]
                assert (control.velocity_ego[..., 1] < 1e-6).all()
                assert self.n_heading in [1, 2, 4], \
                    'piling up of probability at boundaries ' \
                    'only implemented for n_heading in [1, 2, 4]'

                speed = control.velocity_ego[0]
                # p[dx]
                p_dx = self.noise_shift_gampdf(
                    noise_control_shift_per_speed[None, 0], speed
                )[0]
                ndx1 = len(p_dx) - 1
                p_dx = npy(F.pad(p_dx, [ndx1, 0])[:, None])

                def transform_into_heading0(
                        v: np.ndarray, heading: float
                ) -> np.ndarray:
                    if heading == 0:
                        return v
                    elif heading == 90:
                        return v.T
                    elif heading == 180:
                        return np.flip(v, 0)
                    elif heading == 270:
                        return np.flip(v.T, 0)  # order of .T & .flip is important!
                    else:
                        raise ValueError()

                def transform_from_heading0(
                        v: np.ndarray, heading: float
                ) -> np.ndarray:
                    if heading == 0:
                        return v
                    elif heading == 90:
                        return v.T
                    elif heading == 180:
                        return np.flip(v, 0)
                    elif heading == 270:
                        return np.flip(v, 0).T  # order of .T & .flip is important!
                    else:
                        raise ValueError()

                # p_state[x, y, heading]
                # print(p_state.shape)
                p_state0 = p_state.clone()
                for i_heading, (heading1, p0, incl) in enumerate(zip(
                        self.headings_deg,
                        npy(permute2st(p_state0)),
                        npy(permute2st(self.state_incl))
                )):
                    if p0.sum() == 0:
                        continue

                    # --- transform into the same orientation as heading = 0
                    p01 = transform_into_heading0(p0, heading1)
                    incl = transform_into_heading0(incl, heading1)

                    # --- convolve
                    from scipy.signal import convolve2d
                    # p0 = np.pad(p0, ((ndx1, ndx1), (0, 0)))
                    # p1 = convolve2d(p0, p_dx, mode='same')
                    p1 = convolve2d(p01, p_dx, mode='full')

                    # --- crop
                    p2 = p1.copy()
                    p2 = p2[ndx1:-ndx1, :]
                    p21 = p2.copy()

                    # --- pile prob mass outside boundary back to the boundary
                    for jj in range(p21.shape[1]):
                        if not any(incl[:, jj]):
                            continue

                        i_last = np.nonzero(incl[:, jj])[0][-1]
                        p21[i_last, jj] = p1[(i_last + ndx1):, jj].sum()
                        p21[(i_last + 1):, jj] = 0

                    # --- add lapse
                    if self.to_use_lapse:
                        p21[incl] = (
                                p21[incl] * (1 - 1e-9)
                                + 1e-9 * p21[incl].sum() / incl.sum())

                    # --- transform back to the original orientation
                    p11 = transform_from_heading0(p1, heading1)  # for plotting
                    p22 = transform_from_heading0(p21, heading1)
                    p_dxy_given_heading = transform_from_heading0(p_dx, heading1)

                    p_state[..., i_heading] = npt_tensor(p22.copy())

                    # --- sanity check
                    assert p22.shape == p0.shape
                    assert issimilar(p22.sum(), p0.sum(), 1e-9)

                    # # if using pytorch's F.conv2d, which is in fact crosscorr:
                    # p_dxy_given_heading = p_dxy_given_heading.flip(0).flip(1)

                    # print(p.shape)
                    # print(p_dxy_given_heading.shape)

                    def imshow1(v):
                        plt.imshow(npy(v).T, origin='lower')

                    if verbose:
                        print(p0.shape)
                        print(p11.shape)
                        print(p22.shape)
                        print(f'sum bef crop: {p1.sum() - p0.sum()}')
                        print(f'sum bef piling: {p2.sum() - p0.sum()}')
                        print(f'sum aft piling: {p22.sum() - p0.sum()}')

                    if to_plot:  # CHECKED sum bef & aft piling
                        axs = GridAxes(1, 4)
                        plt.sca(axs[0, 0])
                        # plt.plot(p0.sum(1), '.-')
                        imshow1(p0)
                        plt.title(f'{joinformat(p0.shape)}')

                        plt.sca(axs[0, 1])
                        # plt.plot(p_dxy_given_heading.sum(1), '.-')
                        imshow1(p_dxy_given_heading)
                        plt.title(f'{joinformat(p_dxy_given_heading.shape)}')

                        plt.sca(axs[0, 2])
                        # plt.plot(p11.sum(1), '.-')
                        imshow1(p11)
                        plt.title(f'{joinformat(p1.shape)}')

                        plt.sca(axs[0, 3])
                        # plt.plot(p21.sum(1), '.-')
                        imshow1(p22)
                        plt.title(f'{joinformat(p22.shape)}')

                        # plt2.sameaxes(axs)
                        plt.show()

                    if to_plot:  # CHECKED
                        axs = GridAxes(
                            2, 3, widths=2, heights=2,
                            wspace=1.5, hspace=0.75, left=0.75
                        )
                        plt.sca(axs[0, 0])
                        plt.imshow(npy(p_dxy_given_heading).T,
                                   origin='lower')
                        plt.title(f'p_dxy gv heading={heading1}\n'
                                  f'(flipped for crosscorr)')
                        plt.xlabel('dx')
                        plt.ylabel('dy')

                        plt.sca(axs[0, 1])
                        # plt.imshow(npy(p_state0[..., i]).T, origin='lower')
                        self.plot_p_loc(p_state0[..., i_heading][self.state_loc_incl])
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.title('p_state: bef conv')
                        self.env.plot_walls()

                        plt.sca(axs[0, 2])
                        # plt.imshow(npy(p_state[..., i]).T, origin='lower')
                        self.plot_p_loc(p_state[..., i_heading][self.state_loc_incl])
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.title('p_state: aft conv')

                        plt.sca(axs[1, 0])
                        plt.plot(p_dx, color='k')
                        plt.ylabel(f'p_dxy gv heading={heading1}')
                        box_off()

                        plt.sca(axs[1, 1])
                        h0 = plt.plot(
                            sumto1(p_state0[:, p_state0.shape[1] // 2, i_heading]),
                            color='b')
                        h1 = plt.plot(
                            sumto1(p_state[:, p_state.shape[1] // 2, i_heading]),
                            color='r', ls='--')
                        plt.title('along x')
                        plt.xlabel('x')
                        plt.ylabel(f'p(x | y=middle,\nheading={heading1})')
                        plt.legend([h0[0], h1[0]], ['bef', 'aft'],
                                   **kw_legend)
                        box_off()

                        plt.sca(axs[1, 2])
                        plt.plot(
                            sumto1(p_state0[p_state0.shape[0] // 2, :, i_heading]),
                            color='b')
                        plt.plot(
                            sumto1(p_state[p_state.shape[0] // 2, :, i_heading]),
                            color='r', ls='--')
                        plt.title('along y')
                        plt.ylabel(f'p(y | x=middle,\nheading={heading1})')
                        plt.xlabel('y')
                        box_off()

                        file = locfile.get_file_fig(
                            'pred', {
                                'h': npy(heading1),
                                **dict(dict_file_plot)
                            })
                        mkdir4file(file)
                        savefig(file, dpi=150)
                        print(f'Saved to {file}')
                        plt.close(axs.figure.number)

                # --- make sure no state that's not included has nonzero probability
                assert issimilar(p_state0.sum(), p_state.sum(), 1e-9)
                assert not p_state0[~self.state_incl].any()
                assert not p_state[~self.state_incl].any()

            elif self.noise_shift_kind == 'n':
                ndxy = int(min([
                    np.ceil((self.max_speed + self.max_noise_speed * 2) / self.dx),
                    max([self.nx_incl, self.ny_incl])
                ]))
                dxs1 = npt_arange(-ndxy, ndxy + 1) * self.dx
                dxs, dys = torch.meshgrid([dxs1, dxs1])

                # mu_loc_next_given_heading[heading, xy] = mu|heading
                mu_loc_next_given_heading = (
                                                    permute2st(control.velocity_ego)[0]
                                                    * prad2unitvec(self.headings_deg / 360.)
                                            ) + (
                                                    permute2st(control.velocity_ego)[1]
                                                    * prad2unitvec((self.headings_deg - 90.) / 360.)
                                            )

                # sigma_given_heading[heading, xy, xy] = sigma|heading
                # sigma[xy, xy]: 2 x 2 matrix
                sigma_loc = (
                        torch.diag(noise_control_shift) ** 2
                        + torch.diag(noise_control_shift_per_speed) ** 2
                        * (control.velocity_ego ** 2).sum(-1).sqrt()
                )
                rot_heading = rotation_matrix(
                    self.headings_deg[:, None, None] / 360.,
                    (1, 2)
                )
                sigma_given_heading = (
                        rot_heading
                        @ sigma_loc[None, :, :]
                        @ rot_heading.permute([0, 2, 1])
                )

                # p_dxy_given_heading[dx, dy, heading]
                # = P(dx,dy|heading) ~ N(mu|heading, sigma|heading)
                p_dxy_given_heading = sumto1(torch.exp(mvnpdf_log(
                    x=torch.stack([dxs, dys], -1).unsqueeze(-2),
                    mu=-mu_loc_next_given_heading[None, None, :],
                    cov=sigma_given_heading[None, None, :]
                )), [0, 1])

                # # CHECKED: p_dxy_given_heading
                # axs = plt2.GridAxes(p_dxy_given_heading.shape[-1], 1)
                # for row, p in enumerate(npt.permute2st(p_dxy_given_heading)):
                #     plt.sca(axs[row, 0])
                #     plt.imshow(
                #         npy(p),
                #         extent=np.array([-ndxy, ndxy, -ndxy, ndxy]) * self.dx
                #     )
                #     plt.axhline(0, color='w', linestyle='--', lw=0.5)
                #     plt.axvline(0, color='w', linestyle='--', lw=0.5)
                # plt.show()

                # p_state0[x, y, heading]
                p_state = F.conv2d(
                    input=p_state.permute([2, 0, 1]).unsqueeze(0),
                    weight=p_dxy_given_heading.unsqueeze(0).permute([3, 0, 1, 2]),
                    groups=self.n_heading,
                    padding=ndxy
                ).squeeze(0).permute([1, 2, 0])

            else:
                raise ValueError()

            if use_boundary_prior:
                p_state[~self.state_incl] = 0.  # mask

            p_state = nan2v(sumto1(p_state, [0, 1], keepdim=True))
            p_state = p_state * p_angle0
            p_state = sumto1(p_state)

            # d_angle = self.headings_deg[None, :] - self.headings_deg[:, None]
            # p_heading1_given_heading0 = npt.sumto1(npt.vmpdf(
            #     prad2unitvec(d_angle / 360.),
            #     (prad2unitvec(control.dheading_deg / 360.)[None,:]
            #      * pconc2conc(self.noise_control_rotate_pconc[0]))
            #     , normalize=False
            # ), 0)

            # plt.imshow(p_heading1_given_heading0)
            # plt.show()

            # if control.velocity_ego[1] != 0:
            #     print('--')  # CHECKED

            if update_state:
                # assert to_normalize

                self.p_state0 = p_state

                # CHECKED whether location shift happens in the
                #  heading direction
                # if control.speed == 5. and control.dheading_deg[-1] == 0.:
                #     s = torch.argmax(p_state0)
                #     if (
                #             (torch.abs(self.heading_deg0.flatten()[s] - 45.)
                #              < 1e-6) and
                #             (torch.abs(self.x0.flatten()[s] - 0.) < 1e-6) and
                #             (torch.abs(self.y0.flatten()[s] - 0.) < 1e-6)
                #     ):
                #         plt.figure(figsize=[8, 1.5])
                #         for i in range(self.n_heading):
                #             ax = plt.subplot(1, self.n_heading, i + 1)
                #             plt.imshow(npy(p_state[:,:,i]).T, origin='lower')
                #             plt.xticks([])
                #             plt.yticks([])
                #         plt.show()
                #
                #         plt.figure(figsize=[4, 3])
                #         # plt.subplot(1, 2, 1)
                #         plt.plot(self.headings_deg,
                #                  npy(p_angle0.flatten()), 'ko-')
                #         plt.xticks(self.headings_deg)
                #         plt.show()
                #
                #         print('--')

                # self.plot_p_loc(heading_deg=45.)
                # plt.show()

            if return_p_state_incl:
                return p_state[self.state_incl]
            else:
                return p_state

    def ____Vectorized_sampling____(self):
        pass

    def transition_vec(
            self,
            s_true: torch.Tensor, control: torch.Tensor,
            noise_control_rotate_pconc1=None,
            noise_control_shift_per_speed1=None,
            noise_control_shift1=None,
            use_i_state=False,
            lapse_rate=0.,
            return_p_tran=False,
            to_plot=False,  # PARAM CHECKED plot keeping transition inside env
    ) -> torch.Tensor:
        """
        :param noise_control_rotate_pconc1:
        :param noise_control_shift_per_speed1:
        :param noise_control_shift1:
        :param use_i_state:
        :param lapse_rate:
        :param return_p_tran:
        :param to_plot:
        :param s_true: [batch, (x, y, heading)]
        :param control: [batch, (forward, rightward, dheading, speed)]
        :return: s_true[batch, (x, y, heading)]
        """
        # s_true0 = npt.dclone(s_true)

        assert self.dt == 1, \
            'Noise scaling for dt != 1 not implemented yet! ' \
            'For high-res grid fields, use interpolation of trajectory ' \
            'and smoothing while keeping dt=1.'
        # if self.dt != 1:
        #     print('check for dt != 1')

        control = control.clone()

        # DEF: xy[batch, (x, y)], th[batch]
        if use_i_state:
            xy = torch.stack([self.x[s_true], self.y[s_true]], -1)
            th = self.heading_deg[s_true]
        else:
            xy = s_true[..., :2]
            th = s_true[..., 2]

        xy00 = xy.clone()  # for plotting
        th00 = th.clone()

        # --- Turn first
        if noise_control_rotate_pconc1 is None:
            noise_control_rotate_pconc1 = self.noise_control_rotate_pconc[0]
        if noise_control_shift_per_speed1 is None:
            noise_control_shift_per_speed1 = \
                self.noise_control_shift_per_speed[:]
            # assert (
            #     self.noise_control_shift_per_speed[0]
            #     == self.noise_control_shift_per_speed[1]
            # )
            # noise_control_shift_per_speed1 = \
            #     self.noise_control_shift_per_speed[0]
        if noise_control_shift1 is None:
            assert (
                    self.noise_control_shift[0]
                    == self.noise_control_shift[1]
            )
            noise_control_shift1 = self.noise_control_shift

        # DEBUGGED: (forward, rightward) -> dxy,
        #  in preparation for rotate_to_align.
        dxy_ego = torch.stack([control[..., 0], -control[..., 1]], -1)
        dth = control[..., 2]
        if control.shape[-1] == 4:
            print('speed is redundant with dxy_ego and is removed!')
            speed = (control[:, :2] ** 2).sum(-1).sqrt()
            control[speed == 0, :2] = 0.
            if any(speed != 0):
                control[speed != 0, :2] = control[:, :2] / speed[speed != 0]
            control = control[:, :3]
            assert control.shape[-1] == 3, \
                'speed is redundant with dxy_ego and is removed!'

        # Use (wrapped) normal instead of vmpdf
        # so that scaling of variance with dt is straightforward
        variance = pconc2var(noise_control_rotate_pconc1)
        # can skip wrapping when stdev is small
        assert (variance < 0.25 ** 2).all(), \
            'we currently skip wrapping, so you need to use a small variance' \
            '= high pconc!'

        dheading_deg = (
                               self.headings_deg[None, :]
                               - (th + dth)[:, None]
                               + 180.
                       ) % 360. - 180.  # ranges from -180 to +180

        p_th = torch.softmax(log_normpdf(
            dheading_deg / 360., 0.,
            (npt_tensor(variance) * self.dt).sqrt()
        ), 1)
        # p_th = npt.vmpdf_prad_pconc(
        #     self.headings_deg[None, :] / 360.,
        #     (th / 360. + dth / 360.)[:, None],
        #     npt_tensor(noise_control_rotate_pconc1)
        # )
        assert not torch.isnan(p_th).any()
        i_th = categrnd(p_th)
        th = self.headings_deg[i_th]

        # # CHECKED
        # if th != 0:
        #     print('th != 0 although policy=straight!')
        #     print(p_th)
        #     print('--')

        dxy_th = torch.stack([torch.cos(th / 180. * np.pi),
                              torch.sin(th / 180. * np.pi)], -1)

        # --- Translate after turning
        # dxy_allo[batch, xy]
        dxy_allo = geom.h2e(geom.rotate_to_align(
            geom.e2h(F.pad(dxy_ego, [0, 1])),
            geom.e2h(F.pad(dxy_th, [0, 1])),
            True
        ))[:, :2]
        # print(np.r_[
        #     npy(xy[0]),
        #     npy(th[0]),
        #     np.sign(np.round(npy(dxy_allo[0]), 6))
        # ])  # CHECKED

        if (noise_control_shift1[:] > 0).any():
            raise NotImplementedError()

        if (noise_control_shift_per_speed1[:] == 0).all():
            # no noise
            pass
        else:
            if noise_control_shift_per_speed1[1] > 0:
                raise NotImplementedError()

            if self.noise_shift_kind == 'g':
                # if th[0] != 180:  # CHECKED
                #     print('--')

                if self.n_heading not in [1, 2, 4]:
                    raise NotImplementedError()
                speed = (dxy_allo ** 2).sum(-1).sqrt()

                incl = speed >= 1e-6
                dxy_allo[~incl] = 0.

                if incl.any():
                    p = self.noise_shift_gampdf(
                        noise_control_shift_per_speed1[0],
                        speed[incl])

                    idx = categrnd(probs=p)

                    # dxy_allo0 = dxy_allo
                    dxy_allo[incl] = (
                            dxy_allo[incl] / speed[incl]  # unit vector
                            * idx.double() * self.dx)
                    # print(f'{dxy_allo0[0, 0]:5.3f}, {dxy_allo0[0, 1]:5.3f}')
                    # print(f'{dxy_allo[0, 0]:5.3f}, {dxy_allo[0, 1]:5.3f}')
                    # print('--')  # CHECKED

                    # print(f'categrnd: {int(idx)}')  # CHECKED
                    # if idx > 1:  # CHECKED
                    #
                    #     print(idx)
                    #     print('xy: ' + np2.joinformat(xy, '%1.3f'))
                    #     print('dxy: ' + np2.joinformat(dxy_allo, '%1.3f'))
                    #     print('xy + dxy: '
                    #           + np2.joinformat(xy + dxy_allo, '%1.3f'))
                    #     if dxy_allo[0, 0] >= 0:
                    #         print('--')

            elif self.noise_shift_kind == 'n':
                assert (
                        noise_control_shift1[0] ==
                        noise_control_shift1[1]
                ), 'Only admits isometric noise for now'
                assert (
                        noise_control_shift_per_speed1[0] ==
                        noise_control_shift_per_speed1[1]
                ), 'Only admits isometric noise for now'

                noise_dxy = normrnd(
                    0.,
                    1., dxy_allo.shape
                ).squeeze(-1)
                # NOTE: no need to scale noise_control_shift_per_speed1 ** 2
                #  with self.dt in Observer2D.transition_vec(),
                #  because
                #  noise_control_shift_per_speed1 ** 2
                #  == noise / (distance/dt) * dt = noise / distance,
                #  so time is already considered.
                #  Note that "speed" here is distance per time step.
                noise_dxy = noise_dxy * torch.sqrt(
                    (dxy_allo ** 2).sum(-1, keepdim=True).sqrt()
                    * noise_control_shift_per_speed1[0] ** 2
                    + noise_control_shift1[0] ** 2 * self.dt
                )
                dxy_allo = dxy_allo + noise_dxy

            else:
                raise ValueError()

            # # CHECKED
            # print('xy: ' + np2.joinformat(xy, '%1.3f'))
            # print('dxy: ' + np2.joinformat(dxy_allo, '%1.3f'))
            # print('xy + dxy: '
            #       + np2.joinformat(xy + dxy_allo, '%1.3f'))

            xy = xy + dxy_allo

        # --- find the nearest state along the trajectory
        is_inside = self.env.is_inside(npy(xy))

        s_true0 = s_true.clone()
        s_true = empty_like(s_true)
        s_true[is_inside] = s_true0[is_inside]

        if (~is_inside).any():
            print('--')

        # xy[batch, xy], th[batch]
        for i_batch, (is_inside1, xy1, th1) in enumerate(
                zip(is_inside, xy, th)
        ):
            xy0 = xy1.clone()
            if is_inside1:
                # choose the closest state
                i_states = npt_arange(self.n_state_incl)
                incl = (self.heading_deg == th1)
                i_state_dst = i_states[incl][torch.argmin(
                    (xy1[0] - self.x[incl]).abs()
                    + (xy1[1] - self.y[incl]).abs()
                )]
            else:
                dx = np.round(np.cos(deg2rad(npy(th1))), 6)
                dy = np.round(np.sin(deg2rad(npy(th1))), 6)

                if dx != 0 and dy == 0:
                    i_states = npt_arange(self.n_state_incl)
                    incl = (
                            (self.heading_deg == th1)
                            & issimilar(self.y, xy1[1]))
                    i_state_dst = i_states[incl][torch.argmin(
                        (xy1[0] - self.x[incl]) * np.sign(dx))]
                elif dx == 0 and dy != 0:
                    i_states = npt_arange(self.n_state_incl)
                    incl = (
                            (self.heading_deg == th1)
                            & issimilar(self.x, xy1[0]))
                    i_state_dst = i_states[incl][torch.argmin(
                        (xy1[1] - self.y[incl]) * np.sign(dy))]
                elif dx == 0 and dy == 0:
                    raise ValueError('should not be outside if dx=dy=0')
                else:
                    raise NotImplementedError(
                        'handling non-cardinal movement is not '
                        'implemented')

                if to_plot:  # CHECKED transition_vec()
                    print(
                        f'xy00: {joinformat(xy00)} '
                        f'xy0:  {joinformat(xy0)}, '
                        f'xy1:  {joinformat(xy1)}'
                        f's_true: {s_true}')

                    i_batch = 0
                    th1 = th[0]
                    rec = self.gen.agent_state.get_record()

                    axs = GridAxes(1, 3, widths=2, heights=2)
                    plt.sca(axs[0, 0])
                    self.gen.agent_state.set_state(
                        loc_xy=xy00[i_batch], heading_deg=th00[i_batch])
                    self.gen.agent_state.quiver()
                    self.gen.env.plot_walls()
                    plt.title('bef transition')

                    plt.sca(axs[0, 1])
                    self.gen.agent_state.set_state(
                        loc_xy=xy0, heading_deg=th1)
                    self.gen.agent_state.quiver()
                    self.gen.env.plot_walls()
                    plt.title('aft transition')

                    plt.sca(axs[0, 2])
                    self.gen.agent_state.set_state(
                        loc_xy=xy1, heading_deg=th1)
                    self.gen.agent_state.quiver()
                    self.gen.env.plot_walls()
                    plt.title('brought wi env')

                    self.gen.agent_state.reinstate_record(
                        rec=dictlist2listdict(rec)[0])
                    plt.show()
                    print('--')

            xy1 = torch.stack([self.x[i_state_dst], self.y[i_state_dst]])
            xy[i_batch] = xy1

            s_true[i_batch] = self.get_i_state(
                loc=xy1, heading_deg=th1)

        # # UNUSED: find the state in the env with the minimum Euclidean distance
        # if self.n_state_loc_incl == self.nx_incl * self.ny_incl:
        #     ix = npt.discretize(xy[:, 0], self.xs_incl)
        #     iy = npt.discretize(xy[:, 1], self.ys_incl)
        #     ith = torch.argmin(
        #         npt.circdiff(th[:, None], self.headings_incl[None, :],
        #                      maxangle=360.).abs(),
        #         -1)
        #
        #     s_true = npt.ravel_multi_index(
        #         [npy(v) for v in [ix, iy, ith]],
        #         shape=[
        #             self.nx_incl,
        #             self.ny_incl,
        #             self.nheading_incl
        #         ]
        #     )
        # else:
        #     s_true = torch.argmin(
        #         torch.abs(xy[:, [0]] - self.x[None, :])
        #         + torch.abs(xy[:, [1]] - self.y[None, :])
        #         + torch.abs(th[:, None] - self.heading_deg[None, :])
        #         , -1)

        if lapse_rate > 0:
            incl = npt_rand(s_true.shape[0]) < lapse_rate
            if incl.any():
                s_true[incl] = npt_randint(
                    self.n_state_incl,
                    [int(incl.sum())])

        # DEBUGGED to snap to the nearest state even when use_i_state = False
        if not use_i_state:
            s_true = torch.stack([
                self.x[s_true],
                self.y[s_true],
                self.heading_deg[s_true]
            ], -1)

        if return_p_tran:
            p_th_chosen = p_th[th]
            raise NotImplementedError()

        # if (s_true == s_true0).all():
        #     print('didnt move!')
        if True is True:  # avoids unreachable code error.
            raise DeprecationWarning('Not passing validation - use transition() instead')
        return s_true

    def noise_shift_gampdf(
            self,
            noise_control_shift_per_speed1: torch.Tensor,
            speed: torch.Tensor
    ) -> torch.Tensor:
        """

        :param noise_control_shift_per_speed1: [batch]
        :param speed: [batch]
        :return: p[batch, dx]
        """

        if noise_control_shift_per_speed1.ndim == 0:
            noise_control_shift_per_speed1 = noise_control_shift_per_speed1[None]
        speed, noise_control_shift_per_speed1 = torch.broadcast_tensors(
            speed, noise_control_shift_per_speed1)

        # first get gammapdf and then sample from it
        dx = (self.xs_incl - self.xs_incl[0]).clamp_min(1e-3)

        n_batch = len(speed)
        nx = len(dx)
        # p[batch, dx]
        p = npt_zeros([n_batch, nx])
        nonzero_speed = speed > 1e-6
        if nonzero_speed.any():
            p[nonzero_speed] = torch.softmax(
                gamma_logpdf_ms(
                    dx[None], speed[nonzero_speed],
                    speed[nonzero_speed] * noise_control_shift_per_speed1[0]
                ), -1)

            # plt.plot(npy(p.flatten()))
            # plt.show()
            # print('--')

        p[speed <= 1e-6, 0] = 1
        p[speed <= 1e-6, 1:] = 0
        p = sumto1(p, -1)
        return p

    @staticmethod
    def measurement_step_vec_given_vislik(
            p_state: torch.Tensor, vislik: torch.Tensor
    ) -> torch.Tensor:
        """

        :param p_state: [batch, state_belief]
        :param vislik: [batch_state_true, state_belief]
        :return: p_state: [batch, state_belief]
        """
        return prod_sumto1(p_state, vislik, dim=1)

    def measurement_step_vec(
            self,
            i_state_beliefs: torch.Tensor,
            rate_retina: torch.Tensor,
            tactile_input: torch.Tensor,
            duration_times_gain=1.,
            use_vision=True,
            use_touch=True,
    ) -> torch.Tensor:
        """

        :param i_state_beliefs: [batch, particle]
        :param rate_retina: [batch, x, y, channel]
        :param tactile_input: [batch, dheading]
        :param duration_times_gain: scalar
        :param use_vision:
        :param use_touch:
        :return: loglik[batch, particle]
        """
        n_batch, n_particle = i_state_beliefs.shape

        # visloglik[batch, particle]
        visloglik = self.get_loglik_retina_vectorized(
            rate_retina[:, None, :].expand(
                [n_batch, n_particle] + list(rate_retina.shape[1:])
            ).reshape(
                [n_batch * n_particle] + list(rate_retina.shape[1:])),
            duration=duration_times_gain,
            i_states_belief=i_state_beliefs.flatten()
        ).reshape([n_batch, n_particle])

        # tactile_loglik[batch, particle]
        tactile_loglik = self.get_loglik_tactile_vectorized(
            tactile_input[:, None, :], i_state_beliefs)

        loglik = npt_zeros_like(visloglik)
        if use_vision:
            loglik = loglik + visloglik
        if use_touch:
            loglik = loglik + tactile_loglik

        return loglik

    def get_loglik_tactile_vectorized(
            self, tactile_input: torch.Tensor,
            i_state_beliefs: torch.Tensor,
            return_by_side=False,
    ) -> torch.Tensor:
        return self.tactile_sensor.log_prob(
            tactile_input, i_state_beliefs, return_by_side=return_by_side)

    def measurement_step_particle_filter(
            self,
            i_state: torch.Tensor,
            retinal_image: torch.Tensor,
            duration=None,
    ):
        """

        :param i_state: [batch, particle]
        :param retinal_image: [batch, x, y, c]
        :param duration: time window during which spikes are counted
        :return: p_state[batch, particle], normalized across particles
            within each batch
        """
        if duration is None:
            duration = self.gain_retina
        loglik = self.get_loglik_retina_vectorized(
            retinal_image,
            i_states_belief=i_state,
            duration=duration,
        )
        return torch.log_softmax(loglik, -1).exp()

    @staticmethod
    def resample_particle_filter(i_states: torch.Tensor, p_states) -> torch.Tensor:
        """
        :param i_states: [batch, particle]
        :param p_states: [batch, particle]
        :return: i_states[batch, particle] resampled in proportion to p_states
            within each batch
        """
        n_particle = i_states.shape[-1]
        return torch.stack([
            i_states1[distributions.Categorical(probs=p_states1).sample(torch.Size([n_particle]))]
            for i_states1, p_states1 in zip(i_states, p_states)
        ])

    def prediction_step_samp(
            self,
            p_state: torch.Tensor, control: torch.Tensor,
            n_particle=20,
            lapse_rate=0.,
            **kwargs
    ) -> torch.Tensor:
        """
        :param lapse_rate:
        :param p_state: [batch, state]
        :param control: [batch, (dx_ego, dy_ego, dth, speed)]
        :param n_particle:
        :return:
        """

        # --- First sample states from p_state (= resampling step)
        n_batch, n_state = p_state.shape

        # DEF: states[particle, batch]
        states = categrnd(p_state, sample_shape=[n_particle]
                              )
        # NOTE: using expand() instead of repeat() didn't save time
        # control = control[None, :].expand(
        #     [n_particle] + list(control.shape)
        # ).reshape([n_batch * n_particle, 4])
        control = control[None, :].repeat([n_particle, 1, 1]).reshape([
            n_particle * n_batch, 4])

        # --- Then sample next states using control (= proposal)
        states = self.transition_vec(
            states.flatten(), control,
            use_i_state=True,
            lapse_rate=lapse_rate,
            **kwargs
        )

        # # DEBUGGED: should be done within transition_vec
        # if lapse_rate > 0:
        #     incl = npt.rand(states.shape[0]) < lapse_rate
        #     if incl.any():
        #         states[incl] = npt.randint(
        #             self.n_state_incl,
        #             [n_batch])

        # --- Put back into p_state (in prep for the measurement_step,
        #  which gives the particle weights)
        # NOTE: using expand() instead of repeat() didn't save time
        # p_state = npt_zeros_like(p_state)[None, :].expand(
        #     [n_particle] + list(p_state.shape)
        # ).reshape([n_batch * n_particle, -1])
        # p_state = npt_zeros_like(p_state)[None, :].repeat(
        #     [n_particle, 1, 1]).reshape([n_batch * n_particle, -1])
        # NOTE: using zeros() instead of repeat().reshape() saved time
        p_state = aggregate([
            np.tile(np.arange(n_batch)[None, :], [n_particle, 1]).flatten(),
            npy(states).flatten().astype(int),
        ], np.ones([n_batch * n_particle]), size=[n_batch, n_state]
        ) / n_particle
        # size_p_state = np.prod(list(p_state.shape))
        # p_state = npt_zeros([
        #     n_batch * n_particle, size_p_state // n_batch
        # ])
        # p_state.scatter_add_(1, states[:, None],
        #                      npt_ones([n_batch * n_particle, 1]))
        # p_state = p_state.reshape([n_particle, n_batch, -1]).sum(0) / n_particle
        if True is True:  # Avoids unreachable code error
            raise DeprecationWarning('Not tested! Use prediction_step() instead.')
        return p_state

    def ____SELF_MOTION_OBSERVATION____(self):
        pass

    def ____VISUAL_LIKELIHOOD____(self):
        pass

    def measurement_step(
            self, meas, control=None, p_state_incl=None,
            skip_visual=False,
            update_state=False,
            duration=None,
    ):
        """
        :param skip_visual:
        :param update_state:
        :param duration:
        :type meas: Measurement
        :type control: Control
        :type p_state_incl: torch.Tensor
        :return: p_state_incl (updated)
        :rtype: torch.Tensor
        """
        if duration is None:
            duration = self.gain_retina
        if p_state_incl is None:
            if self.gen.is_binned_space(self.gen.env):
                p_state_incl = self.p_state_incl
            else:
                p_state_incl = self.particles.weight

        if not skip_visual:
            loglik = self.get_loglik_retina_vectorized(
                self.gen.measure_retinal_image(self.gen.agent_state)[None, :],
                duration=duration,
            )[0]

            log_p_state = torch.log(p_state_incl) + loglik
            log_p_state = log_p_state - torch.max(log_p_state)
            p_state_incl = torch.exp(log_p_state)
            p_state_incl = p_state_incl / torch.sum(p_state_incl)

        if update_state:
            if self.gen.is_binned_space(self.gen.env):
                self.p_state_incl = p_state_incl
            else:
                self.particles.weight = p_state_incl

        return p_state_incl

    def resample_cont(self, **kwargs):
        self.particles.resample(**kwargs)
        self.jitter()
        return

    def jitter(self):

        new_loc = self.particles.jitter()
        # --- Check new locations
        trajectories = torch.stack([self.particles.loc[..., :2], new_loc[..., :2]], 1)
        intersections = self.intersection_lines_vvec(trajectories)
        self.particles.loc[..., :2] = torch.where(torch.isinf(intersections).any(-1, keepdim=True),
                                         new_loc[..., :2],
                                         0.05 * self.particles.loc[..., :2] + 0.95 * intersections)
        self.particles.loc[..., 2] = new_loc[..., 2]
        return

    def get_loglik_retina(self, rate_img, duration=None):
        """
        Assume independent Poisson firing rate.
        See Dayan & Abbott Eq. 3.30
        :param duration:
        :param rate_img: [x_retina, y_retina, channel]
        :type rate_img: torch.Tensor
        :return: loglik[state]
        :rtype torch.Tensor
        """
        if self.dt != 1:
            raise NotImplementedError(
                'Yet to scale noise with dt in obs.get_loglik_retina()'
                'when dt != 1\n'
                'Likely need to consider gain_retina separately from dt '
                'for retina')

        if duration is None:
            duration = self.gain_retina
        loglik = npt_zeros(self.n_state_incl)
        for s, tuning_img in enumerate(self.img_given_state):
            loglik[s] = (
                                torch.sum(
                                    rate_img * (torch.log(tuning_img) + np.log(
                                        duration))
                                ) - self.get_sum_img_given_state()[s]
                        ) * duration
        return loglik

    def get_loglik_retina_vectorized(
            self, rate_imgs, duration=None,
            i_states_belief: torch.Tensor = None
    ):
        """
        :param duration:
        :param rate_imgs: [state_true_batch, x_retina, y_retina, channel] : this
            should be sampled from Poisson distribution with the rate *
            duration as the rate parameter then divided by the duration to
            give rate (the duration will be considered in the equation)
        :param i_states_belief: [state_belief_to_consider]
            or [state_true_batch, state_belief_to_consider_for_batch]
        :return: loglik[state_true_batch, state_belief]
        """
        if duration is None:
            duration = self.gain_retina
        # NOTE: potential over/underflow with duration >= 4 ?
        # assert duration < 4, 'potential over/underflow with duration >= 4'
        if i_states_belief is None:
            return (
                    torch.sum(
                        # [state_true_batch, (state_belief), x, y, chn]
                        rate_imgs.unsqueeze(1)
                        #
                        # [1, state_belief, x, y, chn]
                        * (self.log_img_given_state()[None, :]
                           + np.log(duration)),
                        [-3, -2, -1]
                    ) - self.get_sum_img_given_state()[None, :]  # [1, state_belief]
            ) * duration
        elif i_states_belief.ndim == 1:
            return (
                    torch.sum(
                        # [state_true_batch, x, y, chn]
                        rate_imgs
                        #
                        # [state_beliefs_to_consider, x, y, chn]
                        * (self.log_img_given_state()[i_states_belief]
                           + np.log(duration)),
                        [-3, -2, -1]
                    ) - self.get_sum_img_given_state()[i_states_belief]  # [state_belief]
            ) * duration
        elif i_states_belief.ndim == 2:
            return (
                    torch.sum(
                        # [state_true_batch, x, y, chn]
                        rate_imgs[:, None, :]
                        #
                        # [state_true_batch, state_beliefs_to_consider, x, y, chn]
                        * (indexshape(self.log_img_given_state(), i_states_belief)
                           + np.log(duration)),
                        [-3, -2, -1]
                    ) - indexshape(self.get_sum_img_given_state(), i_states_belief)
            ) * duration
        else:
            raise ValueError()

    def ____MAXLIK____(self):
        pass

    def get_state_maxlik(
            self, state: AgentState = None,
            to_update_state_template=False
    ):
        """
        :return: (state_maxlik, i_state_maxlik, heading_deg_maxlik)
        """
        if to_update_state_template:
            state = self.state_template
        elif state is None:
            state = deepcopy(self.state_template)

        i_state_maxlik = torch.argmax(self.p_state_incl)
        loc_xy_maxlik = npt_tensor([
            self.x[i_state_maxlik].clone(),
            self.y[i_state_maxlik].clone()
        ])
        heading_deg_maxlik = self.heading_deg[i_state_maxlik].clone()
        # state_maxlik = AgentState(
        #     **{k:state.__dict__[k].clone() for k in state.keys}
        # )
        state.set_state(loc_xy=loc_xy_maxlik, heading_deg=heading_deg_maxlik)

        # state.loc[0] = self.x[i_state_maxlik].clone()
        # state.loc[1] = self.y[i_state_maxlik].clone()
        # state.heading[0] = torch.cos(npt.deg2rad(heading_deg_maxlik))
        # state.heading[1] = torch.sin(npt.deg2rad(heading_deg_maxlik))

        return state, i_state_maxlik, heading_deg_maxlik
