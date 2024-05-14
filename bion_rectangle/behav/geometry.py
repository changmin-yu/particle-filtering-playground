from scipy.io import loadmat
import numpy as np
from importlib import reload
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from pprint import pprint
import os
from warnings import warn

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data_util
from torch.distributions import MultivariateNormal, Uniform

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from bion_rectangle.utils import plt2
from bion_rectangle.utils import yktorch as ykt
from bion_rectangle.utils import numpytorch as npt

from bion_rectangle.hyperspherical_vae import von_mises_fisher as vmf

#%%
reload(ykt)
reload(npt)
npy = npt.npy
npys = npt.npys

#%% Utility
def rad2pang(rad):
    return rad / np.pi / 2. % 1.

def pang2rad(pang):
    return pang * 2. * np.pi

def rand(shape, low=0, high=1):
    d = Uniform(low=low, high=high)
    return d.rsample(shape)

def mvnrnd(mu, sigma, sample_shape=torch.Size([])):
    d = MultivariateNormal(loc=mu, covariance_matrix=sigma)
    return d.rsample(sample_shape)

def vmpdf(x, mu, scale, normalize=True):
    vm = vmf.VonMisesFisher(mu, scale + torch.zeros([1,1]))
    p = torch.exp(vm.log_prob(x))
    if normalize:
        p = npt.sumto1(p)
    return p

def pang2xy(pang, dim=1):
    rad = pang2rad(pang)
    return torch.cat((torch.cos(rad.view([-1,1])),
                      torch.sin(rad.view([-1,1]))), dim=dim)

def rotation_matrix(pang, dim=(0,1)):
    rad = pang2rad(pang)
    return torch.cat((
        torch.cat((torch.cos(rad), -torch.sin(rad)), dim[1]),
        torch.cat((torch.sin(rad), torch.cos(rad)), dim[1])), dim[0])

def rotate(v, pang):
    rotmat = rotation_matrix(pang[:,None,None], (1,2))
    return v.matmul(rotmat)

def plot_centroid_mvn(dist_mvn, *args, **kwargs):
    out = plt2.plot_centroid(np.array(dist_mvn.loc),
                              np.array(dist_mvn.covariance_matrix),
                              *args, **kwargs)
    # plt.axis('equal')
    return out

def beautify_plot():
    plt.axis('equal')
    plt.axis('square')
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))

#%%
class SimUniqueRoundNoproj(data_util.Dataset):
    """
    Simulation of unique, round (no orientation), 2D (no projection) objects.
    Cast as a regression problem.
    Data: batch x variables x object x view
    - Variables:
    -- 2D coordinates (x, y) & uncertainty (sig_x, sig_y, sig_xy)
    -- 3D orientation & uncertainty (3D canonical vector for von Mises-Fisher
    distribution)
    -- Seen (boolean scalar)
    - Object
    -- first is always the treasure; the rest are other landmarks
    -- total n_obj
    - View: 0 or 1 (first or second view)
    Target: batch x variables
    - Variables: 2D coordinates (x, y) of the treasure.

    """
    DIM_BATCH = 0
    DIM_VIEW = 1 # previously 2
    DIM_OBJECT = 2 # previously 3
    DIM_VARIABLE = 3 # previously 1

    OBJ_REWARD = 0

    def __init__(self, n_batch_total=5, n_obj=4, n_view=2,
                 sigma_per_dist_meas=0.1,
                 sigma_min_meas=0.05,
                 sigma_per_dist_control=0.05,
                 sigma_min_control=0.01,
                 n_dim_ori=0,
                 to_add_rotation=False,
                 train=True,
                 seed=None,
                 prior_vehicle_mu=None,
                 prior_vehicle_sigma=None,
                 control_vehicle_mu=None,
                 control_vehicle_sigma=None
                 ):
        """
        :param n_batch_total:
        :param n_obj:
        :param n_view:
        :param sigma_per_dist_meas:
        :param sigma_min_meas:
        :param to_add_rotation:
        :param train:
        :param seed:
        :param prior_vehicle_mu: None or tensor[batch, view, 1, vars]
        :param prior_vehicle_sigma: None or tensor[batch, view, 1, vars, vars]
        """
        self.n_batch_total = n_batch_total
        self.n_obj = n_obj
        self.n_view = n_view
        self.train = train

        self.sigma_per_dist_meas = sigma_per_dist_meas
        self.sigma_min_meas = sigma_min_meas

        self.sigma_per_dist_control = sigma_per_dist_control
        self.sigma_min_control = sigma_min_control

        #%%
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Sample 2D landmark locations & orientations
        # Landmarks are assumed stationary, so sample for only one view
        self.n_dim_loc = 2
        self.dims_loc = torch.arange(0, self.n_dim_loc,
                                     dtype=torch.long)
        self.distrib_landmark_loc = MultivariateNormal(
            loc=torch.zeros(self.n_dim_loc),
            covariance_matrix=torch.eye(self.n_dim_loc)
        )
        self.landmark_loc = self.distrib_landmark_loc.rsample(
                (self.n_batch_total, 1, self.n_obj)
            )

        # landmarks are assumed isometric in this class
        # (i.e., no orientation)
        self.to_add_rotation = to_add_rotation
        self.n_dim_ori = n_dim_ori
        if to_add_rotation and n_dim_ori == 0:
            raise ValueError('if to_add_rotation=True, n_dim_ori must be >0')
        self.dims_ori = torch.arange(
            self.n_dim_loc,
            self.n_dim_loc + self.n_dim_ori,
            dtype=torch.long)
        self.landmark_ori_pang = rand((
            self.n_batch_total,
            1,
            self.n_obj,
            self.n_dim_ori,
        ))

        # All dims per obj
        self.n_dim_per_obj = self.n_dim_loc + self.n_dim_ori

        # 2. Sample vehicle location/orientation at each view
        if prior_vehicle_mu is None:
            prior_vehicle_mu = torch.zeros(
                1, 1, 1, self.n_dim_loc)
        if prior_vehicle_sigma is None:
            prior_vehicle_sigma = npt.attach_dim(
                torch.eye(self.n_dim_loc), 3)

        if control_vehicle_mu is None:
            # Sample every view independently
            prior_vehicle_mu = npt.repeat_to_shape(
                npt.prepend_to_ndim(prior_vehicle_mu, 4),
                shape=[self.n_batch_total, self.n_view, 1, self.n_dim_loc]
            )
            prior_vehicle_sigma = npt.repeat_to_shape(
                npt.prepend_to_ndim(prior_vehicle_sigma, 5),
                shape=[self.n_batch_total, self.n_view, 1,
                       self.n_dim_loc, self.n_dim_loc]
            )
            self.distrib_vehicle_loc = MultivariateNormal(
                loc=prior_vehicle_mu,
                covariance_matrix=prior_vehicle_sigma
            )
            self.vehicle_loc = self.distrib_vehicle_loc.rsample()

            # No control in this case.
            # Different from ZERO control: rather, vehicle locations are
            # sampled independently across views.
            self.distrib_vehicle_control = None
            self.vehicle_control = None
        else:
            # First sample view 0 only
            prior_vehicle_mu = npt.repeat_to_shape(
                npt.prepend_to_ndim(prior_vehicle_mu, 4),
                shape=[self.n_batch_total, 1, 1, self.n_dim_loc]
            )
            prior_vehicle_sigma = npt.repeat_to_shape(
                npt.prepend_to_ndim(prior_vehicle_sigma, 5),
                shape=[self.n_batch_total, 1, 1,
                       self.n_dim_loc, self.n_dim_loc]
            )
            self.distrib_vehicle_loc = MultivariateNormal(
                loc=prior_vehicle_mu,
                covariance_matrix=prior_vehicle_sigma
            )
            self.vehicle_loc = self.distrib_vehicle_loc.rsample()

            # Then sample the control vector
            control_vehicle_mu = npt.repeat_to_shape(
                npt.prepend_to_ndim(control_vehicle_mu, 4),
                shape=[self.n_batch_total, self.n_view - 1, 1, self.n_dim_loc]
            )
            control_vehicle_sigma = npt.repeat_to_shape(
                npt.prepend_to_ndim(control_vehicle_sigma, 5),
                shape=[self.n_batch_total, self.n_view - 1, 1,
                       self.n_dim_loc, self.n_dim_loc]
            )
            self.distrib_vehicle_control = MultivariateNormal(
                loc=control_vehicle_mu,
                covariance_matrix=control_vehicle_sigma
            )
            self.vehicle_control = self.distrib_vehicle_loc.rsample()

            # Then add the control to view 0 to get later views
            cum_control = torch.cumsum(self.vehicle_control,
                                       self.DIM_VIEW)
            self.vehicle_loc = torch.cat((
                self.vehicle_loc,
                self.vehicle_loc + self.vehicle_control
            ), self.DIM_VIEW)

        # There is no uncertainty regarding vehicle orientations
        # (although it is hidden)
        vehicle_ori_pang_size = (
            self.n_batch_total,
            self.n_view,
            1,
            self.n_dim_ori
        )
        if self.to_add_rotation:
            self.vehicle_ori_pang = rand(vehicle_ori_pang_size)
        else:
            self.vehicle_ori_pang = torch.zeros(vehicle_ori_pang_size)

        # Concatenate to get ground truths
        self.truths_landmark = torch.cat((
            self.landmark_loc, self.landmark_ori_pang
        ), self.DIM_VARIABLE)
        self.truths_vehicle = torch.cat((
            self.vehicle_loc, self.vehicle_ori_pang
        ), self.DIM_VARIABLE)

        # 3. Compute distribution of observation on each viewing as
        #    2D landmark locations/orientations relative to the vehicle.
        #    Rotate locations considering vehicle orientation on each view.
        self.obs_loc = self.landmark_loc - self.vehicle_loc

        # Then turn it using a rotation matrix
        def rot_loc(loc, pang):
            rotmat = rotation_matrix(pang[:,None,None], dim=(1,2))

            # loc_new[B,O,D,1]
            loc_new = rotmat[:,None,:,:] @ loc[:,:,self.dims_loc,None]
            return loc_new[:,:,:,0]

        dloc = self.landmark_loc - self.vehicle_loc
        self.obs_mu = torch.empty((
            self.n_batch_total,
            self.n_view,
            self.n_obj,
            self.n_dim_loc,
        ))
        self.obs_sigma = torch.empty((
            self.n_batch_total,
            self.n_view,
            self.n_obj,
            self.n_dim_loc,
            self.n_dim_loc,
        ))
        for view in range(self.n_view):
            if self.n_dim_ori > 0:
                obs_mu = rot_loc(dloc[:,view,:,:],
                                 -self.vehicle_ori_pang[:,view,0,0])
            else:
                obs_mu = dloc[:,view,:,:]
            # [B,O,D]
            obs_sigma = \
                self.distsq2sigmasq(
                    torch.sum(obs_mu ** 2, dim=-1)[:,:,None,None]) \
                * npt.attach_dim(torch.eye(self.n_dim_loc), 2, 0)

            self.obs_mu[:,view,:,:] = obs_mu
            self.obs_sigma[:,view,:,:,:] = obs_sigma

        # 4. Sample observations
        obs_loc, obs_ori = self.sample_obs()
        self.obs_loc[:,:,:,self.dims_loc] = obs_loc

        # Relative orientations
        # TODO: sample object orientations from vMF
        self.obs_ori_pang = obs_ori

        # 5. Concatenate to generate the data and the target
        self.data = torch.cat((self.obs_loc, self.obs_ori_pang),
                              self.DIM_VARIABLE)
        self.target = npt.attach_dim(self.data[:,:,:,0], 0, 1)

    def sample_obs(self, sample_shape=torch.Size([])):
        """
        :param sample_shape: prepended to obs_mu and obs_sigma's shape
        :return: obs_loc, obs_ori,
            each with shape [sample_shape, batch, view, obj, variable]
        """
        obs_loc = mvnrnd(self.obs_mu, self.obs_sigma,
                         sample_shape=sample_shape)
        obs_ori = (self.landmark_ori_pang - self.vehicle_ori_pang) % 1.
        return obs_loc, obs_ori

    ############################################################################
    def ____NOISE_MODEL____(self):
        pass
    
    def distsq2sigmasq(self, distsq):
        return distsq * self.sigma_per_dist_meas ** 2 + self.sigma_min_meas ** 2

    def dist2sigma(self, dist):
        return torch.sqrt(self.distsq2sigmasq(dist ** 2))

    def meas_input2noise(self, meas_input):
        """
        Return meas_noise given meas_input
        :param meas_input: [batch_size, obj, xy], aka "obs" in ssm.update()
        :type meas_input: torch.Tensor
        :return: meas_noise: [batch_size] + [n_obj * n_dim_per_obj] * 2
        :rtype: torch.Tensor
        """
        n_obj = meas_input.shape[-2]
        # dist[batch, obj]
        dist = torch.sqrt(torch.sum(meas_input ** 2, -1))
        # noise_obj[batch, obj]
        noise_obj = self.dist2sigma(dist)
        # All dims except last two [obj, xy] are for the batch
        ndim_batch = meas_input.ndimension() - 2
        eye1 = npt.attach_dim(torch.eye(self.n_dim_per_obj), ndim_batch, 0)

        meas_noise = npt.block_diag([
            eye1 * npt.attach_dim(noise_obj[:,obj], 0, 2) for obj in
            torch.arange(n_obj)
        ])
        return meas_noise

    def control_input2noise(self, control_input):
        """
        Return control_noise given control_input
        It is assumed that control is given only to the vehicle for now.
        :param control_input: [batch_size, xy], aka "control" in ssm.update()
        :type control_input: torch.Tensor
        :return: control_noise: [batch_size] + [(n_obj + 1) * n_dim_per_obj] * 2
        :rtype: torch.Tensor
        """
        # if control_input.shape[-2] != 1:
        #     raise ValueError('Currently only noise to vehicle is supported!')

        control_input = control_input.reshape(
            list(control_input.shape[:-1])
            + [self.n_obj + 1] + [self.n_dim_per_obj])

        dist = torch.sqrt((control_input ** 2).sum(-1, keepdim=True))
        noise = npt.attach_dim(
            dist * self.sigma_per_dist_control + self.sigma_min_control,
            0, 1)

        # All dims except last two [obj, xy] are for the batch
        ndim_batch = control_input.ndimension() - 2
        ndim = self.n_dim_per_obj

        eye1 = npt.attach_dim(torch.eye(self.n_dim_per_obj), ndim_batch, 0)
        eye1 = eye1 * noise

        control_noise = npt.block_diag(
            [eye1.index_select(-3, ii).squeeze(-3)
             for ii in torch.arange(eye1.shape[-3])]
        )
        return control_noise

    ############################################################################
    def ____PLOTTING____(self):
        pass

    def plot(self, data=None, truths_landmark=None, truths_vehicle=None,
             scale_obs_noise=10.,
             to_plot_crosshair=False,
             to_plot_centroid=False):
        #%%
        # if data is None:
        ix_batch = 0
        data = self.data[ix_batch,:]
        landmark = self.truths_landmark[ix_batch,:]
        vehicle = self.truths_vehicle[ix_batch,:]

        loc_landmark = landmark.index_select(-1, self.dims_loc)
        loc_vehicle = vehicle.index_select(-1, self.dims_loc)
        ori_vehicle = vehicle.index_select(-1, self.dims_ori)

        loc_obs = data.index_select(-1, self.dims_loc)
        ori_obs = data.index_select(-1, self.dims_ori)

        #%%
        # ax = plt2.subplotRCs(1, self.n_view)
        n_col = 2
        ax = np.empty((self.n_view, n_col), dtype=object)

        for view in range(self.n_view):
            # Ground truth
            ax[view,0] = plt2.subplotRC(self.n_view, n_col, view + 1, 1)
            if to_plot_crosshair:
                self.plot_crosshair()

            for obj in range(self.n_obj):
                plt.plot(*npys(loc_landmark[0,obj,0], loc_landmark[0,obj,1]),
                         'o', color=self.get_color_obj(obj))
            plt.plot(*npys(
                loc_vehicle[view,0,0], loc_vehicle[view,0,1]), 'ko')

            if self.n_dim_ori > 0:
                uv_vehicle = pang2xy(ori_vehicle[view,0])
                quiver_scale=1

                plt.quiver(*npys(
                    loc_vehicle[view,0,0], loc_vehicle[view,0,1],
                    uv_vehicle[:,0], uv_vehicle[:,1]
                ), scale=quiver_scale, scale_units='xy')

            beautify_plot()
            if view == 0:
                plt.title('Ground truth')
                plt2.hide_ticklabels('x')
            elif view == 1:
                plt.xlabel('$x$')
                plt.ylabel('$y$')

            # Observations
            ax[view,1] = plt2.subplotRC(self.n_view, n_col, view + 1, 2)
            if to_plot_crosshair:
                self.plot_crosshair()

            rot_mat = rotation_matrix(torch.tensor([[0.]])) # 0.25]]))
            for obj in range(self.n_obj):
                loc_obs1 = rot_mat @ loc_obs[view,obj,:]
                sigma_obs1 = rot_mat @ torch.eye(self.n_dim_loc) \
                             * self.distsq2sigmasq(torch.sum(loc_obs1 ** 2))

                color = self.get_color_obj(obj)
                if to_plot_centroid:
                    plt2.plot_centroid(*npys(loc_obs1, sigma_obs1), color=color)

                plt.plot(*npys(loc_obs[view,obj,0], loc_obs[view,obj,1]), 'o',
                         color=color)

            plt.plot(0, 0, 'ko')
            if self.n_dim_ori > 0:
                plt.quiver(0, 0, 1, 0, scale=quiver_scale, scale_units='xy')

            beautify_plot()
            if view == 0:
                plt.title('Observation')
                plt2.hide_ticklabels('x')
            elif view == 1:
                plt.xlabel('$x_\\textrm{obs}$')
                plt.ylabel('$y_\\textrm{obs}$')
        return ax

    def plot_crosshair(self):
        grey = np.zeros(3) + 0.7
        plt.axhline(0, color=grey, linestyle=':')
        plt.axvline(0, color=grey, linestyle=':')

    def get_color_obj(self, obj=None):
        if obj is None:
            return [self.get_color_obj(obj) for obj in range(self.n_obj + 1)]

        if obj == self.n_obj:
            color = 'k'
        else:
            cmap = mpl.cm.get_cmap('Set1')
            color = cmap(obj)
        # if obj == 0:
        #     color = 'r'
        # elif obj == self.n_obj:
        #     color = 'k'
        # else:
        #     color = ['b', 'g', 'c', 'm'][obj - 1]
        return color

    def get_label_obj(self, obj=None):
        if obj is None:
            return [self.get_label_obj(obj) for obj in range(self.n_obj + 1)]

        if obj == 0:
            label = 'R'
        elif obj == self.n_obj:
            label = 'S'
        else:
            label = 'L%d' % obj
        return label

    ############################################################################
    # Functions required as an iterator
    def ____ITERATION____(self):
        pass

    def __getitem__(self, index):
        return self.data[index,:], self.target[index,:]

    def __len__(self):
        return len(self.data)


def get_unit_vec(v, dim=-1):
    return v / torch.sqrt(torch.sum(v ** 2, dim=dim))