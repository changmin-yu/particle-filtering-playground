import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import numpy_groupies as npg
import torch
from torch import nn
from torch import distributions as distrib
from collections import OrderedDict as odict
from importlib import reload

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from bion_rectangle.utils import numpytorch as npt
from bion_rectangle.utils import yktorch as ykt

torch.set_default_tensor_type(torch.DoubleTensor) # To prevent underflow

reload(npt)
enforce_tensor = npt.enforce_tensor
v2m = npt.v2m
m2v = npt.m2v
p2st = npt.p2st
p2en = npt.p2en
matsum = npt.matsum

#%% Kalman filter
class KalmanFilter(ykt.BoundedModule):
    def __init__(self,
                 ndim_hidden, ndim_obs, ndim_control,
                 tran_gain=None, tran_control_gain=None, tran_noise=None,
                 meas_gain=None, meas_control_gain=None, meas_noise=None,
                 mu_prior=None, sigma_prior=None,
                 ):
        """
        Linear Gaussian State Space Model. See Murphy 2012, Ch. 18
        :param tran_gain: A. batch_size + [ndim_hidden, ndim_hidden]
        :param tran_control_gain: B. batch_size + [ndim_hidden, ndim_control]
        :param tran_noise: Q. batch_size + [ndim_hidden, ndim_hidden]
        :param meas_gain: C. batch_size + [ndim_obs, ndim_hidden]
        :param meas_control_gain: D. batch_size + [ndim_obs, ndim_control]
        :param meas_noise: R. batch_size + [ndim_obs, ndim_obs]
        :param mu_prior: prior location of the hidden state z.
            batch_size + [ndim_hidden]
        :param sigma_prior: prior covariance of the hidden state z.
            batch_size + [ndim_hidden, ndim_hidden]

        Transition model:
        z = A @ z + B @ u + e
            where e ~ N(0, Q)

        Observation model:
        y = C @ z + D @ u + d
            where d ~ N(0, R)
        """
        super().__init__()

        self.ndim_hidden = ndim_hidden
        self.ndim_obs = ndim_obs
        self.ndim_control = ndim_control

        if tran_gain is None:
            tran_gain = torch.eye(ndim_hidden)
        self.tran_gain = tran_gain

        if tran_control_gain is None:
            tran_control_gain = torch.eye(ndim_hidden, ndim_control)
        self.tran_control_gain = tran_control_gain

        if tran_noise is None:
            tran_noise = torch.eye(ndim_hidden)
        self.tran_noise = tran_noise

        if meas_gain is None:
            meas_gain = torch.eye(ndim_obs, ndim_hidden)
        self.meas_gain = meas_gain

        if meas_control_gain is None:
            meas_control_gain = torch.eye(ndim_obs, ndim_control)
        self.meas_control_gain = meas_control_gain

        if meas_noise is None:
            meas_noise = torch.eye(ndim_obs)
        self.meas_noise = meas_noise

        if mu_prior is None:
            mu_prior = torch.zeros(ndim_hidden)
        self.mu_prior = mu_prior

        if sigma_prior is None:
            sigma_prior = torch.eye(ndim_hidden)
        self.sigma_prior = sigma_prior

        self.mu = self.mu_prior.clone()
        self.sigma = self.sigma_prior.clone()

    def update_params(self, **kwargs):
        for key in kwargs.keys():
            if key in ['tran_gain', 'tran_control_gain', 'tran_noise',
                       'meas_gain', 'meas_control_gain', 'meas_noise']:
                self.__dict__[key] = kwargs[key]
            else:
                raise ValueError('Updating %s this way is not allowed!'
                                 % key)

    def update(self, obs,
               control=None,
               skip_prediction=False,
               skip_measurement=False,
               skip_obs_dims=None,
               update_state=True,
               **kwargs):
        """
        Perform prediction and measurement steps of the Kalman filter.
        :param obs: y. batch_size + [ndim_obs]
        :param control: u. batch_size + [ndim_control]
        :param skip_prediction
        :param skip_measurement
        :param skip_obs_dims: indices of observation dimension to skip
        :param update_state: if True (default), update self.mu and sigma
        :param kwargs: update params using update_params
        :return: mu, sigma.
            mu: batch_size + [ndim_hidden]
            sigma: batch_size + [ndim_hidden, ndim_hidden]
        """

        self.update_params(**kwargs)

        if skip_prediction:
            mu_pred = self.mu
            sigma_pred = self.sigma
        else:
            mu_pred, sigma_pred = self.prediction_step(control=control)

        if skip_measurement:
            mu = mu_pred
            sigma = sigma_pred
        else:
            mu, sigma = self.measurement_step(
                obs=obs,
                mu_pred=mu_pred,
                sigma_pred=sigma_pred,
                control=control,
                skip_obs_dims=skip_obs_dims
            )

        if update_state:
            self.mu = mu
            self.sigma = sigma

        return mu, sigma

    def prediction_step(self, mu=None, sigma=None, control=None):
        """
        Prediction step of the Kalamn filter.
        :param control: u_t. batch_size + [ndim_control]
        :return: mu_pred, sigma_pred
            mu_pred: batch_size + [ndim_hidden]
            sigma_pred: batch_size + [ndim_hidden, ndim_hidden]
        """
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma

        mu, tran_gain, sigma, tran_noise = npt.repeat_batch(
            npt.attach_dim(mu, 0, 1),
            self.tran_gain,
            sigma,
            self.tran_noise
        )
        mu_pred = tran_gain @ mu

        if control is not None:
            mu_pred, tran_control_gain, control_all = npt.expand_upto_dim((
                mu_pred,
                self.tran_control_gain,
                npt.attach_dim(control, 0, 1)
            ), -2)
            mu_pred = mu_pred + tran_control_gain @ control_all

        sigma_pred = tran_gain @ sigma @ npt.t(tran_gain) \
                     + tran_noise
        mu_pred, sigma_pred = npt.expand_upto_dim((
            mu_pred, sigma_pred
        ), -2)
        mu_pred = mu_pred.flatten(-2)

        return mu_pred, sigma_pred

    def measurement_step(self, obs, mu_pred, sigma_pred, control=None,
                         skip_obs_dims=None):
        """
        Measurement step of the Kalman filter.
        :param obs: y_t. batch_size + [ndim_obs]
        :param mu_pred: batch_size + [ndim_hidden]
        :param sigma_pred: batch_size + [ndim_hidden, ndim_hidden]
        :param control: u_t. batch_size + [ndim_control]
        :return: mu, sigma
            mu: batch_size + [ndim_hidden]
            sigma: batch_size + [ndim_hidden, ndim_hidden]
        """

        meas_gain = self.meas_gain.clone()
        if skip_obs_dims is not None:
            meas_gain = npt.permute2st(meas_gain, 2)
            meas_gain[skip_obs_dims,:] = 0.
            meas_gain = npt.permute2en(meas_gain, 2)

        S0 = meas_gain @ sigma_pred @ meas_gain.t()

        S0_all, meas_noise_all = npt.expand_upto_dim((S0, self.meas_noise), -2)
        S = S0_all + meas_noise_all

        sigma_pred1, meas_gain1, S1 = npt.repeat_batch(
            sigma_pred, meas_gain, S)
        K = sigma_pred1 @ npt.t(meas_gain1) @ S1.inverse()

        obs_pred = (meas_gain @ npt.attach_dim(mu_pred, 0, 1)).flatten(
            mu_pred.ndimension() - 1
        )
        if control is not None:
            meas_control_gain, control = npt.repeat_batch(
                self.meas_control_gain,
                npt.attach_dim(control, 0, 1)
            )
            obs_pred += (meas_control_gain @ control).flatten(-2)
        obs1, obs_pred1 = npt.repeat_batch(obs, obs_pred)

        r = obs - obs_pred

        Kr = (K @ npt.attach_dim(r, 0, 1)).flatten(r.ndimension()-1)
        mu, Kr = npt.repeat_batch(self.mu, Kr)
        mu = mu + Kr

        eye1, K1, meas_gain1, sigma_pred1 = npt.repeat_batch(
            torch.eye(self.ndim_hidden), K, meas_gain, sigma_pred)
        sigma = (eye1 - K1 @ meas_gain1) @ sigma_pred1

        return mu, sigma

    def posterior_predictive(self, mu=None, sigma=None, control=None):
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma

        mu_pred, sigma_pred = self.prediction_step(mu=mu, sigma=sigma,
                                                   control=control)

        mu_pred, meas_gain, sigma_pred, meas_noise = npt.repeat_batch(
            npt.attach_dim(mu_pred, 0, 1),
            self.meas_gain,
            sigma_pred,
            self.meas_noise
        )
        mu_obs = meas_gain @ mu_pred
        mu_obs = mu_obs.flatten(-2)
        sigma_obs = meas_gain @ sigma_pred @ npt.t(meas_gain) \
                    + meas_noise

        return mu_obs, sigma_obs

class ExtendedKalmanFilter(KalmanFilter):
    """
    Get a function (and its Jacobian) instead of meas_gain.
    """
    def __init__(self,
                 ndim_hidden, ndim_obs, ndim_control,
                 tran_fun=None, tran_jacobian=None,
                 meas_fun=None, meas_jacobian=None,
                 **kwargs
                 ):
        super().__init__(ndim_hidden, ndim_obs, ndim_control, **kwargs)

        # if tran_fun is None, fall bak to the linear behavior using tran_gain
        self._tran_fun = tran_fun
        self._tran_jacobian = tran_jacobian

        # if meas_fun is None, fall bak to the linear behavior using meas_gain
        self._meas_fun = meas_fun
        self._meas_jacobian = meas_jacobian

    def tran_fun(self, vec_hid, control):
        if self._tran_fun is not None:
            return self._tran_fun(vec_hid, control)
        else:
            return m2v(
                self.tran_gain @ v2m(vec_hid) \
                + self.tran_control_gain @ v2m(control)
            )

    @property
    def tran_jacobian(self):
        if self._tran_jacobian is not None:
            return self._tran_jacobian
        else:
            return self.tran_gain

    @tran_jacobian.setter
    def tran_jacobian(self, value):
        self._tran_jacobian = value

    def meas_fun(self, vec_obs, control=None):
        if self._meas_fun is not None:
            return self._meas_fun(vec_obs, control)
        else:
            return m2v(
                self.meas_gain @ v2m(vec_obs)
                + self.meas_control_gain @ v2m(control)
            )

    @property
    def meas_jacobian(self):
        if self._meas_jacobian is not None:
            return self._meas_jacobian
        else:
            return self.meas_gain

    @meas_jacobian.setter
    def meas_jacobian(self, value):
        self._meas_jacobian = value

    def update_params(self, **kwargs):
        for key in kwargs.keys():
            if key in ['tran_fun', '_tran_jacobian', 'tran_noise',
                       'meas_fun', '_meas_jacobian', 'meas_noise']:
                self.__dict__[key] = kwargs[key]
            else:
                raise ValueError('Updating %s this way is not allowed!'
                                 % key)

    def prediction_step(self, mu=None, sigma=None, control=None):
        """
        Prediction step of the Extended Kalamn filter.
        :param control: u_t. batch_size + [ndim_control]
        :return: mu_pred, sigma_pred
            mu_pred: batch_size + [ndim_hidden]
            sigma_pred: batch_size + [ndim_hidden, ndim_hidden]
        """
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        if control is None:
            control = torch.zeros(self.ndim_control)

        mu_pred = self.tran_fun(mu, control)
        tran_jac = self.tran_jacobian
        sigma_pred = matsum(tran_jac @ sigma @ npt.t(tran_jac), self.tran_noise)

        n_hid = sigma_pred.shape[-1]
        sigma_pred[:, torch.arange(n_hid), torch.arange(n_hid)] = \
            torch.clamp(
                sigma_pred[:, torch.arange(n_hid), torch.arange(n_hid)],
                min=1e-12)

        if torch.any(torch.diag(sigma_pred[0,:,:]) < 0):
            print('Negative variance!')
            print('--')

        return mu_pred, sigma_pred

    def measurement_step(self, obs, mu_pred, sigma_pred, control=None,
                         skip_obs_dims=None):
        if control is None:
            control = torch.zeros(self.ndim_control)

        meas_jac = self.meas_jacobian.clone()
        if skip_obs_dims is not None:
            meas_jac = npt.permute2st(meas_jac, 2)
            meas_jac[skip_obs_dims,:] = 0.
            meas_jac = npt.permute2en(meas_jac, 2)

        S = matsum(meas_jac @ sigma_pred @ npt.t(meas_jac), self.meas_noise)
        K = sigma_pred @ npt.t(meas_jac) @ S.inverse()

        obs_pred = self.meas_fun(mu_pred, control)
        Kr = K @ matsum(v2m(obs), -v2m(obs_pred))
        mu = m2v(matsum(v2m(mu_pred), Kr))

        sigma = matsum(torch.eye(self.ndim_hidden), -K @ meas_jac) @ sigma_pred
        n_hid = sigma.shape[-1]
        sigma[:, torch.arange(n_hid), torch.arange(n_hid)] = \
            torch.clamp(sigma[:, torch.arange(n_hid), torch.arange(n_hid)],
                        min=1e-12)

        if torch.any(torch.diag(sigma[0,:,:]) < 0):
            print('Negative variance!')
            print('--')

        return mu, sigma

    def posterior_predictive(self, mu=None, sigma=None, control=None):
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma

        mu_pred, sigma_pred = self.prediction_step(mu=mu, sigma=sigma,
                                                   control=control)

        mu_obs = self.meas_fun(mu_pred, control)

        meas_jac = self.meas_jacobian
        sigma_obs = matsum(meas_jac @ sigma_pred @ npt.t(meas_jac),
                           self.meas_noise)

        return mu_obs, sigma_obs