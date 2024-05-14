"""
Use a grid representation of the self location & orientation,
and pixel (bitmap) representation of the retina
to model boundaries
"""

import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from typing import Iterable, Union, Sequence, Dict, Mapping, List, Type
from scipy import ndimage

import torch
from torch import distributions as distrib
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from bion_rectangle.behav.geometry import get_unit_vec
from bion_rectangle.utils import np2
from bion_rectangle.utils import plt2
from bion_rectangle.utils.np2 import AliasStr as Short
from bion_rectangle.utils import numpytorch as npt
from bion_rectangle.utils import yktorch as ykt
from bion_rectangle.utils import argsutil
from bion_rectangle.utils.numpytorch import (
    npy, 
    npys, 
    p2st, 
)
from bion_rectangle.utils.localfile import LocalFile, mkdir4file, Cache
import bion_rectangle.behav.projection_3D as proj3d
from bion_rectangle.behav import env_boundary
from bion_rectangle.behav.geometry import rotation_matrix
from bion_rectangle.utils import mplstyle

device0 = torch.device(
    # 'cuda:0' if torch.cuda.is_available() else
    'cpu'
)
npt.set_device(device0)

geom = proj3d.ProjectiveGeometry3D()
eps = 1e-6

locfile = LocalFile('../Data_SLAM/slam_pixel_grid', kind2subdir=True)


def ____UTILS____():
    pass


def prad2unitvec(prad, dim=-1):
    """
    :param prad:
    :param dim: dimension to stack x and y. Either 0 or -1.
    :rtype: torch.FloatTensor    """
    rad = prad * 2. * np.pi
    return torch.stack([torch.cos(rad), torch.sin(rad)], dim=dim)


def pconc2conc(pconc):
    pconc = torch.clamp(pconc, min=1e-6, max=1-1e-6)
    return 1. / (1. - pconc) - 1.


def vmpdf_ix(self, x_ix, mu_ix, pconc, nx):
    return npt.vmpdf(prad2unitvec(x_ix / nx),
                     prad2unitvec(mu_ix / nx),
                     pconc2conc(pconc))


def vmpdf_a_given_b(self, pconc):
    dist = ((npt.arange(self.nz)[:, None]
             - npt.arange(self.nz)[None, :]) % self.nz).double()
    return npt.sumto1(vmpdf_ix(
        dist.flatten(), npt.tensor([0.]),
        pconc
    ).reshape([self.nz] * 2), 0)


def enforce3d(v0: Union[float, Sequence[float], np.ndarray, torch.Tensor]) -> \
        torch.Tensor:
    """
    enforce a scalar, 1D, or 2D velocity into 3D by padding last dim.
    :param v0: [batch, (forward, rightward, upward)]: rightward or upward may be missing.
    :return: v[batch, (forward, rightward, upward)
    """
    v0 = npt.tensor(v0)
    if v0.ndim == 0:
        v0 = v0[None]

    if v0.shape[-1] < 3:
        v = F.pad(v0, [0, 3 - v0.shape[-1]])
    else:
        assert v0.shape[-1] == 3
        v = v0
    return v


def ___EnvTable____():
    pass


class StateTable:
    """
    Summary:
    -----------
    A table of states (x, y, heading) for a given environment.

    Description:
    ------------
    This class is used to represent the state space of an agent in a given environment in a discrete manner.

    Attributes:
    ------------
    - env
    - x_max
    - y_max
    - dx
    - dy
    - ddeg
    - to_use_inner
    - xs0
    - ys0
    - to_use_env_xy
    - nx
    - n_heading
    - xs
    - ys
    - x0
    - y0
    - heading_deg0
    - n_state
    - headings_deg
    - x
    - y
    - heading_deg
    - state_incl
    - n_state_incl
    - x_loc
    - y_loc
    - state_loc_incl
    - n_state_loc_incl
    - x_loc_incl
    - y_loc_incl
    - state_incl_inner

    Methods:
    -----------
    - get_env_from_kw
    - p_loc_incl2p_xy_incl
    - p_xy_incl2p_loc_incl
    - get_state_table_w_dx
    - resample
    - xyd_incl
    - xy_loc_incl
    - xy_loc_incl_flat
    - xy_loc_all
    - i_state2i_loc
    - i_state2ixy
    - i_loc2ixy
    - xy2ixy
    - xy2i_loc
    - set_state_table
    - get_state_table_as_dict
    - get_kw_init
    - plot_p_loc
    - plot_p_heading
    - p_state_incl2p_state_loc_incl
    - p_state_loc_incl2p_state_incl
    - i_states2states
    - p_tran_loc_heading2loc
    - p_locOri_locOri2p_loc_loc
    """
    list_table_init_attr = [
        'env',
        'x_max',
        'y_max',
        'dx',
        'dy',
        'ddeg',
        'to_use_inner',
        'xs0',
        'ys0',
        'to_use_env_xy'
    ]

    list_table_attr = [
        'nx',
        'n_heading',
        'xs',
        'ys',
        'x0',
        'y0',
        'heading_deg0',
        'n_state',
        'headings_deg',
        'x',
        'y',
        'heading_deg',
        'state_incl',
        'n_state_incl',
        'x_loc',
        'y_loc',
        'state_loc_incl',
        'n_state_loc_incl',
        'x_loc_incl',
        'y_loc_incl',
        'state_incl_inner'
    ]

    def __init__(
        self,
        env: Union[env_boundary.EnvBoundary,
                   'StateTable', 'EnvTable'],
        x_max=None,
        y_max=None,
        dx=None,
        dy=None,
        ddeg=None,
        to_use_inner=False,
        xs0=None,
        ys0=None,
        to_use_env_xy=True,
        **kwargs
    ):
        """

        :param env:
        :param x_max:
        :param dx:
        :param dy:
        :param ddeg:
        :param to_use_inner:
        :param kwargs:
        """
        super().__init__()

        self.env = env

        try:
            self.set_state_table(env)
        except AttributeError:
            pass

        self.to_use_inner = to_use_inner

        # Define table (x, y, heading)
        if x_max is None:
            x_max = env.x_max
        self.x_max = x_max

        if y_max is None:
            y_max = env.y_max
        self.y_max = y_max

        if dx is None:
            dx = env.dx
        self.dx = dx
        self.to_use_env_xy = to_use_env_xy

        if dy is None:
            dy = env.dy
        self.dy = dy

        self.xs0 = xs0
        self.ys0 = ys0
        if xs0 is None:
            if to_use_env_xy and hasattr(env, 'xs'):
                xs0 = env.xs
            else:
                # start from the center
                xs0 = npt.arange(
                    0, self.x_max + self.dx - 1e-12,
                    self.dx)
                xs0 = torch.cat([
                    -xs0[1:].flip(0), xs0
                ], 0)
        self.xs = xs0
        self.nx = xs0.shape[0]

        if ys0 is None:
            if to_use_env_xy and hasattr(env, 'ys'):
                ys0 = env.ys  # noqa
            else:
                ys0 = xs0
        self.ys = ys0
        self.ny = ys0.shape[0]

        if ddeg is None:
            try:
                ddeg = env.ddeg  # noqa
            except AttributeError:
                ddeg = 90.  # PAPERPARAM
        headings_deg = npt.arange(0., 360., ddeg)
        self.ddeg = ddeg
        self.n_heading = headings_deg.shape[0]

        # All grid positions
        x, y, heading_deg = torch.meshgrid([xs0, ys0, headings_deg])
        incl = self.is_inside(x, y)
        self.x0, self.y0, self.heading_deg0 = (
            v.clone() for v in [x, y, heading_deg])
        self.n_state = self.x0.numel()

        # Define eligible grid positions (within the environment)
        x, y, heading_deg = (v[incl] for v in [x, y, heading_deg])
        self.headings_deg, self.x, self.y, self.heading_deg, self.state_incl = \
                headings_deg, x, y, heading_deg, incl
        self.n_state_incl = self.x.numel()

        self.xs_incl, self.ix_incl = torch.unique(self.x, return_inverse=True)
        self.ys_incl, self.iy_incl = torch.unique(self.y, return_inverse=True)
        self.headings_incl, self.iheading_incl = torch.unique(
            self.heading_deg, return_inverse=True)

        self.nx_incl = len(self.xs_incl)
        self.ny_incl = len(self.ys_incl)
        self.nheading_incl = len(self.headings_incl)

        if self.nx_incl > 0:
            self.x_incl_min = self.xs_incl[0]
            self.x_incl_max = self.xs_incl[-1]
        else:
            self.x_incl_min = np.nan
            self.x_incl_max = np.nan

        if self.ny_incl > 0:
            self.y_incl_min = self.ys_incl[0]
            self.y_incl_max = self.ys_incl[-1]
        else:
            self.y_incl_min = np.nan
            self.y_incl_max = np.nan

        if self.nheading_incl > 0:
            self.headings_incl_min = self.headings_incl[0]
            self.headings_incl_max = self.headings_incl[-1]
        else:
            self.headings_incl_min = np.nan
            self.headings_incl_max = np.nan

        # Location-only states (marginalizing over headings)
        self.x_loc, self.y_loc = torch.meshgrid([xs0, ys0])
        self.state_loc_incl = self.is_inside(self.x_loc, self.y_loc)
        self.n_state_loc_incl = np.sum(npy(self.state_loc_incl))
        self.x_loc_incl = self.x_loc.flatten()[self.state_loc_incl.flatten()]
        self.y_loc_incl = self.y_loc.flatten()[self.state_loc_incl.flatten()]

        self.ix_loc_incl = torch.unique(self.x_loc_incl, return_inverse=True)[1]
        self.iy_loc_incl = torch.unique(self.y_loc_incl, return_inverse=True)[1]

        # ixs_incl[ix] = True or False
        self.ixs_incl = np.isin(*npys(self.xs, self.xs_incl))
        self.iys_incl = np.isin(*npys(self.ys, self.ys_incl))

        # to_check_points = True  # CHECKED
        # if to_check_points:
        #     plt.plot(*np2.npys(self.x_loc.flatten()), self.y_loc.flatten(), 'k.')
        #     self.env.plot_walls(mode='line')
        #     plt.axis('equal')
        #     plt.show()

        if self.n_state_loc_incl == self.nx_incl * self.ny_incl:
            self.iloc_incl_by_ix_iy = -np.ones([self.nx_incl, self.ny_incl], dtype=int)
            for ixy, xy in enumerate(self.xy_loc_incl_flat.T):
                ix, iy = self.xy2ixy(npy(xy))
                self.iloc_incl_by_ix_iy[ix, iy] = ixy
        else:
            self.iloc_incl_by_ix_iy = None

        self.extent = npys(
            self.x_incl_min - self.dx / 2,
            self.x_incl_max + self.dx / 2,
            self.y_incl_min - self.dx / 2,
            self.y_incl_max + self.dx / 2
        )
        self.xlim = self.extent[:2]
        self.ylim = self.extent[2:]

        # Inner locations
        self.state_incl_inner = self.is_inside(
            self.x_loc, self.y_loc, use_inner=True)

    @staticmethod
    def get_env_from_kw(
        type_env: Type[env_boundary.EnvBoundary],
        kw_init_env: dict = (),
        kw_init_state_table: dict = ()
    ) -> 'StateTable':
        """

        :param type_env:
        :param kw_init_env: from env.get_kw_init()
        :param kw_init_state_table: from state_table.get_kw_init()
        :return: state_table
        """
        return StateTable(
            **{**kw_init_state_table, **{'env': type_env(**kw_init_env)}}
        )


    def p_loc_incl2p_xy_incl(
        self, p: np.ndarray, fill_value=np.nan
    ) -> np.ndarray:
        """

        :param p: [..., loc_incl]
        :param fill_value:
        :return: p[..., xs_incl, ys_incl]
        """
        shape_batch = p.shape[:-1]
        p_s_xy = np.zeros(shape_batch + (self.nx * self.ny,)) + fill_value
        p_s_xy[..., self.state_loc_incl.flatten()] = p.reshape(
            shape_batch + (-1,))
        p_s = p_s_xy.reshape(shape_batch + (self.nx, self.ny))
        p_s = p_s[..., self.ixs_incl, :][..., :, self.iys_incl]
        return p_s

    def p_xy_incl2p_loc_incl(
        self, p: np.ndarray
    ) -> np.ndarray:
        """

        :param p: [..., xs_incl, ys_incl]
        :param fill_value:
        :return: p[..., loc_incl]
        """
        shape_batch = p.shape[:-2]
        p = p.reshape(shape_batch + (-1,))
        p_loc_incl = p[..., self.state_loc_incl[
            self.ixs_incl, :][:, self.iys_incl].flatten()]
        return p_loc_incl

    def get_state_table_w_dx(self, dx_out: float) -> 'StateTable':
        to_use_env_xy0 = self.to_use_env_xy
        v = StateTable(
            **{
                **self.get_kw_init(),
                'dx': dx_out,
                'to_use_env_xy': False
            }
        )
        v.to_use_env_xy = to_use_env_xy0
        return v

    def resample(
        self, v: Union[np.ndarray, Sequence[np.ndarray]],
        dx_out: float, method='linear'
    ) -> np.ndarray:
        """

        :param self:
        :param v: [dim, i_loc_incl_in]
        :param dx_out:
        :param method:
        :return: v[dim, i_loc_incl_out]
        """
        env_out = self.get_state_table_w_dx(dx_out)
        return np.stack([
            np2.griddata_fillnearest(
                npys(self.x_loc_incl, self.y_loc_incl),
                v1.flatten(),
                npys(env_out.x_loc_incl, env_out.y_loc_incl),
                method=method,
            ) for v1 in v
        ])

    @property
    def xyd_incl(self) -> torch.Tensor:
        """
        :return: xyd_incl[state_incl, xyd]
        """
        return torch.stack([
            self.x, self.y, self.heading_deg
        ], -1)

    @property
    def xy_loc_incl(self) -> torch.Tensor:
        """
        :return: xy_loc_incl[xy, xs_incl, ys_incl]
        """
        return torch.stack([
            self.xs_incl[:, None].expand([self.nx_incl, self.ny_incl]),
            self.ys_incl[None, :].expand([self.nx_incl, self.ny_incl])
        ])

    @property
    def xy_loc_incl_flat(self) -> torch.Tensor:
        """

        :return: xy_loc_incl_flat[xy, i_loc_incl]
        """
        return torch.stack([self.x_loc_incl, self.y_loc_incl])

    @property
    def xy_loc_all(self) -> torch.Tensor:
        """
        :return: xy_loc_all[xy, xs, ys]
        """
        return torch.stack([
            self.xs[:, None].expand([self.nx, self.ny]),
            self.ys[None, :].expand([self.nx, self.ny])
        ])

    def i_state2i_loc(self, i_state):
        # Currently, this is valid because all headings are allowed at each
        # location, and heading comes last in meshgrid.
        return i_state // self.n_heading

    def i_state2ixy(self, i_state):
        # Currently, this is valid because all headings are allowed at each
        # location, and heading comes last in meshgrid.
        # return i_state // self.n_heading
        i_loc = self.i_state2i_loc(i_state)
        return self.ix_loc_incl[i_loc], self.iy_loc_incl[i_loc]

    def i_loc2ixy(self, i_loc):
        return self.ix_loc_incl[i_loc], self.iy_loc_incl[i_loc]

    def xy2ixy(self, xy: np.ndarray):
        """
        :param xy: [i, (x, y)]
        :return: ix[i], iy[i]
        """
        ix = np.round(
            (xy[..., 0] - npy(self.x_incl_min)) / self.dx).astype(int)
        iy = np.round(
            (xy[..., 1] - npy(self.y_incl_min)) / self.dx).astype(int)
        return ix, iy

    def xy2i_loc(self, xy: np.ndarray, find_nearest=False) -> np.ndarray:
        """
        Only valid for rectangles!
        Returns -1 for out-of-range xy's.

        :param xy: [i, (x, y)]
        :return: i_loc[i]
        """
        # if find_nearest:
        shape0 = xy.shape[:-1]
        xy = npt.tensor(xy).reshape([-1, 2])
        i_loc = np.array([
            npy(self.i_state2i_loc(
                self.get_i_state(
                    xy1, self.headings_deg[0],
                    find_nearest=find_nearest
                )
            )) for xy1 in xy
        ]).reshape(shape0)
        return i_loc

        # else:
        #     ix, iy = self.xy2ixy(xy)
        #     return self.iloc_incl_by_ix_iy[ix, iy]
        # if self.nx_incl * self.ny_incl == self.n_state_loc_incl:
        #     ix, iy = self.xy2ixy(xy)
        #
        #     i_loc = ix * self.ny_incl + iy
        #     i_loc[(ix < 0) | (ix >= self.nx_incl)
        #           | (iy < 0) | (iy > self.ny_incl)] = -1
        # else:
        #
        #
        # return i_loc

    @property
    def state_template(self) -> 'AgentState':
        raise NotImplementedError()

    @property
    def xy(self) -> torch.Tensor:
        return torch.stack([self.x, self.y], -1)

    def is_inside(
        self, x, y=None, use_inner=None, x_incl='all'
    ) -> torch.Tensor:
        """

        :param x: [point] or [point, (x, y)] (omit y for the latter)
        :param y: [point] if given
        :param x_incl: 'all'(default)|'left'|'right'
        :param use_inner: use polygon_inner
            (e.g., when subjects cannot approach
            the walls beyond a certain margin)
        :return: inside[point]
        """
        if y is not None:
            xy = torch.stack([x, y], -1)
        else:
            xy = x
        return npt.tensor(
            self.env.is_inside(
                npy(xy), use_inner=use_inner, x_incl=x_incl,
            ))

    def get_loc_heading_by_i_state(self, i_state):
        return (self.x[i_state], self.y[i_state]),\
               npt.tensor([self.heading_deg[i_state]])

    def get_state(self, i_state=None,
                  xy=(0., 0.),
                  heading_deg=0.,
                  state_template=None):
        if i_state is not None:
            xy, heading_deg = self.get_loc_heading_by_i_state(i_state)
        if state_template is None:
            state = deepcopy(self.state_template)
        else:
            state = state_template

        xy = npt.tensor(xy)
        state.loc[0] = xy[0]
        state.loc[1] = xy[1]

        heading_deg = npt.tensor(heading_deg, min_ndim=0)
        heading_rad = npt.deg2rad(heading_deg)
        state.heading[0] = torch.cos(heading_rad)
        state.heading[1] = torch.sin(heading_rad)
        return state

    def set_state(self, i_state: int = None):
        self.state_template.loc_xy = npt.tensor([
            self.x[i_state], self.y[i_state]])
        self.state_template.heading_deg = npt.tensor(self.heading_deg[i_state])

    def get_i_state(
        self,
        loc: Union[torch.Tensor, Sequence[float]] = None,
        heading_deg: Union[torch.Tensor, Sequence[float], float] = None,
        state: Union[torch.Tensor, Sequence[float]] = None,
        find_nearest=True,
        sample_p_aliased=False,
    ) -> Union[torch.Tensor, int]:
        """

        :param state: [..., (x, y, heading_deg)]
        :param loc: [..., (x, y)]
        :param heading_deg: [...] = deg
        :param find_nearest:
        :return: i_state[...]
        """
        if state is not None:
            heading_deg = state[..., 2]
            loc = state[..., :2]
        else:
            assert loc is not None
            assert heading_deg is not None

        if sample_p_aliased:
            p_aliased = self.get_p_state_aliased(loc, heading_deg)
            return torch.multinomial(p_aliased, 1).item()
        elif find_nearest:
            return torch.argmin(
                (self.x - loc[0]) ** 2 +
                (self.y - loc[1]) ** 2 +
                npt.circdiff(self.heading_deg, heading_deg, 360.).abs()
            )
        else:
            return torch.nonzero(
                (torch.abs(self.x - loc[0]) < eps) &
                (torch.abs(self.y - loc[1]) < eps) &
                (torch.abs((self.heading_deg - heading_deg + 180.) %
                           360. - 180.) < eps)
            )[0, 0]

    def get_p_state_aliased(
        self,
        loc: Union[torch.Tensor, Sequence[float]],
        heading_deg: Union[torch.Tensor, Sequence[float], float] = None,
    ) -> torch.Tensor:
        """

        :param loc:
        :param heading_deg:
        :return: [..., n_state]
        """

        if heading_deg is None:
            p = npt.get_p_state_aliased(
                loc, self.xyd_incl[:, :2]
            ) # uniform distribution over heading
            return p
        else:
            p = npt.get_p_state_aliased(
                torch.cat([loc, heading_deg[..., None]], -1),
                self.xyd_incl
            )
            return p

    def get_i_loc(
            self,
            loc: torch.Tensor = None,
            find_nearest=True
    ) -> torch.Tensor:
        """

        :param loc: [x, y]
        :param find_nearest:
        :return: i_state
        """
        if loc is None:
            loc = self.state_template.loc_xy

        if find_nearest:
            return torch.argmin(
                (self.x_loc_incl - loc[0]) ** 2 +
                (self.y_loc_incl - loc[1]) ** 2
            )
        else:
            return torch.nonzero(
                (torch.abs(self.x_loc_incl - loc[0]) < eps) &
                (torch.abs(self.y_loc_incl - loc[1]) < eps)
            )[0, 0]

    def set_state_table(self, src: Union[dict, 'StateTable'],
                        to_deepcopy=True):
        if not isinstance(src, dict):
            src = src.get_state_table_as_dict()
        if to_deepcopy:
            src = deepcopy(src)
        self.__dict__.update(src)

    def get_state_table_as_dict(self) -> dict:
        return {k: self.__dict__[k] for k in self.list_table_attr}

    def get_kw_init(self, exclude_env=False) -> dict:
        return {
            k: self.__getattribute__(k) for k in self.list_table_init_attr
            if (not exclude_env) or (k not in ['env'])
        }

    def ____PLOT_LIKELIHOOD____(self):
        pass

    def plot_p_loc(
            self,
            prob: torch.Tensor = None,
            heading_deg=None,
            log2prob=False,
            to_normalize=True,
            cmap='jet',
            to_white_background=True,
            to_use_inner=False,
            plot_kind='imshow',
            kw_imshow=(),
    ):
        """
        :param prob: from loglik_retina(), p_state_incl, or p_state_loc_incl
        :type prob: Union[np.ndarray, torch.Tensor]
        :param heading_deg:
        :param cmap:
        :return: h: matplotlib object drawn
        """
        if prob is None:
            raise ValueError()
            # prob = npy(self.p_state_incl.flatten())
            # log2prob = False
        else:
            prob = npy(prob)

        if prob.size == self.n_state_incl:
            if heading_deg is None:
                # Plot likelihood of location marginalized over headings
                ll = np.zeros(self.n_state) + np.nan
                ll[npy(self.state_incl.flatten())] = prob.flatten()
                ll1 = ll.reshape([self.nx, self.ny, self.n_heading])
                if log2prob:
                    lik1 = np.exp(ll1 - np.nanmax(ll1))
                    lik1 = np.nansum(lik1, -1)
                    if to_normalize:
                        lik1 = np2.sumto1(lik1)
                else:
                    if to_normalize:
                        lik1 = np.nanmean(ll1, -1)
                    else:
                        lik1 = np.nansum(ll1, -1)
            else:
                ll = np.zeros(self.n_state) + np.nan
                ll[self.state_incl.flatten()] = prob
                heading_deg = self.headings_deg[np.argmin(np.abs(
                    (self.headings_deg - heading_deg) % 360.
                ))]  # noqa
                incl1 = self.heading_deg0 == heading_deg
                ll1 = ll[incl1.flatten()].reshape([self.nx, self.ny])
                if log2prob:
                    lik1 = np.exp(ll1 - np.nanmax(ll1))
                    if to_normalize:
                        lik1 = np2.sumto1(lik1)
                else:
                    lik1 = ll1
        else:
            assert prob.size == self.n_state_loc_incl, \
                'size should be either n_state_incl (joint of location ' \
                'and heading) or n_state_loc_incl (location only)'

            if log2prob:
                prob = np2.sumto1(np.exp(prob))

            lik1 = np.zeros(self.n_state // self.n_heading) + np.nan
            lik1[npy(self.state_loc_incl.flatten())] = prob.flatten()
            lik1 = np.reshape(lik1, [self.nx, self.ny])

        if to_white_background:
            if to_use_inner:
                lik1[~self.state_incl_inner] = np.nan
            else:
                lik1[~npy(self.state_incl[:, :, 0])] = np.nan
            plt.gca().set_facecolor('w')

        if plot_kind == 'imshow':
            cmap1 = plt.get_cmap(cmap) if type(cmap) is str else cmap
            h = plt.imshow(
                lik1[self.ixs_incl, :][:, self.iys_incl].T,
                cmap=cmap1,
                extent=self.extent,
                # extent=[
                #     npy(self.xs[0]) - self.dx / 2.,
                #     npy(self.xs[-1]) + self.dx / 2.
                # ] * 2,
                origin='lower',
                **dict(kw_imshow))
        elif plot_kind == 'contourf':
            # since x0, y0 are fed to contourf, indexing should be kept 'xy'
            # as is default
            x0, y0 = np.meshgrid(*npys(self.xs, self.xs))
            h = plt.contourf(x0, y0, lik1.T, cmap=plt.get_cmap(cmap))
        else:
            raise ValueError()

        plt.axis('equal')
        plt.ylim(self.gen.env.get_ylim_tight())
        plt.xlim(self.gen.env.get_xlim_tight())

        return h

    def plot_p_heading(
        self,
        prob,
        loc=None,
        use_polar=True,
    ):
        """
        :param prob:
        :param heading:
        :param cmap:
        :return:
        """
        if prob is None:
            raise ValueError()
            # prob = npy(self.p_state_incl.flatten())
        else:
            prob = npy(prob)

        if loc is None:
            # Plot likelihood of headings marginalized over locations
            prob1 = np.zeros(self.n_state) + np.nan
            prob1[self.state_incl.flatten()] = prob
            prob1 = prob1.reshape([self.nx, self.ny, self.n_heading])
            prob1 = np2.sumto1(np.nansum(np.nansum(prob1, 0), 0))
        else:
            raise NotImplementedError()

        if use_polar:
            h = plt.polar(
                np.deg2rad(np.r_[self.headings_deg, self.headings_deg[0]]),
                np.r_[prob1, prob1[0]],
                'k-'
            )
        else:
            h = plt.plot(
                np.deg2rad(self.headings_deg),
                prob1,
                'k-'
            )
        return h

    def p_state_incl2p_state_loc_incl(
        self, p_state_incl: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        return p_state_incl.reshape(
            p_state_incl.shape[:-1] +
            torch.Size([self.n_state_loc_incl, self.n_heading])
        ).sum(-1)

    def p_state_loc_incl2p_state_incl(
        self, p_state_loc_incl: torch.Tensor
    ) -> torch.Tensor:
        """

        :param p_state_loc_incl: [..., i_loc_incl]
        :return: p_state_incl[...,
        """
        return (p_state_loc_incl[..., None].expand(
            p_state_loc_incl.shape +
            torch.Size([self.n_heading])
        ) / self.n_heading).reshape(
            p_state_loc_incl.shape[:-1]
            + torch.Size([self.n_state_incl])
        )

    def i_states2states(self, i_states: Iterable[int]) -> torch.Tensor:
        """
        :param i_states: [batch]
        :return: states[batch, (x, y, heading_deg)]
        """
        return torch.stack([self.x[i_states],
                            self.y[i_states],
                            self.heading_deg[i_states]], -1)

    def p_tran_loc_heading2loc(self, p_tran_loc_heading,
                               expand_again=True):
        n_loc = p_tran_loc_heading.shape[0] // self.n_heading
        p = torch.mean(torch.reshape(
            p_tran_loc_heading,
            [n_loc, self.n_heading, n_loc, self.n_heading]
        ), [1, 3], keepdim=True)
        if expand_again:
            p = p + npt.zeros([1, self.n_heading] * 2) / self.n_heading
            p = torch.reshape(p, [
                n_loc * self.n_heading, n_loc * self.n_heading
            ])
        else:
            p = p.squeeze(3).squeeze(1)
        return p

    def p_locOri_locOri2p_loc_loc(
            self, p_locOri_locOri: Union[np.ndarray, torch.Tensor],
            sumto1=None, eps=1e-12,
    ) -> Union[np.ndarray, torch.Tensor]:
        """

        :param p_locOri_locOri:
        :param sumto1: axis to sum to 1 (to get conditional probability).
        Defaults to None, in which case normalization is skipped.
        :param eps: a small number to prevent division by zero during
        normalization. Defaults to 1e-12.
        :return:
        """
        if isinstance(p_locOri_locOri, np.ndarray):
            p = np.nanmean(np.nanmean(np.reshape(
                p_locOri_locOri,
                [self.n_state_loc_incl, self.n_heading] * 2
            ), 3), 1)
            if sumto1 is not None:
                p = np2.sumto1(p + eps, axis=sumto1)
        else:
            p = npt.nanmean(npt.nanmean(torch.reshape(
                p_locOri_locOri,
                [self.n_state_loc_incl, self.n_heading] * 2
            ), 3), 1)
            if sumto1 is not None:
                p = npt.sumto1(p + eps, dim=sumto1)
        return p


class EnvTable(StateTable):
    """
    StateTable + access to attribute of env
    """
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def __getattr__(self, item):
        if item == 'env':
            return None
        else:
            return getattr(self.env, item)


def get_state_table_obj(src: StateTable) -> StateTable:
    s = StateTable(src.env)
    s.set_state_table(src)
    return s


def ____AGENT____():
    pass


def deepcopy_tensor(v):
    if torch.is_tensor(v):
        v = v.clone().detach()
    else:
        v = deepcopy(v)
    return v


class Recordable(object):
    keys = ()

    def __init__(self):
        self.rec = {}
        self.rec_type = None
        self.init_record()

    def init_record(self):
        self.rec = {k: [] for k in self.keys}

    def get_record(self):
        return {k:[npy(self.__dict__[k])] for k in self.keys}

    def add_record(self, rec=None):
        if rec is None:
            rec = self.rec
        for k in self.keys:
            rec[k].append(deepcopy_tensor(self.__dict__[k]))

        if self.rec_type is None:
            self.rec_type = {k: type(self.__dict__[k]) for k in self.keys}
        return rec

    def get_records(self, rec: Dict[str, Iterable] = None
                    ) -> Dict[str, np.ndarray]:
        if rec is None:
            rec = self.rec
        return {k: np.array([
            npy(v1) for v1 in v]) for k, v in rec.items()}

    def set_records(self, rec: Dict[str, Sequence]):
        self.rec = {k: list(rec[k]) for k in self.keys}

    def append_records(self, rec: Dict[str, Sequence]):
        self.rec = {k: self.rec[k] + list(rec[k]) for k in self.keys}

    def reinstate_record(self, i_rec: int = None, rec: dict = None):
        """

        :param i_rec:
        :param rec: use np2.dictlist2listdict(self.get_records())[i_rec]
        :return:
        """
        if rec is None:
            assert i_rec is not None
            rec = rec[i_rec]
        
        for k in self.keys:
            self.__dict__[k] = deepcopy_tensor(rec[k])
            # try:
            #     self.__dict__[k] = deepcopy_tensor(self.rec[k][i_rec])
            # except KeyError:
            #     if k == 'velocity_ego':
            #         self.__dict__[k] = deepcopy_tensor(self.rec['speed'][i_rec])


class AgentState(Recordable):
    keys = ['loc', 'heading', 'velocity_ego', 'dloc', 'dheading',
            'dvelocity_ego']

    def __init__(
            self,
            loc=(0., 0., 0.0402),
            heading=(0., 1., 0.),
            velocity_ego=(1., 0., 0.),
            dloc=None,
            dheading=None,
            dvelocity_ego=None,
    ):
        """

        :param loc:
        :param heading:
        :param velocity_ego: [batch, (forward, rightward)]
        :param dloc:
        :param dheading:
        :param dvelocity_ego: [batch, (forward, rightward)]
        """
        super().__init__()

        self.loc = npt.tensor(loc)
        self.heading = npt.tensor(heading)
        self.velocity_ego = enforce3d(velocity_ego)

        if dloc is None:
            dloc = npt.zeros_like(self.loc)
        self.dloc = dloc

        if dheading is None:
            dheading = npt.zeros_like(self.heading)
        self.dheading = dheading

        if dvelocity_ego is None:
            dvelocity_ego = npt.zeros_like(self.velocity_ego)
        self.dvelocity_ego = enforce3d(dvelocity_ego)


    @property
    def heading_deg(self) -> torch.Tensor:
        return npt.rad2deg(torch.atan2(self.heading[1], self.heading[0]))

    @heading_deg.setter
    def heading_deg(self, v):
        heading_rad = npt.deg2rad(npt.tensor(v))
        self.heading[0] = torch.cos(heading_rad)
        self.heading[1] = torch.sin(heading_rad)

    @property
    def loc_xy(self):
        return self.loc[:2]

    @loc_xy.setter
    def loc_xy(self, xy):
        self.loc[:2] = npt.tensor(xy)

    def set_state(self, loc_xy=None, heading_deg=None, **kwargs):
        """

        :param loc_xy:
        :param heading_deg:
        :param kwargs:
        :return:
        """
        if loc_xy is not None:
            self.loc_xy = npt.tensor(loc_xy)

        if heading_deg is not None:
            self.heading_deg = npt.tensor(heading_deg)

        return self

    def quiver(self, color='r', kw_quiver=(),):
        kw_quiver = {
            'scale_units': 'inches',
            'width': 0.025,
            'headwidth': 4,
            'scale': 5.,
            # 'headlength': 20.,
            # 'headwidth': 40.,
            'edgecolor': color,
            'linewidth': 1.,
            'facecolor': [1., 1., 1., 0.],
            **dict(kw_quiver)
        }

        h = plt.quiver(
            *npys(
                self.loc[0], self.loc[1],
                self.heading[0], self.heading[1],
            ),
            **dict(kw_quiver)
        )
        # plt.axis('equal')
        # plt.axis('square')
        y_len = min([
            np.diff(np.array(plt.ylim())),
            np.diff(np.array(plt.xlim()))
        ])
        # plt.ylim(-y_len / 2, y_len / 2)
        # plt.xlim(-y_len / 2, y_len / 2)
        return h

    def reinstate_record(self, i_rec: int = None, rec: dict = None):
        super().reinstate_record(i_rec, rec)
        for k in self.keys:
            self.__dict__[k] = npt.tensor(self.__dict__[k])


class TactileSensor(Recordable):
    keys=('tactile_input',)

    def __init__(self):
        super().__init__()
        self.tactile_input = None

        pass  # implement in subclasses

    def sense(self, state: AgentState) -> torch.Tensor:
        raise NotImplementedError()


class TactileSensorTabularBinary(TactileSensor):
    def __init__(
            self,
            statetab: StateTable,
            touch_range=0., touch_reliability=0.,
    ):
        """

        :param statetab: either a true or a believed environment
        :param touch_range: defaults to 0 = disabled
        :param touch_reliability: between 0 (default: totally unreliable)
            and 1. (completely reliable)
        """
        super().__init__()
        self.statetab = statetab
        self.touch_range = touch_range
        self.touch_reliability = touch_reliability
        self.tactile_input = npt.zeros(statetab.n_heading)

        i_states = npt.arange(self.statetab.n_state_incl)

        # xy_src[state, 1, xy]
        xy_src = self.statetab.xy[i_states, None]

        # heading_src[state]
        heading_src = self.statetab.heading_deg[i_states]

        # dheadings[side]
        dheadings = self.statetab.headings_deg

        # heading_dst[state, side]
        heading_dst = heading_src[:, None] + dheadings[None, :]

        # xy_dst[state, side, xy]
        xy_dst = xy_src + torch.stack([
            torch.cos(npt.deg2rad(heading_dst)),
            torch.sin(npt.deg2rad(heading_dst))
        ], -1) * self.touch_range

        # in_touch[state, side]
        in_touch = npt.tensor(self.statetab.env.is_crossing_boundary(
            *npys(xy_src, xy_dst))).float()

        # p_touch1[state, side]
        p_touch1 = 1/2 + (self.touch_reliability * (in_touch - 0.5))

        # p_touch[state, side, touched]
        self.p_touch = torch.stack([1. - p_touch1, p_touch1], -1)

    def clone(self, **kwargs) -> 'TactileSensorTabularBinary':
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
            'env': self.statetab,
            'touch_range': self.touch_range,
            'touch_reliability': self.touch_reliability}

    def sense_vec(
            self, i_states: torch.Tensor
    ) -> torch.Tensor:
        """
        :param i_states: [batch]
        :return: touched[batch, side]
        """
        sides = npt.arange(self.statetab.n_heading)

        p_touch1 = self.p_touch[
            i_states[..., None],
            npt.prepend_dim(sides, i_states.ndim), 1]
        touched = distrib.Bernoulli(probs=p_touch1).sample()
        return touched

    def log_prob(
            self,
            touched: torch.Tensor,
            i_states: torch.Tensor = None,
            return_by_side=False,
    ) -> torch.Tensor:
        """
        ...: e.g., [batch, particle_belief]
        :param touched: [..., side]
        :param i_states: [...]
        :param return_by_side
        :return: log_prob[...]
        """
        if i_states is None:
            assert touched.ndim == 2
            assert touched.shape[0] == self.statetab.n_state_incl
            i_states = npt.arange(self.statetab.n_state_incl)
        sides = npt.arange(self.statetab.n_heading)

        # assert i_states.shape == touched.shape[:-1]  # incorrect: ignores broadcasting
        assert touched.shape[-1] == self.statetab.n_heading

        # loglik_touch[..., side]
        loglik_touch = self.p_touch[
            npt.append_dim(i_states, 1), # add side dimension
            npt.prepend_dim(sides, touched.ndim - 1),
            touched.long()].log()
        # assert loglik_touch.shape == touched.shape  # incorrect: ignores broadcasting

        if return_by_side:
            return loglik_touch
        else:
            # pool across sides
            return loglik_touch.sum(-1)


class Measurement(Recordable):
    keys = ('retinal_image', 'dloc', 'dheading', 'dvelocity_ego',
            'tactile_input')

    def __init__(
            self,
            retina=None,
            dloc=None,
            dheading=None,
            dvelocity_ego=None,
            tactile_sensor: TactileSensorTabularBinary = None,
    ):
        """

        :type retinal_image: torch.Tensor
        :type dloc: torch.Tensor
        :type dheading: torch.Tensor
        :type dvelocity_ego: torch.Tensor
        """
        super().__init__()

        if retina is None:
            retina = Retina()
        self.retina = retina
        self.dloc = dloc
        self.dheading = dheading
        # self.dvelocity_ego = dvelocity_ego
        self.tactile_sensor = tactile_sensor

    @property
    def retinal_image(self):
        return self.retina.image


def ____Control_Policy____():
    pass


class Control(Recordable):
    keys = ['dheading_deg', 'velocity_ego']

    def __init__(
            self,
            dheading_deg: Union[torch.Tensor, float] = 0.,
            velocity_ego: Union[torch.Tensor,
                                Sequence[float]] = (0., 0., 0.),
    ):
        """
        :param dheading_deg: [batch] or [batch, (deg_yx, deg_zy, deg_xz)]
        :param velocity_ego: velocity_ego[batch, (forward, rightward)]
        """
        super().__init__()

        if not torch.is_tensor(dheading_deg) \
                or dheading_deg.ndim < 1 \
                or dheading_deg.shape[-1] < 3:
            # dheading_deg[batch, (deg_zy, deg_xz, deg_yx)]
            self.dheading_deg \
                = GenerativeModelSingleEnvTabular.fill_control_heading(
                    dheading_deg)
        else:
            self.dheading_deg = npt.tensor(dheading_deg)
        self.velocity_ego = enforce3d(velocity_ego)

    def get_control_vec(self) -> torch.Tensor:
        """
        :return: control_vec[(forwrad, rightward, dheading_yx)]
        """
        return torch.stack([
            self.velocity_ego, self.dheading_deg[..., -1]], -1)


class Policy:
    def sample(self, size):
        """
        :return: policy; can be converted to control = AgentControl(**policy)
        :rtype: dict
        """
        raise NotImplementedError()

    def get_dict(self) -> Dict[str, str]:
        return {}

    def get_dict_param(self) -> Dict[str, Union[str, float]]:
        """

        :return: dict_param['description (unit)'] = value
            for use in a table listing parameter values in publication
        """
        raise NotImplementedError()


class PolicyDiscrete(Policy):
    def __init__(self, name: str, *args, **kwargs):
        # noinspection PyTypeChecker
        self.actions: Mapping[Union[int, Sequence[int]], Union[dict, Sequence[dict]]] = None
        self.name = name
        self.state = torch.LongTensor([])

    @property
    def n_batch(self):
        return len(self.state)

    @property
    def n_action(self):
        """
        :rtype: int
        """
        return len(self.actions)

    @staticmethod
    def is_action_complete_stop(
        velocity_ego: float, dheading_deg: float
    ) -> bool:
        """Actions without translation or rotation"""
        return (velocity_ego == 0) and (dheading_deg == 0)

    @staticmethod
    def is_action_stop_translation(
        velocity_ego: float, dheading_deg: float
    ) -> bool:
        # return np.isin(
        #     np2.npy(i_action % self.n_action),
        #     np2.npy(self.I_ACTION_STOP % self.n_action)
        # )
        return velocity_ego == 0

    def get_i_actions_except_translation(self) -> Sequence[int]:
        """Get pure rotation and complete stop"""
        return [
            i_action for i_action in range(self.n_action)
            if PolicyDiscrete.is_action_stop_translation(
                **self.actions[i_action]
            )
        ]

    def get_i_actions_except_pure_rotation(self):
        return [
            i_action for i_action in range(self.n_action)
            if (
                PolicyDiscrete.is_action_complete_stop(**self.actions[i_action])
                or not PolicyDiscrete.is_action_stop_translation(
                    **self.actions[i_action])
            )
        ]

    def get_i_action_complete_stop(self) -> int:
        for i_action in range(self.n_action):
            if PolicyDiscrete.is_action_complete_stop(**self.actions[i_action]):
                return i_action

    def update(self, state: torch.LongTensor) -> None:
        """
        For compatiblity with PolicyPellet.
        Makes sample_i_action() return i_action[shape[state]] = i_action[batch]
        :param state: [batch]
        :return: None
        """
        self.state = state

    def sample_i_action(self, size=(), **kwargs) -> Sequence[int]:
        raise NotImplementedError()

    def sample(self, size=()):
        return self.actions[self.sample_i_action(size)]


class PolicyDiscreteRandom(PolicyDiscrete):
    def __init__(self,
                 p_actions: torch.Tensor,
                 actions: Sequence[dict], name: str):
        """

        :type p_actions: torch.Tensor
        :type actions: Sequence[dict]
        """
        super().__init__(name=name)
        self.p_actions = p_actions
        self.actions = actions
        self.name = name
        self.size_batch = ()

    def sample_i_action(self, size=()) -> Sequence[int]:
        return np.random.choice(
            self.n_action,
            size=list(size) + list(self.size_batch),
            replace=True,
            p=npy(self.p_actions)
        )


class PolicyUnitAction(PolicyDiscrete):
    name = Short('ua', 'unit action')

    def __init__(
        self, statetab: StateTable,
        dt_sec: float,
        speed_meter_per_dt: float=None,
        urgency_start_sec: float = 90.,
        urgency_duration_sec: float = 10.,
        **kwargs
    ):
        super().__init__(self.name, **kwargs)
        self.statetab = statetab
        if speed_meter_per_dt is None:
            speed_meter_per_dt = statetab.dx
        self.speed_meter_per_dt = speed_meter_per_dt

        assert dt_sec >= 0
        self.dt_sec = dt_sec

        assert urgency_start_sec >= 0
        self.urgency_start_sec = urgency_start_sec

        assert urgency_duration_sec >= 0
        self.urgency_duration_sec = urgency_duration_sec

    def get_dict_param(self) -> Dict[str, Union[str, float]]:
        return {
            'speed (m/s)': f'{self.speed_meter_per_dt / self.dt_sec}',  # m/s = (m/dt) / (s/dt)
            'onset of urgency (s)': self.urgency_start_sec,
            'deadline for stopping (s)': self.urgency_duration_sec,
        }

    @property
    def speed_meter_per_sec(self) -> float:
        return self.speed_meter_per_dt / self.dt_sec  # (m / dt) / (sec / dt)

    def get_actions(self, to_add_pure_rotation=False) -> List[Dict[str, float]]:
        ddeg = self.statetab.ddeg
        speed = self.speed_meter_per_dt
        assert ddeg == 90., \
            'general case not implemented yet; ' \
            'should allow turning to any angle, ' \
            'with or without moving forward, ' \
            'to avoid the agent giving up when the goal is directly behind, ' \
            'at which point partial turning and moving forward ' \
            'wouldn''t bring the agent closer to the goal.'
        actions = []
        assert np.abs(speed / self.statetab.dx - 1) < 1e-6, \
            ("Only a unit action is supported for now, "
             "so set speed_meter_per_dt to statetab.dx, "
             "and dt_sec to dx / speed_meter_per_sec, "
             "where speed_meter_per_sec is given by the species, "
             "e.g., behav.params_human0.SPEED_METER_PER_SEC, "
             "and dx is set to be the largest value possible "
             "to save compute, as long as a smaller value doesn't change "
             "the results qualitatively.")
        for speed_dx in range(int(np.ceil(speed / self.statetab.dx)), -1, -1):
            for dheading_deg in [-ddeg, +ddeg, +2 * ddeg, 0.]:
                if (
                    (not to_add_pure_rotation)
                    and (speed_dx == 0)
                    and (dheading_deg != 0.)
                ):
                    continue

                actions.append({
                    'dheading_deg': dheading_deg,
                    'velocity_ego': speed_dx * self.statetab.dx,
                })
        # last action is stop: {'dheading_deg':0, 'velocity_ego':0}
        return actions

    def actions2control_vecs(self) -> torch.Tensor:
        return torch.stack([
            self.action2control_vec(action)
            for action in self.actions
        ])

    def action2control_vec(self, action: dict) -> torch.Tensor:
        """

        :param action:
        :return: control_vec[(speed_forward, speed_rightward, dheading_deg)]
        """
        control = npt.tensor([
            action['velocity_ego'], 0., action['dheading_deg']])
        return control


class UNUSED_PolicyGoal(PolicyUnitAction):  # UNUSED: use PolicyPath instead
    """
    Choose control to follow a straight path toward xyd_destination
    """

    name = 'goal'

    def __init__(
        self,
        statetab: StateTable,
        speed_meter_per_dt: float,  # = None,
        state: torch.Tensor = None,
        xyd_destination: torch.Tensor = None,
    ):
        """
        :param statetab:
        :param speed_meter_per_dt:
        :param state: [batch]
        :param xyd_destination: [batch, (x, y, heading_deg)]
        """
        raise DeprecationWarning(
            'use PolicyPath instead, which is strictly more general!'
        )

        assert len(state) == 1, \
            'only n_batch=1 is supported for now, for unambiguous termination'

        super().__init__(
            statetab=statetab,
            speed_meter_per_dt=speed_meter_per_dt,
        )
        self.actions = self.get_actions()

        # DEF: deg_vels[i_action, (deg, vel)]
        self.ddeg_vels = npt.tensor(
            [
                [a['dheading_deg'], a['velocity_ego']]
                for a in self.actions
            ])

        self.statetab = statetab
        self.state = state

        self.xyd_destination = self.discretize_xyd_destination(
            self.statetab, xyd_destination)
        self.state_prev: Union[None, torch.Tensor] = None

    @staticmethod
    def discretize_xyd_destination(
        statetab: StateTable, xyd_destination: torch.Tensor
    ) -> torch.Tensor:
        # descretize destination so that they can be reached
        xyd_destination1 = []
        for xyd0 in xyd_destination:
            if torch.isnan(xyd0[-1]):
                i_state_destination = npt.tensor(
                    statetab.get_i_state(
                        xyd0[:2], statetab.headings_deg[0]
                    ), min_ndim=0
                )
                xyd1 = torch.cat(
                    [
                        statetab.xyd_incl[i_state_destination, :2],
                        npt.tensor([np.nan])], 0
                )
            else:
                i_state_destination = npt.tensor(
                    statetab.get_i_state(state=xyd0), min_ndim=0
                )
                xyd1 = statetab.xyd_incl[i_state_destination]
            xyd_destination1.append(xyd1)
        # xyd_destination[batch, (x,y,d)]
        return torch.stack(xyd_destination1)

    def sample_i_action(
        self,
        size=(),
        # to_plot=False,  # UNUSED
        **kwargs
    ) -> torch.Tensor:
        """
        Sample from the straight path to xyd_destination[batch, xyd]
        Sample from directions weighted by cosine
        Always make one step
        First turn, then move

        :param size:
        :param kwargs:
        # :param to_plot:
        :return: i_action[batch]: self.I_ACTION_STOP to terminate
        """
        x_dst = self.xyd_destination[..., 0]
        y_dst = self.xyd_destination[..., 1]
        d_dst = self.xyd_destination[..., 2]

        x_src = self.statetab.x[self.state]
        y_src = self.statetab.y[self.state]
        d_src = self.statetab.heading_deg[self.state]
        dx = x_dst - x_src
        dy = y_dst - y_src

        same_loc = (dx == 0.) & (dy == 0.)

        # if same_loc.any():
        #     print('same loc!')  # CHECKED
        ddeg, _ = self._get_dheading_deg(d_src, d_dst, same_loc)

        def deg2dxy(deg):
            return (
                torch.sign((torch.cos(deg / 180. * np.pi) * 1e6).round()),
                torch.sign((torch.sin(deg / 180. * np.pi) * 1e6).round())
            )

        dx_sign, dy_sign = [(v * 1e6).round().sign() for v in [dx, dy]]

        deg_bef_turn = d_src
        dx_bef_turn, dy_bef_turn = deg2dxy(deg_bef_turn)

        dx_opposite = (dx_bef_turn - dx_sign).abs() == 2
        dy_opposite = (dy_bef_turn - dy_sign).abs() == 2
        facing_goal = ~same_loc & ~dx_opposite & ~dy_opposite
        opposite_from_goal = ~same_loc & dx_opposite & dy_opposite

        # --- Facing the opposite direction from goal
        # noinspection PyTypeChecker
        clockwise_among_opposite = npt.rand(opposite_from_goal.shape) > 0.5
        ddeg[opposite_from_goal & clockwise_among_opposite] = -90.
        ddeg[opposite_from_goal & ~clockwise_among_opposite] = 90.

        # --- Not facing but not opposite
        not_facing_but_not_opposite = ~same_loc & ~facing_goal & ~opposite_from_goal
        dx_aft_clockwise, dy_aft_clockwise = deg2dxy(deg_bef_turn - 90.)
        # noinspection PyTypeChecker
        clockwise_among_not_opposite = (
            not_facing_but_not_opposite
            & (dx_aft_clockwise != -dx_sign)
            & (dy_aft_clockwise != -dy_sign)
        )
        ddeg[not_facing_but_not_opposite & clockwise_among_not_opposite] = -90.
        ddeg[not_facing_but_not_opposite & ~clockwise_among_not_opposite] = 90.

        # --- Among those facing the goal
        # noinspection PyUnresolvedReferences
        match_y = (npt.rand(size=[(facing_goal).sum()]) < (
            dy[facing_goal].abs()
            / (dx[facing_goal].abs() + dy[facing_goal].abs())))
        dx1 = dx[facing_goal]
        dy1 = dy[facing_goal]
        dx1[match_y] = 0.
        dy1[~match_y] = 0.
        d_dst1 = torch.round(torch.atan2(dy1, dx1) / np.pi * 180.)
        ddeg1, _ = self._get_dheading_deg(d_src[facing_goal], d_dst1)
        ddeg[facing_goal] = ddeg1

        # --- Determine speed:
        #   only move when expecting to face toward the goal after turn
        deg_aft_turn = d_src + ddeg
        dx_aft_turn, dy_aft_turn = deg2dxy(deg_aft_turn)

        will_face_goal = (
                             dx.abs() + dy.abs()
                         ) > (
                             (x_src + dx_aft_turn * self.speed_meter_per_dt - x_dst).abs()
                             + (y_src + dy_aft_turn * self.speed_meter_per_dt - y_dst).abs()
                         )

        vel = npt.zeros(self.state.shape)
        vel[~same_loc & will_face_goal] = self.speed_meter_per_dt

        # DEF: ddeg_vel_states[batch, (ddeg, vel)]
        ddeg_vel_states = torch.stack([ddeg, vel], -1)

        # DEF: ddeg_vels[i_deg_vel, (ddeg, vel)]
        i_action = (
            ddeg_vel_states.T == self.ddeg_vels[..., None]
        ).all(-2).nonzero(as_tuple=False)[..., 0]

        d_next = d_src + ddeg_vel_states[:, 0]
        dx_next, dy_next = deg2dxy(d_next)
        x_next = x_src + dx_next * ddeg_vel_states[:, 1]
        y_next = y_src + dy_next * ddeg_vel_states[:, 1]

        # to terminate
        i_action[same_loc & (ddeg == 0.)] = self.get_i_action_complete_stop()

        if any([
            PolicyDiscrete.is_action_stop_translation(**self.actions[i_action1])
            for i_action1 in i_action
        ]):
            print('PolicyGoal: stop')  # CHECK

        return i_action

    def _get_dheading_deg(
        self, heading_deg_current: torch.Tensor,
        heading_deg_destination: torch.Tensor,
        to_turn: torch.BoolTensor = None
    ):
        """
        :param heading_deg_current: [batch_to_turn]
        :param heading_deg_destination: [batch_to_turn]
        :param to_turn: [batch] if not given, turn all
        :return: dheading_deg[batch], same_heading[batch]
        """
        if to_turn is None:
            # if not given, turn all
            to_turn = npt.ones(heading_deg_destination.shape, dtype=torch.bool)
        dheading_deg = npt.zeros(to_turn.shape)
        same_heading = (
            torch.isnan(heading_deg_destination) | (
                ((heading_deg_current - heading_deg_destination) % 360.).round()
                == 0
            ))
        to_turn = to_turn & ~same_heading
        if to_turn.any():
            diff_deg = torch.round(
                heading_deg_destination
                - heading_deg_current + 180.
            ) % 360. - 180.
            is_clockwise = (
                (diff_deg < 0)
                | (
                    (diff_deg == -180)
                    & npt.randint(
                    high=1, size=[diff_deg.numel()],
                    dtype=torch.bool)
                ))
            dheading_deg[to_turn & is_clockwise] = -90.
            dheading_deg[to_turn & ~is_clockwise] = 90.
        return dheading_deg, same_heading


class PolicyBelief(PolicyUnitAction):
    name = Short('bel', 'belief')

    def __init__(
        self,
        obs_bel: 'Observer2D',
        w_state_bel0: torch.Tensor,
        i_state_bel0: torch.Tensor = None,
        speed_meter_per_dt: float = None,
        sigma_value: float = None,
        urgency_start_sec: float = 40.,
        urgency_duration_sec: float = 10.,
    ):
        """

        :param obs_bel:
        :param w_state_bel0: initial state weights
        :param i_state_bel0: initial state indices
        :param speed_meter_per_dt:
        :param sigma_value:
        :param urgency_start_sec: time step at which to start urgency
        :param urgency_duration_sec: decision is always made
            by urgency_start + urgency_duration
        """
        super().__init__(
            statetab=obs_bel,
            speed_meter_per_dt=speed_meter_per_dt,
            dt_sec=obs_bel.dt,
            urgency_start_sec=urgency_start_sec,
            urgency_duration_sec=urgency_duration_sec,
        )
        self.obs_bel = obs_bel

        self.w_state_bel = w_state_bel0
        if i_state_bel0 is None:
            assert self.w_state_bel.shape[1] == self.obs_bel.n_state_incl
            i_state_bel0 = npt.arange(
                self.obs_bel.n_state_incl
            )[None].expand([self.n_batch, self.n_particle])
        else:
            assert self.w_state_bel.shape == i_state_bel0.shape
            assert (
                i_state_bel0 == npt.arange(
                    self.obs_bel.n_state_incl
                )[None].expand([self.n_batch, self.n_particle])
            ).all()
            assert self.n_batch == 1, \
                'only n_batch=1 is supported for now!'
        self.i_state_bel = i_state_bel0

        if sigma_value is None:
            sigma_value = self.obs_bel.dx
        self.sigma_value = sigma_value

        self.actions = self.get_actions()
        # control_vecs[i_action, (speed_forward, speed_rightward, dheading_deg)]
        self.control_vecs = self.actions2control_vecs()

    def get_dict_param(self) -> Dict[str, Union[str, float]]:
        return {
            **super().get_dict_param(),
            'decision temperature (m)': self.sigma_value,
        }

    @property
    def n_batch(self) -> int:
        return self.w_state_bel.shape[0]

    @property
    def n_particle(self) -> int:
        return self.w_state_bel.shape[1]

    def get_dict(self) -> Dict[str, str]:
        return np2.shorten_dict({
            **super().get_dict(),
            Short('sg', 'sigma marker (meter)'): f'{self.sigma_value}',
        }, shorten_zero=True)

    def update_beliefs(
        self, w_state_belief: torch.Tensor, i_state_belief: torch.Tensor
    ):
        """

        :param w_state_belief: [batch, i_particle]
        :param i_state_belief: [batch, i_particle]
        :return:
        """
        assert w_state_belief.shape == torch.Size(
            [self.n_batch, self.n_particle])
        assert i_state_belief.shape == torch.Size(
            [self.n_batch, self.n_particle])
        self.w_state_bel = w_state_belief
        self.i_state_bel = i_state_belief

    def sample_i_action(
        self,
        size=(),
        step: int = None,
        to_plot=False,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Select based on counterfactual beliefs
        :param size:
        :param to_plot:
        :param kwargs:
        :return: i_action[batch]: self.I_ACTION_STOP to terminate
        """

        value_action = []
        p_pred_locs = []
        for i_action in range(self.n_action):
            p_pred_loc = self.obs_bel.p_state_incl2p_state_loc_incl(
                self.predict_belief(i_action))

            value_action.append(self.get_value(p_pred_loc))
            p_pred_locs.append(p_pred_loc)

        p_pred_locs = torch.stack(p_pred_locs, 0)

        # value_action[batch, i_action] = float
        value_action = torch.stack(value_action, -1)

        # p_action[batch, i_action]
        p_action = torch.softmax(
            value_action / (self.sigma_value ** 2),
            dim=-1
        )

        # Add urgency to prevent infinite loop
        assert step is not None
        urgency = npt.zeros_like(p_action)
        urgency[:, self.get_i_action_complete_stop()] = 1.
        t = step * self.dt_sec
        p_urgency = np.clip(
            (t - self.urgency_start_sec) / self.urgency_duration_sec,
            a_min=0., a_max=1.
        )
        p_action = p_action * (1 - p_urgency) + p_urgency * urgency

        # Sample action
        i_action = npt.categrnd(probs=p_action)

        if to_plot:
            self.plot_sample_i_action(i_action, p_pred_locs, value_action)

        # print(torch.softmax(value_action / (self.sigma_value ** 2), -1))
        return i_action

    def plot_sample_i_action(self, i_action, p_pred_locs, value_action):
        p_choice = torch.softmax(value_action / (self.sigma_value ** 2), -1)
        axs = plt2.GridAxes(
            2, self.n_action, top=1.5, widths=2, heights=2, left=1.5
        )
        for i_action1, (value1, p_pred_loc1, p_choice1) in enumerate(
            zip(
                value_action[0], p_pred_locs, npy(p_choice[0])
            )
        ):
            plt.sca(axs[0, i_action1])

            p_pred_state = self.predict_belief(i_action1)
            # p_pred_loc = self.obs_bel.p_state_incl2p_state_loc_incl(
            #     p_pred_state)

            self.obs_bel.env.plot_walls()
            self.obs_bel.plot_p_loc(
                p_pred_loc1[0], cmap='gray_r',
                kw_imshow={'vmin': 0}
            )
            plt.title(
                'dh_deg:%g\nvel:%g\nvalue - max value:%g\nP(choice):%g' % (
                    self.actions[i_action1]['dheading_deg'],
                    self.actions[i_action1]['velocity_ego'],
                    npy(value1 - value_action.max()),
                    p_choice1
                )
            )

            plt.sca(axs[1, i_action1])
            self.obs_bel.plot_p_heading(p_pred_state[0], use_polar=False)
        plt2.rowtitle(['P(loc)', 'P(heading)'], axes=axs)
        axs.suptitle(
            f'sigma: {self.sigma_value:g}, '
            f'i_action_choice:{i_action}', pad=1.
        )
        localfile = LocalFile('../Data_Nav/slam_pixel_grid/PolicyBelief')
        fname = localfile.get_file_fig('sample_i_action')
        plt2.savefig(fname)
        axs.close()

    def predict_belief(self, i_action: int) -> torch.Tensor:
        """

        :param i_action:
        :return: p_pred[batch, i_state_incl_bel]
        """
        assert self.n_batch == 1, 'n_batch > 1 not implemented yet!'
        assert self.n_particle == self.obs_bel.n_state_incl, \
            'n_particle != n_state_incl not implemented yet!'

        control = self.control_vecs[i_action]

        p_pred = self.obs_bel.prediction_step(
            Control(control[2], control[:2]),
            p_state_incl=self.w_state_bel[0],
            return_p_state_incl=True,
            to_plot=False)
        return p_pred[None]

    def get_value(
        self, p_pred_loc: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError()


class PolicyBeliefDistance(PolicyBelief):
    name = Short('dbel', 'distance belief')

    def __init__(
        self,
        obs_bel: 'Observer2D',
        p_distance_bel: torch.Tensor,
        distance_bel: torch.Tensor,
        w_state_bel0: torch.Tensor,
        i_state_bel0: torch.Tensor,
        w_loc_bel0: torch.Tensor,
        i_loc_bel0: torch.Tensor,
        speed_meter_per_dt: float = None,
        sigma_value: float = None,
    ):
        """

        :param obs_bel: observer in the belief space
        :param p_distance_bel: [batch, i_distance]
        :param distance_bel: [batch, i_distance]
        :param p_destination_bel: [batch, i_loc_incl]
        :param w_state_bel0: [batch, i_particle]
            where 0 <= i_state_incl < obs_bel.n_state_incl
        :param i_state_bel0: [batch, i_particle]
            If None, taken as torch.arange(obs_bel.n_state_incl)
        :param w_loc_bel0: [batch, i_loc]
        :param i_loc_bel0: [batch, i_loc]
        :param speed_meter_per_dt: if None, set to obs_bel.dx
        :param sigma_value: acts as the temperature parameter for the softmax.
            Divides the distance to the goal.
        """
        super().__init__(
            obs_bel=obs_bel,
            w_state_bel0=w_state_bel0,
            i_state_bel0=i_state_bel0,
            speed_meter_per_dt=speed_meter_per_dt,
            sigma_value=sigma_value
        )

        assert p_distance_bel.shape == distance_bel.shape
        self.p_distance_bel = p_distance_bel.flatten()  # [i_dist]
        self.distance_bel = distance_bel.flatten()  # [i_dist]

        # [i_loc]
        self.p_loc_bel0 = w_loc_bel0.clone()

        # [i_loc]
        self.xy_loc_bel0 = torch.stack([
            self.obs_bel.x_loc_incl[i_loc_bel0],
            self.obs_bel.y_loc_incl[i_loc_bel0]
        ], -1)

    def get_dict_param(self) -> Dict[str, Union[str, float]]:
        return super().get_dict_param()

    def get_value(
        self, p_pred_loc: torch.Tensor,
        p_loc_bel0: torch.Tensor = None,
        xy_loc_bel0: torch.Tensor = None,
    ) -> torch.Tensor:
        """

        :param p_pred_loc: [batch, i_loc_incl]
        :param p_loc_bel0: [batch, particle]
        :param i_loc_bel0: [batch, particle, (x, y)]
        :return: value[batch]
        """
        p_distance_bel = self.p_distance_bel
        distance_bel = self.distance_bel

        if p_loc_bel0 is None:
            p_loc_bel0 = self.p_loc_bel0
        if xy_loc_bel0 is None:
            xy_loc_bel0 = self.xy_loc_bel0
        assert p_loc_bel0.shape == torch.Size(xy_loc_bel0.shape[:-1])

        # [loc_incl, xy]
        xy_loc_incl = torch.stack([
            self.obs_bel.x_loc_incl, self.obs_bel.y_loc_incl], -1)

        def _get_value(decompose_value: bool):
            if decompose_value:
                # TODO: consider updating following PolicyBeliefGoal.get_value()._get_value()
                raise NotImplementedError()
            else:
                # [..., i_loc_now, i_loc_start, 1]
                dist_pred = (
                    # [..., i_loc_now, i_loc_start, i_dist, xy]
                    torch.linalg.vector_norm(
                        xy_loc_incl[..., :, None, None, :]
                        - xy_loc_bel0[..., None, :, None, :],
                        dim=-1
                    )
                )

                # value[batch]: expected squared difference between
                #   the predicted and goal distances
                value = -(
                    # [..., i_loc_now, i_loc_start, i_dist]
                    p_pred_loc[..., :, None, None] *
                    p_loc_bel0[..., None, :, None] *
                    p_distance_bel[..., None, None, :]
                    * (dist_pred - distance_bel[..., None, None, :]) ** 2
                ).sum([-3, -2, -1])
            return value

        # t_st = time.time()
        value0 = _get_value(False)
        # t_en = time.time()
        # print(f'time elapsed: {t_en - t_st:.3f} sec')
        return value0

    def get_actions(self, to_add_pure_rotation=False) -> List[Dict[str, float]]:
        assert not to_add_pure_rotation, \
            'pure rotation is not supported for PolicyBeliefDistance'
        return [
            {'dheading_deg': 0., 'velocity_ego': self.speed_meter_per_dt},
            {'dheading_deg': 0., 'velocity_ego': 0.},  # stop at destination
        ]


class PolicyBeliefGoal(PolicyBelief):
    name = Short('gbel', 'goal belief')

    def __init__(
        self,
        obs_bel: 'Observer2D',
        p_destination_loc_bel: torch.Tensor,
        w_state_bel0: torch.Tensor,
        i_state_bel0: torch.Tensor = None,
        speed_meter_per_dt: float = None,
        sigma_value: float = None,
        urgency_start_sec: float = 40.,
        urgency_duration_sec: float = 10.,
    ):
        """

        :param obs_bel: observer in the belief space
        :param p_destination_loc_bel: [batch, i_loc_incl]
        :param w_state_bel0: [batch, i_particle]
            where 0 <= i_state_incl < obs_bel.n_state_incl
        :param i_state_bel0: [batch, i_particle]
            If None, taken as torch.arange(obs_bel.n_state_incl)
        :param speed_meter_per_dt: if None, set to obs_bel.dx
        :param sigma_value: acts as the temperature parameter for the softmax.
            Divides the distance to the goal.
        """
        super().__init__(
            obs_bel=obs_bel, w_state_bel0=w_state_bel0,
            i_state_bel0=i_state_bel0,
            speed_meter_per_dt=speed_meter_per_dt, sigma_value=sigma_value,
            urgency_start_sec=urgency_start_sec, urgency_duration_sec=urgency_duration_sec
        )

        # [batch, i_loc_incl]
        assert p_destination_loc_bel.shape[-1] == self.obs_bel.n_state_loc_incl
        self.p_destination_bel = p_destination_loc_bel

    def get_value(
        self,
        p_pred_loc: torch.Tensor,
        p_destination_loc: torch.Tensor = None,
        to_check=False
    ) -> torch.Tensor:
        """

        :param p_pred_loc: [batch, i_loc_incl]
        :param p_destination_loc: [batch, i_loc_incl]
            Defaults to self.p_destination_bel
        :param to_check: if True, check that the value is the same
            when decomposed into x and y
        :return: value[batch]
        """
        if p_destination_loc is None:
            p_destination_loc = self.p_destination_bel

        # speed up PolicyBeliefGoal.get_value()
        #   by computing dx^2 and dy^2 for unique combinations of x and y
        #   Use sum_{x'} sum_x P(x') P(x|a) (x-x')**2
        #   = <x'>**2 - 2<x'><x> + <x**2>
        #   where x' is the goal location, a is the action

        def _get_value(decompose_value: bool):
            if decompose_value:
                x = self.obs_bel.x_loc_incl
                x2 = x ** 2
                y = self.obs_bel.y_loc_incl
                y2 = y ** 2
                mxpred2 = (x2 * p_pred_loc).sum(-1)
                mxpred = (x * p_pred_loc).sum(-1)
                mypred2 = (y2 * p_pred_loc).sum(-1)
                mypred = (y * p_pred_loc).sum(-1)
                mxgoal2 = (x2 * p_destination_loc).sum(-1)
                mxgoal = (x * p_destination_loc).sum(-1)
                mygoal2 = (y2 * p_destination_loc).sum(-1)
                mygoal = (y * p_destination_loc).sum(-1)
                value = -(
                    (mxpred2 - 2 * mxpred * mxgoal + mxgoal2)
                    + (mypred2 - 2 * mypred * mygoal + mygoal2)
                )
            else:
                # xy[i_loc, xy]
                xy = torch.stack([
                    self.obs_bel.x_loc_incl,
                    self.obs_bel.y_loc_incl], -1)
                dist = torch.linalg.vector_norm(
                    xy[:, None, :] - xy[None, :, :], dim=-1, ord=2)

                value = -(
                    dist[None] ** 2
                    * p_pred_loc[..., :, None]
                    * p_destination_loc[..., None, :]
                ).sum([-1, -2])
            return value

        value1 = _get_value(decompose_value=True)
        if to_check:
            value0 = _get_value(decompose_value=False)
            assert torch.allclose(value0, value1)

        # value0 = -(
        #     dist[None] ** 2
        #     # * p_pred_loc[..., :, None]
        #     * p_destination_loc[..., None, :]
        # ).sum([-1])
        # plt.imshow(np2.npy(value0.reshape([5, 5])).T, origin='lower')
        # plt.colorbar()
        # plt.show()

        return value1


class PolicyPath(PolicyUnitAction):
    name = 'path'

    def __init__(
        self,
        obs_tru: 'Observer2D',
        speed_meter_per_dt: float = None,
        state: torch.LongTensor = None,
        xyd_destinations: torch.Tensor = None,
        urgency_start_sec: float = np.inf,
        urgency_duration_sec: float = 10.,
    ):
        """
        :param statetab:
        :param speed:
        :param state: [batch]
        :param xyd_destinations: [waypoint, batch, (x, y, heading_deg)]
        """
        super().__init__(
            statetab=obs_tru,
            speed_meter_per_dt=speed_meter_per_dt,
            dt_sec=obs_tru.dt,
            urgency_start_sec=urgency_start_sec,
            urgency_duration_sec=urgency_duration_sec,
        )

        self.state = state
        n_batch = self.n_batch

        self.waypoint_dest = npt.zeros([n_batch], dtype=torch.long)

        self.xyd_destinations = torch.stack([
            UNUSED_PolicyGoal.discretize_xyd_destination(obs_tru, xyd_destination)
            for xyd_destination in xyd_destinations
        ])
        xyd_destination_loc = self.xyd_destinations[0, :, :2]  # first step, across all batches
        # noinspection PyTypeChecker
        self.policy_goal = PolicyBeliefGoal(
            obs_bel=obs_tru,
            p_destination_loc_bel=self.xy2p_loc(xyd_destination_loc),
            i_state_bel0=npt.arange(obs_tru.n_state_incl).expand([
                n_batch, obs_tru.n_state_incl
            ]),
            w_state_bel0=self.state2p_state(state),
            speed_meter_per_dt=speed_meter_per_dt,
            sigma_value=1e-6,
            urgency_start_sec=urgency_start_sec,
            urgency_duration_sec=urgency_duration_sec,
        )
        self.actions = self.get_actions(to_add_pure_rotation=True)

    def get_dict_param(self) -> Dict[str, Union[str, float]]:
        dict_param = {
            **self.policy_goal.get_dict_param()
        }
        dict_param[
            'onset of urgency for passive navigation (s)'
        ] = dict_param.pop('onset of urgency (s)')
        dict_param[
            'decision temperature for passive navigation (m)'
        ] = dict_param.pop('decision temperature (m)')
        return dict_param

    @property
    def n_batch(self) -> int:
        return len(self.state)

    def state2p_state(self, state: torch.LongTensor) -> torch.Tensor:
        """
        return delta distributions that peak at state
        :param state: [batch]
        :return: p_state[batch, i_state_incl]
        """
        # noinspection PyTypeChecker
        return npt.sumto1(
            npt.arange(self.statetab.n_state_incl)[None, :] == state[:, None],
            -1
        )

    def xy2p_loc(self, xyd_destination_loc: torch.Tensor) -> torch.Tensor:
        """

        :param xyd_destination_loc: [batch, (x, y)]
        :return: p_dest_loc: [batch, i_loc_incl]
        """
        # noinspection PyTypeChecker
        return npt.sumto1(
            torch.all(
                self.statetab.xy_loc_incl_flat.mT[None, :, :]
                == xyd_destination_loc[:, None, :],
                dim=-1  # both x and y should be the same
            ), 1  # sum to 1 across i_loc_incl
        )

    @property
    def n_waypoints_max(self) -> int:
        return self.xyd_destinations.shape[0]

    @property
    def xyd_destination(self) -> torch.Tensor:
        return torch.stack([
            self.xyd_destinations[i_waypoint, i_batch]
            for i_batch, i_waypoint in enumerate(self.waypoint_dest)
        ])

    def update(self, state: torch.LongTensor) -> None:
        """
        For compatiblity with PolicyPellet.
        Makes sample_i_action() return i_action[shape[state]] = i_action[batch]
        :param state: [batch]
        :return: None
        """
        self.state = state

    def sample_i_action(
        self, size=(), to_plot=None, step: int = None, **kwargs
    ) -> Sequence[int]:
        self.policy_goal.w_state_bel = self.state2p_state(self.state)
        self.policy_goal.p_destination_bel = self.xy2p_loc(
            self.xyd_destination[..., :2])
        i_action = self.policy_goal.sample_i_action(
            size=size, step=step, to_plot=to_plot)
        actions = [
            self.policy_goal.actions[i_action1] for i_action1 in i_action
        ]

        assert size == (), \
            'checking next_waypoint not implemented for nonempty size'

        next_waypoint = npt.ones([self.n_batch], dtype=torch.bool)
        while next_waypoint.any():
            # Allow waypoint_dest to increase even when
            # some waypoints are identical to others,
            # making policy_goal to sample i_action_stop.
            # noinspection PyTypeChecker
            next_waypoint = (
                torch.all(
                    self.statetab.xy[self.state, :2]
                    == self.xyd_destination[..., :2],
                    dim=-1
                ) & ((
                        self.statetab.heading_deg[self.state]
                        == self.xyd_destination[..., 2]
                    ) | torch.isnan(self.xyd_destination[..., 2])
                ) & (self.waypoint_dest < self.n_waypoints_max - 1)
            )
            # next_waypoint = (
            #     npt.tensor([
            #         PolicyDiscrete.is_action_stop_translation(
            #             **action1
            #         ) for action1 in actions
            #     ]) & (self.waypoint_dest < self.n_waypoints_max - 1)
            # )
            self.waypoint_dest = self.waypoint_dest + next_waypoint.long()

            # print(actions)  # CHECKED

            if next_waypoint.any():
                self.policy_goal.p_destination_bel = self.xy2p_loc(
                    self.xyd_destination[..., :2])
                i_action[next_waypoint] = self.policy_goal.sample_i_action(
                    step=step,
                    to_plot=to_plot,  # CHECKED
                )[next_waypoint]
                actions = [
                    self.policy_goal.actions[i_action1]
                    for i_action1 in i_action
                ]

        for i, (action1, d_dest, state1) in enumerate(zip(
            actions,
            self.xyd_destination[..., 2],
            self.state
        )):
            if (
                PolicyDiscrete.is_action_stop_translation(**action1)
                and not np.isnan(d_dest)
            ):
                # turn to match the heading direction
                heading_difference = (np2.npy(
                    d_dest
                    - self.statetab.heading_deg[state1]
                ) + 180) % 360 - 180
                if isinstance(heading_difference, np.ndarray):
                    heading_difference[heading_difference == -180] = 180
                else:
                    if heading_difference == -180:
                        heading_difference = 180.
                speeds = np2.npy(
                    [action1['velocity_ego'] for action1 in self.actions])
                dheadings = np2.npy(
                    [action1['dheading_deg'] for action1 in self.actions])
                i_action[i] = npt.tensor(np.nonzero(
                    np2.issimilar(speeds, 0.) &
                    np2.issimilar(dheadings, np2.npy(heading_difference))
                )[0][0])
            else:
                for i_action1 in range(self.n_action):
                    if self.actions[i_action1] == action1:
                        i_action[i] = i_action1
                        break

        # # translate actions to i_action
        # i_action = []
        # for action1 in actions:
        #     for i_action1 in range(self.n_action):
        #         if self.actions[i_action1] == action1:
        #             i_action.append(i_action1)
        #             break

        return i_action


class UNUSEDPolicyGoalPreplan(UNUSED_PolicyGoal):
    """
    Pre-plan trajectory to the goal assuming no transition noise
    """
    name = 'preplan'

    def __init__(
        self,
        statetab: EnvTable,
        speed_meter_per_dt: float = None,
        state: torch.Tensor = None,
        xyd_destination: torch.Tensor = None,
    ):
        """
        :param statetab:
        :param speed_meter_per_dt:
        :param state: [batch]
        :param xyd_destination: [batch, (x, y, heading_deg)]
        """
        assert len(state) == 1, \
            'only n_batch=1 is supported for now, for unambiguous termination'

        super().__init__(
            statetab=statetab,
            speed_meter_per_dt=speed_meter_per_dt,
            state=state,
            xyd_destination=xyd_destination
        )

        # def is_at_goal() -> torch.BoolTensor:
        #     same_loc = (
        #         env.x[self.state] == xyd_destination[..., 0]
        #     ) & (
        #         env.y[self.state] == xyd_destination[..., 1]
        #     )
        #     same_heading = (
        #         env.heading_deg[self.state] == xyd_destination[..., 2]
        #     ) | torch.isnan(xyd_destination[..., 2])
        #     return same_loc & same_heading

        self.i_actions = []
        state0 = self.state
        self.state = state0.clone()
        # while not is_at_goal():
        while True:
            i_action = self.sample_i_action(use_preplan=False)
            self.i_actions.append(i_action)
            self.update(self.simulate_action_wo_noise(self.state, i_action))
            if PolicyDiscrete.is_i_action_stop_translation(**self.actions[i_action]).all():
                break
        self.state = state0

    def simulate_action_wo_noise(
        self,
        state: torch.Tensor,
        i_action: torch.Tensor
    ) -> torch.Tensor:
        """
        :param state: [batch]
        :param i_action: [batch]
        :return: state_next[batch]
        """
        state_next = []
        for state0, i_action1 in zip(state, i_action):
            x = self.statetab.x[state0]
            y = self.statetab.y[state0]
            d = self.statetab.heading_deg[state0]

            action1 = self.actions[i_action1]

            d = torch.round(d + action1['dheading_deg'])
            dx = torch.cos(npt.deg2rad(d))
            dy = torch.sin(npt.deg2rad(d))
            x = x + action1['velocity_ego'] * dx
            y = y + action1['velocity_ego'] * dy
            state_next.append(self.statetab.get_i_state(
                npt.tensor([x, y]), d
            ))
        return npt.tensor(np.array(state_next))

    def sample_i_action(
        self,
        size=(),
        to_plot=False,  # PARAM CHECKED PolicyPellet.sample_i_action()
        use_preplan=True,
        step=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample from the straight path to xyd_destination[batch, xyd]
        Sample from directions weighted by cosine
        Always make one step
        First turn, then move

        :param size:
        :param kwargs:
        :param to_plot:
        :return: i_action[batch]: self.I_ACTION_STOP to terminate
        """
        if use_preplan and step is not None:
            return self.i_actions[step]
        else:
            return super().sample_i_action(
                size=size, to_plot=to_plot, **kwargs)


class PolicyPathPellet(PolicyPath):
    name = 'ppellet'

    def __init__(
        self,
        obs_tru: 'Observer2D',
        speed_meter_per_dt: float = None,
        state: torch.LongTensor = None,
        xyd_destinations: torch.Tensor = None,
        urgency_start_sec: float = np.inf,
        urgency_duration_sec: float = 10.,
        **kwargs
    ):
        """
        :param obs_tru:
        :param speed_meter_per_dt:
        :param state: [batch]
        :param xyd_destinations: omit for PolicyPathPellet;
            generated automatically to be the least visited state
        :param urgency_start_sec: defaults to inf to avoid stopping
        :param urgency_duration_sec: ignored by default
        """

        if state is None:
            state = npt.randint(obs_tru.n_state_incl, [1])
        n_batch = len(state)

        # xyd_destinations[waypoint, batch, (x, y, heading_deg)]:
        #   unlike PolicyPath, the first is set randomly and updated after
        #   arriving at each destination to the state with the least number of
        #   visits.
        assert xyd_destinations is None
        xyd_destinations = obs_tru.xyd_incl[npt.randint(
            obs_tru.n_state_incl,
            [1, n_batch]
        )]

        super().__init__(
            obs_tru=obs_tru, speed_meter_per_dt=speed_meter_per_dt,
            state=state, xyd_destinations=xyd_destinations,
            urgency_start_sec=urgency_start_sec, urgency_duration_sec=urgency_duration_sec,
            **kwargs
        )
        self.n_visits = npt.zeros([
            self.n_batch, obs_tru.n_state_incl], dtype=torch.long
        )

    def update(self, state: torch.LongTensor ) -> None:
        """
        Also update the number of visits

        For compatiblity with PolicyPellet.
        Makes sample_i_action() return i_action[shape[state]] = i_action[batch]
        :param state: [batch]
        :return: None
        """
        super().update(state)
        for i_batch, state1 in enumerate(state):
            self.n_visits[i_batch, state] += 1

    def sample_i_action(
        self, size=(), to_plot=None, step: int = None, **kwargs
    ) -> Sequence[int]:
        """
        Always choose the next destination that is least visited whenever the
        state arrives at the current destination.
        :param size:
        :param to_plot:
        :param step:
        :param kwargs:
        :return:
        """
        assert size == (), 'nonempty size not implemented yet!'

        def is_stop(i_action11: Sequence[int]) -> Sequence[bool]:
            return [
                self.is_action_complete_stop(**self.actions[i_action1])
                for i_action1 in i_action11
            ]

        def choose_least_visited_destination(i_batch: int) -> int:
            state_candidate = npt.arange(self.statetab.n_state_incl)[
                self.n_visits[i_batch] == self.n_visits[i_batch].min()
            ]
            return state_candidate[npt.randint(len(state_candidate), size=())]

        i_action = super().sample_i_action(
            size=size, to_plot=to_plot, step=step, **kwargs
        )

        while np.any(is_stop(i_action)):
            for i_batch, is_stop1 in enumerate(is_stop(i_action)):
                if is_stop1:
                    choose_least_visited_destination(i_batch)
                    self.xyd_destinations[i_batch] = self.statetab.xyd_incl[
                        choose_least_visited_destination(i_batch)
                    ]
            i_action = super().sample_i_action(
                size=size, to_plot=to_plot, step=step, **kwargs
            )
        return i_action


class UNUSED_PolicyPellet(PolicyUnitAction):  # UNUSED: used PolicyPathPellet instead
    name = 'pellet'

    """
    Pursue pellets dropped so that min number of visit to a state is maximized.
    """
    def __init__(
        self,
        statetab: StateTable,
        speed_meter_per_dt: float = None,
        state: torch.Tensor = None,
        # state_destination: torch.Tensor = None,
    ):
        """
        """
        raise DeprecationWarning('use PolicyPathPellet for more robust control!')

        super().__init__(
            statetab=statetab,
            speed_meter_per_dt=speed_meter_per_dt
        )
        self.n_visit: Union[None, torch.Tensor] = None

        self.actions = self.get_actions()

        # DEF: deg_vels[i_action, (deg, vel)]
        self.ddeg_vels = npt.tensor(
            [
                [a['dheading_deg'], a['velocity_ego']]
                for a in self.actions
            ])

        self.state = state
        self.state_destination = None

    def update(self, state: torch.Tensor):
        """

        :param state: [batch]
        """
        if self.n_visit is None:
            self.n_visit = npt.zeros(
                [state.numel(), self.statetab.n_state_incl])

        self.n_visit[
            npt.arange(state.numel()),
            state.flatten()
        ] = self.n_visit[
            npt.arange(state.numel()),
            state.flatten()
        ] + 1
        if self.state is not None:
            self.state_prev = npt.dclone(self.state)
        self.state = state

        self.choose_destination()

    def choose_destination(self):
        if self.state_destination is None:
            self.state_destination = npt.dclone(self.state)
        to_choose_destination = (
                self.state_destination == self.state).flatten()
        if to_choose_destination.sum() > 0:
            for i_batch in to_choose_destination.nonzero()[..., 0]:
                n_visit = npt.dclone(self.n_visit[i_batch])
                # noinspection PyTypeChecker
                n_visit[self.state[i_batch]] = np.inf
                candidate = npt.tensor(
                    np.nonzero(npy(n_visit == n_visit.min()).flatten())[0]
                )
                self.state_destination[i_batch] = candidate[npt.randint(
                    len(candidate), ())]
                if self.state[i_batch] == self.state_destination[i_batch]:
                    print('destination == state!')

    @property
    def xyd_destination(self):
        if self.state_destination is None:
            return None
        else:
            return torch.stack([
                self.statetab.x[self.state_destination],
                self.statetab.y[self.state_destination],
                self.statetab.heading_deg[self.state_destination],
            ], -1)

    def sample_i_action(
        self,
        size=(),
        to_plot=False,  # PARAM CHECKED PolicyPellet.sample_i_action()
        **kwargs
    ):
        """
        sample from the straight path to xyd_destination[batch, xyd]
        sample from directions weighted by cosine
        always make one step

        :param size:
        :param kwargs:
        :param to_plot:
        :return: i_action[batch]
        """
        # noinspection PyUnresolvedReferences
        assert (self.state_destination != self.state).all()

        x_dst = self.statetab.x[self.state_destination]
        y_dst = self.statetab.y[self.state_destination]
        d_dst = self.statetab.heading_deg[self.state_destination]

        x_src = self.statetab.x[self.state]
        y_src = self.statetab.y[self.state]
        d_src = self.statetab.heading_deg[self.state]
        dx = x_dst - x_src
        dy = y_dst - y_src

        same_loc = (dx == 0.) & (dy == 0.)
        # if same_loc.any():
        #     print('same loc!')
        ddeg = self.get_dheading_deg(d_src, d_dst, same_loc)

        def deg2dxy(deg):
            return (
                torch.sign((torch.cos(deg / 180. * np.pi) * 1e6).round()),
                torch.sign((torch.sin(deg / 180. * np.pi) * 1e6).round())
            )

        dx_sign, dy_sign =  [(v * 1e6).round().sign() for v in [dx, dy]]

        deg_bef_turn = d_src
        dx_bef_turn, dy_bef_turn = deg2dxy(deg_bef_turn)

        dx_opposite = (dx_bef_turn - dx_sign).abs() == 2
        dy_opposite = (dy_bef_turn - dy_sign).abs() == 2
        facing_goal = ~same_loc & ~dx_opposite & ~dy_opposite
        opposite_from_goal = ~same_loc & dx_opposite & dy_opposite

        # --- Facing the opposite direction from goal
        # noinspection PyTypeChecker
        clockwise_among_opposite = npt.rand(opposite_from_goal.shape) > 0.5
        ddeg[opposite_from_goal & clockwise_among_opposite] = -90.
        ddeg[opposite_from_goal & ~clockwise_among_opposite] = 90.

        # --- Not facing but not opposite
        not_facing_but_not_opposite = ~same_loc & ~facing_goal & ~opposite_from_goal
        dx_aft_clockwise, dy_aft_clockwise = deg2dxy(deg_bef_turn - 90.)
        # noinspection PyTypeChecker
        clockwise_among_not_opposite = (
            not_facing_but_not_opposite
            & (dx_aft_clockwise != -dx_sign)
            & (dy_aft_clockwise != -dy_sign)
        )
        ddeg[not_facing_but_not_opposite & clockwise_among_not_opposite] = -90.
        ddeg[not_facing_but_not_opposite & ~clockwise_among_not_opposite] = 90.

        # --- Among those facing the goal
        # noinspection PyUnresolvedReferences
        match_y = npt.rand(size=[(facing_goal).sum()]) < (
                dy[facing_goal].abs()
                / (dx[facing_goal].abs() + dy[facing_goal].abs()))
        dx1 = dx[facing_goal]
        dy1 = dy[facing_goal]
        dx1[match_y] = 0.
        dy1[~match_y] = 0.
        d_dst1 = torch.round(torch.atan2(dy1, dx1) / np.pi * 180.)
        ddeg1 = self.get_dheading_deg(d_src[facing_goal], d_dst1)
        ddeg[facing_goal] = ddeg1

        # --- Determine speed:
        #   only move when expecting to face toward the goal after turn
        deg_aft_turn = d_src + ddeg
        dx_aft_turn, dy_aft_turn = deg2dxy(deg_aft_turn)

        will_face_goal = (
            dx.abs() + dy.abs()
        ) > (
            (x_src + dx_aft_turn * self.speed_meter_per_dt - x_dst).abs()
            + (y_src + dy_aft_turn * self.speed_meter_per_dt - y_dst).abs()
        )

        vel = npt.zeros(self.state.shape)
        vel[~same_loc & will_face_goal] = self.speed_meter_per_dt

        # DEF: ddeg_vel_states[batch, (ddeg, vel)]
        ddeg_vel_states = torch.stack([ddeg, vel], -1)

        # DEF: ddeg_vels[i_deg_vel, (ddeg, vel)]
        i_action = (
            ddeg_vel_states.T == self.ddeg_vels[..., None]
        ).all(-2).nonzero(as_tuple=False)[..., 0]

        d_next = d_src + ddeg_vel_states[:, 0]
        dx_next, dy_next = deg2dxy(d_next)
        x_next = x_src + dx_next * ddeg_vel_states[:, 1]
        y_next = y_src + dy_next * ddeg_vel_states[:, 1]
        if to_plot:
            # TODO: this doesn't work with a non-rectangular environment
            n_visit = self.n_visit.reshape(
                [self.statetab.nx_incl, self.statetab.ny_incl, self.statetab.n_heading])

            def plot_state(n_visit1, to_add_legend=False):
                def quiver(x, y, deg, color, va):
                    x, y, deg = npys(x[0], y[0], deg[0])
                    ax = plt.gca()  # type: plt.Axes
                    h = plt.plot(x, y, 'o', mfc='None', mec=color)
                    ax.annotate(
                        '',
                        xy=(x + np.cos(deg / 180. * np.pi) * self.speed_meter_per_dt,
                            y + np.sin(deg / 180. * np.pi) * self.speed_meter_per_dt),
                        xytext=(x, y),
                        arrowprops={'fc':color, 'ec': 'None', 'linewidth': 0},
                        bbox={'fc':'None', 'ec': 'None'})
                    plt.text(x, y, '%+1.2f,%+1.2f,%+4d' % (x, y, deg),
                             ha='left', va=va)
                    return h[0]
                
                plt.imshow(
                    npy(n_visit1).T,
                    extent=self.statetab.extent, zorder=-2, cmap='summer',
                    origin='lower'
                )
                self.statetab.env.plot_walls()
                plt.axis('equal')
                h_dst = quiver(x_dst, y_dst, d_dst, 'cyan', 'center')
                h_src = quiver(x_src, y_src, d_src, 'k', 'bottom')
                h_next = quiver(x_next, y_next, d_next, 'r', 'top')
                if to_add_legend:
                    plt.legend(
                        [h_src, h_next, h_dst], ['src', 'next', 'dst'],
                        **mplstyle.kw_legend_rightoutside
                    )
                    
            plt.close('all')
            axs = plt2.GridAxes(
                1, 2, widths=3, heights=3, left=0.5, right=1.5)
            plt.sca(axs[0, 0])
            plot_state(n_visit.min(-1)[0])
            plt.title('min')
            plt.sca(axs[0, 1])
            plot_state(n_visit.sum(-1), True)
            plt2.hide_ticklabels()
            plt.title('sum')
            plt.show()

            print('\n'
                'same loc: %d, facing goal: %d, opposite from goal: %d\n'
                '           x=%+1.3f,  y=%+1.3f,  deg=%+4d ->\n'
                '           x=%+1.3f,  y=%+1.3f,  deg=%+4d\n'
                '         (dx=%+1.3f, dy=%+1.3f; ddeg=%+4d, v=%+1.3f)\n'
                'aft turn: dx=%+1.3f, dy=%+1.3f,  deg=%+4d)\n'
                % (same_loc[0], facing_goal[0], opposite_from_goal[0],
                   x_src[0], y_src[0], d_src[0],
                   x_dst[0], y_dst[0], d_dst[0],
                   dx[0], dy[0], ddeg[0], npy(vel)[0],
                   np.sign(dx_aft_turn[0]), np.sign(dy_aft_turn[0]), deg_aft_turn[0],
                   ),
            )
            if (
                (self.state_prev is not None)
                and (self.state == self.state_prev).all()
            ):
                print('Same state!')
            if len(i_action) == 0:
                print('No action found!')
            print(
                '                 i_action:   %d (ddeg=%+4d, v=%+1.3f)'
                % (               i_action[0],
                   self.ddeg_vels[i_action[0], 0], self.ddeg_vels[i_action[0], 1])
            )
        return i_action  # NOW: debug i_action being []; perhaps make PolicyPellet subclass PolicyPath

    def get_dheading_deg(
            self, heading_deg_current: torch.Tensor,
            heading_deg_destination: torch.Tensor,
            to_turn: torch.BoolTensor = None
    ):
        """
        :param heading_deg_current: [batch_to_turn]
        :param heading_deg_destination: [batch_to_turn]
        :param to_turn: [batch] if not given, turn all
        :return: dheading_deg[batch]
        """
        if to_turn is None:
            # if not given, turn all
            to_turn = npt.ones(heading_deg_destination.shape, dtype=torch.bool)
        dheading_deg = npt.zeros(to_turn.shape)
        to_turn = to_turn & (
            ((heading_deg_current - heading_deg_destination) % 360.).round()
            != 0
        )
        if to_turn.any():
            diff_deg = torch.round(
                heading_deg_destination
                - heading_deg_current + 180.
            ) % 360. - 180.
            is_clockwise = (
                    (diff_deg < 0)
                    | (
                        (diff_deg == -180)
                        & npt.randint(high=1, size=[diff_deg.numel()],
                                        dtype=torch.bool)
                    ))
            dheading_deg[to_turn & is_clockwise] = -90.
            dheading_deg[to_turn & ~is_clockwise] = 90.
        return dheading_deg


class UNUSEDPolicyPong(UNUSED_PolicyPellet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{'name': 'pong', **kwargs})

        # n_destination[batch, loc]
        # = number of times the location is set as a destination
        self.n_destination: Union[torch.Tensor, None] = None

        # is_last_wall_loc[batch, loc]
        # = True if the location touches the last wall
        #   the destination was set from.
        self.is_last_wall_loc: Union[torch.Tensor, None] = None

        self.tactile_sensor = TactileSensorTabularBinary(
            self.statetab,
            self.statetab.dx, touch_reliability=1.,)

    @property
    def in_touch_state(self) -> torch.Tensor:
        """
        :return: in_touch_flat[state]
        """
        return (self.tactile_sensor.p_touch[:,:,1] > 0.5).any(1)

    @property
    def in_touch_state_side(self) -> torch.Tensor:
        """
        :return: in_touch_loc_heading_side[state, ego_side]
        """
        return (self.tactile_sensor.p_touch[..., 1] > 0.5).reshape([
            self.statetab.n_state_incl, self.statetab.n_heading
        ])

    @property
    def in_touch_loc_heading_side(self) -> torch.Tensor:
        """
        :return: in_touch_loc_heading_side[loc, heading, ego_side]
        """
        return (self.tactile_sensor.p_touch[..., 1] > 0.5).reshape([
            self.statetab.n_state_loc_incl, self.statetab.n_heading, self.statetab.n_heading
        ])

    @property
    def in_touch_loc_heading_back(self) -> torch.Tensor:
        I_HEADING_BACK = 2
        return (
            self.tactile_sensor.p_touch[..., I_HEADING_BACK, 1] > 0.5
        ).reshape([self.statetab.n_state_loc_incl, self.statetab.n_heading])

    @property
    def i_state_loc_heading(self) -> torch.Tensor:
        return npt.arange(self.statetab.n_state_incl).reshape(
            self.statetab.n_state_loc_incl, self.statetab.n_heading)

    def choose_destination(
            self,
            to_plot=False  # CHECKED
    ):
        assert self.state.ndim == 1
        n_batch = len(self.state)

        if self.state_destination is None:
            self.state_destination = -npt.ones(n_batch, dtype=torch.long)

            # self.n_destination[batch, loc]
            self.n_destination = npt.zeros(
                [n_batch, self.statetab.n_state_loc_incl])

            self.is_last_wall_loc = npt.zeros(
                [n_batch, self.statetab.n_state_loc_incl])
        else:
            assert self.n_destination.shape[0] == n_batch

        for i_batch in range(n_batch):
            state = self.state[i_batch]
            iloc = self.statetab.i_state2i_loc(state)
            ix, iy = self.statetab.i_state2ixy(state)
            iheading = self.statetab.iheading_incl[state]

            to_choose_destination = (self.state_destination[i_batch] == -1) or (
                not self.is_last_wall_loc[i_batch, iloc]
                and self.in_touch_state[state])
            if not to_choose_destination:
                continue

            # choose among states that would touch the side that is currently
            # not touching a wall
            is_side_touching_wall = self.in_touch_state_side[state, :]

            loc_same_side = self.in_touch_loc_heading_side[
                        :, iheading, is_side_touching_wall].any(1)
            loc_diff_side = self.in_touch_loc_heading_side[
                        :, iheading, ~is_side_touching_wall].any(1)

            self.is_last_wall_loc[i_batch, :] = loc_same_side

            candidate_destination_loc = loc_diff_side & ~loc_same_side
            assert candidate_destination_loc.any()

            n_destination = npt.dclone(self.n_destination[i_batch])
            # # noinspection PyTypeChecker
            # n_destination[iloc] = np.inf  # not necessary;
            #                               # going to other borders anyway

            # find a loc in one of the other 3 walls
            #  with the min loc visit count
            candidate_destination_loc_min = (
                candidate_destination_loc
                & (n_destination == n_destination[
                    candidate_destination_loc].min()))

            # Head away from the wall at the destination
            #  (find the state among the given loc where the back touches)
            assert self.statetab.n_heading == 4  # so that back = i_heading == 2
            i_states_candidate_w_back_toward_wall = self.i_state_loc_heading[
                candidate_destination_loc_min[:, None].expand(
                    self.statetab.n_state_loc_incl, self.statetab.n_heading)
                & self.in_touch_loc_heading_back
            ]
            # print(i_states_candidate_w_back_toward_wall)
            assert len(i_states_candidate_w_back_toward_wall) > 0

            state_destination = \
                i_states_candidate_w_back_toward_wall[
                    npt.randint(
                        len(i_states_candidate_w_back_toward_wall), ())]
            self.state_destination[i_batch] = state_destination

            i_loc_destination = self.statetab.i_state2i_loc(
                state_destination)
            self.n_destination[i_batch, i_loc_destination] = \
                self.n_destination[i_batch, i_loc_destination] + 1

            if to_plot:
                # CHECKED
                def imshow(v: torch.Tensor):
                    im = plt.imshow(npy(
                        v.reshape(self.statetab.nx_incl, self.statetab.ny_incl)).T,
                        origin='lower',
                    )
                    im.set_clim([0, max([1, max(v.flatten())])])

                axs = plt2.GridAxes(2, 4, top=1, hspace=1)
                plt.sca(axs[0, 0])
                imshow(loc_same_side)
                plt.title(f'touching\nsame side\n(heading:{iheading})')
                plt.plot(ix, iy, 'rx')

                plt.sca(axs[0, 1])
                imshow(loc_diff_side)
                plt.title(f'touching\ndiff side\n(heading:{iheading})')

                plt.sca(axs[0, 2])
                imshow(candidate_destination_loc)
                plt.title(f'candidate locs\n=~same side\n& diff side')

                plt.sca(axs[0, 3])
                imshow(self.n_visit[i_batch, :].reshape([
                    self.statetab.n_state_loc_incl, self.statetab.n_heading
                ]).sum(-1))
                plt.title(f'#visit')

                plt.sca(axs[1, 0])
                imshow(n_destination)
                plt.title(f'#dest.\n(bef)')

                plt.sca(axs[1, 1])
                imshow(candidate_destination_loc_min)
                plt.title(f'candidate locs\nw min #dest.')

                plt.sca(axs[1, 2])
                imshow(self.n_destination[i_batch])
                ixd, iyd = self.statetab.i_state2ixy(state_destination)
                ihd = self.statetab.iheading_incl[state_destination]
                plt.plot(ixd, iyd, 'o', mfc='None', mec='r')
                plt.title(f'dest &\n#dest. (aft)\ndst head:{ihd}')

                plt.sca(axs[1, 3])
                plt.gca().set_visible(False)

                plt.show()
                print('--')


class PolicyPongWestEast(UNUSEDPolicyPong):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{'name': 'pongWE', **kwargs})
        
    


def get_policy(policy_name='rnd_nodir', speed_meter_per_dt=0., **kwargs):
    if policy_name == 'straight':
        policy = PolicyDiscreteRandom(
            p_actions=npt.tensor([1.]),
            actions=[
                {'dheading_deg': 0., 'velocity_ego': speed_meter_per_dt}
            ],
            name=policy_name
        )
    elif policy_name == 'pellet':
        policy = UNUSED_PolicyPellet(**{
            'speed_meter_per_dt': speed_meter_per_dt, **kwargs})
    elif policy_name == 'ppellet':
        policy = PolicyPathPellet(
            **{
                'speed_meter_per_dt': speed_meter_per_dt, **kwargs
            }
        )
    elif policy_name == 'pong':
        policy = UNUSEDPolicyPong(**{
            'speed_meter_per_dt': speed_meter_per_dt, **kwargs})
    elif policy_name == 'rnd90_spd_dt':
        policy = PolicyDiscreteRandom(
            p_actions=npt.ones(4) / 4,
            actions=[
                {'dheading_deg': deg, 'velocity_ego': speed_meter_per_dt}
                for deg in [-90, 0, 90, 180]
            ],
            name=policy_name
        )
    elif policy_name == 'rnd_dir90_spd_dt':
        policy = PolicyDiscreteRandom(
            p_actions=npt.ones(3) / 3,
            actions=[
                {'dheading_deg': deg, 'velocity_ego': speed_meter_per_dt}
                for deg in [-90, 0, 90]
            ],
            name=policy_name
        )
    elif policy_name == 'rnd_dir_spd0':
        policy = PolicyDiscreteRandom(
            p_actions=npt.ones(8) / 8,
            actions=[
                {'dheading_deg': deg, 'velocity_ego': 0.}
                for deg in np.arange(-180., 180., 45.)
            ],
            name=policy_name
        )
    elif policy_name == 'rnd_dir45_spd0':
        policy = PolicyDiscreteRandom(
            p_actions=npt.ones(3) / 3,
            actions=[
                {'dheading_deg': deg, 'velocity_ego': 0.}
                for deg in [-45, 0, 45]
            ],
            name=policy_name
        )
    elif policy_name == 'rnd_dir90_spd0':
        policy = PolicyDiscreteRandom(
            p_actions=npt.ones(3) / 3,
            actions=[
                {'dheading_deg': deg, 'velocity_ego': 0.}
                for deg in [-90, 0, 90]
            ],
            name=policy_name
        )
    elif policy_name == 'fwd_dir90_spd5_rat':
        policy = PolicyDiscreteRandom(
            p_actions=npt.tensor([0.25, 0.5, 0.25]),
            actions=[
                {'dheading_deg': deg,
                 'velocity_ego': 5. * env_boundary.RAT_OVER_HUMAN_APPROX
                 }
                for deg in [-90, 0, 90]
            ],
            name=policy_name
        )
    elif policy_name == 'fwd_dir90_spd2_rat':
        policy = PolicyDiscreteRandom(
            p_actions=npt.tensor([0.3, 0.4, 0.3]),
            actions=[
                {'dheading_deg': deg,
                 'velocity_ego': 2. * env_boundary.RAT_OVER_HUMAN_APPROX
                 }
                for deg in [-90, 0, 90]
            ],
            name=policy_name
        )
    elif policy_name == 'fwd_dir90_spd1_rat':
        policy = PolicyDiscreteRandom(
            p_actions=npt.tensor([0.3, 0.4, 0.3]),
            actions=[
                {'dheading_deg': deg,
                 'velocity_ego': 1. * env_boundary.RAT_OVER_HUMAN_APPROX
                 }
                for deg in [-90, 0, 90]
            ],
            name=policy_name
        )
    elif policy_name == 'rnd_dir180_spd0':
        policy = PolicyDiscreteRandom(
            p_actions=npt.ones(2) / 2,
            actions=[
                {'dheading_deg': deg, 'velocity_ego': 0.}
                for deg in [0, 180]
            ],
            name=policy_name
        )
    elif policy_name == 'rnd_dir45_spd5':
        policy = PolicyDiscreteRandom(
            p_actions=npt.ones(3) / 3,
            actions=[
                {'dheading_deg': deg, 'velocity_ego': 5.}
                for deg in np.arange(-45., 90., 45.)
            ],
            name=policy_name
        )
    elif policy_name == 'rnd_dir45_spd5_rat':
        policy = PolicyDiscreteRandom(
            p_actions=npt.ones(3) / 3,
            actions=[
                {'dheading_deg': deg,
                 'velocity_ego': 5. * env_boundary.RAT_OVER_HUMAN_APPROX}
                for deg in np.arange(-45., 90., 45.)
            ],
            name=policy_name
        )
    elif policy_name == 'rnd_dir_spd5':
        policy = PolicyDiscreteRandom(
            p_actions=npt.tensor([0.1, 0.05, 0.05, 0.4, 0.2, 0.2]),
            actions=[
                {'dheading_deg': deg, 'velocity_ego': 5.}
                for deg in np.arange(-180., 180., 45.)
            ],
            name=policy_name
        )
    elif policy_name == 'rnd_fwd8_45deg2_spd0_5':
        policy = PolicyDiscreteRandom(
            p_actions=npt.tensor([0.1, 0.05, 0.05, 0.4, 0.2, 0.2]),
            actions=[
                {'dheading_deg': 0., 'velocity_ego': 0.},
                {'dheading_deg': 45., 'velocity_ego': 0.},
                {'dheading_deg': -45., 'velocity_ego': 0.},
                {'dheading_deg': 0., 'velocity_ego': 5.},
                {'dheading_deg': 45., 'velocity_ego': 5.},
                {'dheading_deg': -45., 'velocity_ego': 5.},
            ],
            name=policy_name
        )
    elif policy_name == 'rnd_nodir':
        policy = PolicyDiscreteRandom(
            p_actions=npt.tensor([1.]),
            actions=[
                {'dheading_deg': 0., 'velocity_ego`': 0.},
            ],
            name=policy_name
        )
    else:
        raise ValueError()
    return policy


class Retina(Recordable):
    keys = ['retinal_iamge']

    def __init__(
        self,
        fov_deg=(80., 80.),
        deg_per_pix=1.,
        n_channel=3,  # number of colors
        focal_length=.01,
    ):
        """
        :param fov_deg: [fov_deg_width, fov_deg_height]
        :param deg_per_pix:
        :param n_channel:
        :param focal_length (m):
        """
        super().__init__()

        # Retinal image
        self.fov_deg = np.array(fov_deg)
        self.deg_per_pix = deg_per_pix
        self.n_channel = n_channel
        self.ever_rendered = False
        self.focal_length = focal_length

        self.xs = [
            npt.arange(
                ap[0] + self.deg_per_pix / 2,
                ap[1] + self.deg_per_pix / 2,
                self.deg_per_pix
            ) for ap in self.aperture_deg
        ]
        # DEBUGGED: nx needs to be even (not odd) for
        #  mayavi.mlab.screenshot() not to leave a blank line at the end
        self.xs = [
            xs1[:-1] if len(xs1) % 2 != 0 else xs1
            for xs1 in self.xs
        ]

        self.nx = [xs1.numel() for xs1 in self.xs]
        self.image = npt.zeros(
            self.nx + [self.n_channel]
        )

        # Figure for rendering with mayavi
        self._fig = None
        # fun_render_init(fig) iscalled by fig() on first use
        self.fun_render_init = None
        self.fun_render_init_args = ()

    def get_dict_param(self) -> Dict[str, Union[str, float]]:
        """For saving parameters to a human-readable table"""
        return {
            'visual field width (deg)': self.fov_deg[0],
            'visual field height (deg)'
            'number of pixels along the width (pixel)': self.nx[0],
            'number of pixels along the height (pixel)': self.nx[1],
            'focal length (m)': self.focal_length,
        }

    @property
    def fig(self):
        if self._fig is None:
            # Only load mlab when needed
            from mayavi import mlab
            mlab.options.offscreen = True

            self._fig = mlab.figure(
                size=self.nx,
                bgcolor=None
            )
            # # CHECKED
            # # import traceback
            # # for line in traceback.format_stack():
            # #     print(line.strip())
            # print('Retina.__init__(): retina ID: %d, fig ID: %d'
            #       % (id(self), id(self.fig)))
            self._fig.scene.off_screen_rendering = True
            self.set_camera_aperture(self._fig)
            if self.fun_render_init is not None:
                self.fun_render_init(self._fig, *self.fun_render_init_args)
        return self._fig

    def set_camera_aperture(self, fig=None):
        if fig is None:
            fig = self.fig
        apertures = self.fov_deg
        if apertures[1] >= apertures[0]:
            fig.scene.camera.view_angle = apertures[1]
        else:
            fig.scene.camera.view_angle = np2.rad2deg(np.arctan(
                apertures[1] / apertures[0]
                * np.tan(np2.deg2rad(apertures[0]) / 2)
            )) * 2
            # # DEBUGGED: compute viewing angle (vertical) based on the larger
            # #    of the horz or vertical viewing angles
            # print(self.fig.scene.camera.view_angle)
            # print('--')

    def get_copy(self, **kwargs):
        kw = argsutil.kwdefault(
            kwargs,
            fov_deg=self.fov_deg,
            deg_per_pix=self.deg_per_pix,
            n_channel=self.n_channel,  # number of colors
        )
        return Retina(**kw)

    @property
    def aperture_deg(self):
        return (
            (-self.fov_deg[0] / 2, +self.fov_deg[0] / 2),
            (-self.fov_deg[1] / 2, +self.fov_deg[1] / 2),
        )

    def plot(
            self, image=None, hide_decoration=True,
            contrast=1.,
    ):
        if image is None:
            image = self.image
        image = npy(image)

        if image.shape[-1] == 1:
            image = image + np.zeros([1, 1, 3])

        if contrast != 1.:
            image = (image - 0.5) * contrast + 0.5

        image = np.clip(image, a_min=0., a_max=1.)

        im = plt.imshow(
            image.transpose([1, 0, 2]),
            origin='lower',
            cmap='gray',
            extent=[*np2.npys(
                self.xs[0][0] - self.deg_per_pix / 2,
                self.xs[0][-1] + self.deg_per_pix / 2,
                self.xs[1][0] - self.deg_per_pix / 2,
                self.xs[1][-1] + self.deg_per_pix / 2,
            )]
        )
        plt.xlim(npy(self.xs[0][[0, -1]])
                 + np.array([-.5, +.5]) * self.deg_per_pix)
        plt.ylim(npy(self.xs[1][[0, -1]])
                 + np.array([-.5, +.5]) * self.deg_per_pix)
        if hide_decoration:
            axhline = None
            axvline = None
            plt.xticks([])
            plt.yticks([])
        else:
            axhline = plt.axhline(0, linestyle=':', color=0.5 + np.zeros(3))
            axvline = plt.axvline(0, linestyle=':', color=0.5 + np.zeros(3))
            plt.xlabel(r'visual angle ($^\circ$)')
            plt.yticks(*plt.xticks())
            plt.gca().set_yticklabels([])

        return {
            'im': im,
            'axhline': axhline,
            'axvline': axvline
        }

    @staticmethod
    def view_fig(
        loc=None, heading_deg=None, fig=None,
        focal_length=None,
    ):
        if heading_deg is None:
            azimuth = None
        else:
            azimuth = (180. + heading_deg) % 360.

        # Only load mlab when needed
        from mayavi import mlab
        mlab.options.offscreen = True

        return mlab.view(
            azimuth=azimuth,
            elevation=90.,
            distance=focal_length,
            focalpoint=loc,
            figure=fig
        )

    def view(self, loc=None, heading_deg=None):
        self.set_camera_aperture()
        # self.fig.scene.camera.view_angle = (
        #     self.aperture_deg[1][1] - self.aperture_deg[1][0]
        # )
        return self.view_fig(
            loc=loc,
            heading_deg=heading_deg,
            fig=self.fig,
            focal_length=self.focal_length,
        )

    def render_image(self) -> np.ndarray:
        # Only load mlab when needed
        from mayavi import mlab
        mlab.options.offscreen = True

        # try:
        self.fig.scene.off_screen_rendering = True
        img = mlab.screenshot(self.fig, mode='rgb')
        self.ever_rendered = True
        if self.n_channel == 1:
            img = np.mean(img, -1, keepdims=True)

        # DEBUGGED: crop off random pixels
        #  - perhaps solved by forcing nx to be odd in __init__()
        # img = img[1:, :-1, :]

        # make it img[-y, +x, c] -> img[+x, +y, c]
        return np.flip(np.transpose(img, [1, 0, 2]), 1)
        # except:
            # # CHECKED
            # import sys, traceback
            # print(sys.exc_info()[1])
            # print(traceback.format_exc())
            # print('retina ID: %d, fig ID: %d' % (id(self), id(self.fig)))
            # print('--')

    def __del__(self):
        """Close mayavi figure to prevent memory leak"""
        try:
            if self._fig is not None:
                # Only load mlab when needed
                from mayavi import mlab
                mlab.options.offscreen = True

                mlab.close(self._fig)
        except TypeError as e:
            if (self.ever_rendered
                    or e.__str__() != 'Scene not attached to a mayavi engine.'):
                raise


def render_walls(
        fig,
        env: env_boundary.EnvBoundary,
        loc=(0., 0., 1.8), heading_deg=0.):
    """

    :param fig: mayavi.core.scene.Scene
    :param env:
    :param loc:
    :param heading_deg:
    :return:
    """
    n_corners = len(env.corners)
    zs = np.array([0., env.height_wall])

    # print('----- slam.render_walls() -----')
    # print('type(env): %s' % type(env))
    # print('-- env.contrast: %f, contrast_btw_walls: %f'
    #       % (env.contrast, env.contrast_btw_walls))
    # print('-- env.color_walls:')
    # print(env.color_walls)
    # print('-- env.color_background:')
    # print(env.color_background)
    # print('-- env.color_floor:')
    # print(env.color_floor)
    # print('-----')

    # Only load mlab when needed
    from mayavi import mlab
    mlab.options.offscreen = True

    for i in range(n_corners):
        i1 = (i + 1) % n_corners

        # CAVEAT: not sure if indexing='ij' is needed, but seems to have worked
        #   so far.
        x, z = np.meshgrid(
            env.corners[[i, i1], 0],
            zs,
        )
        y, _ = np.meshgrid(
            env.corners[[i, i1], 1],
            zs,
        )
        mesh1 = mlab.mesh(
            x, y, z,
            color=tuple(env.color_walls[i]),
            figure=fig
        )
        mesh1.actor.property.lighting = False

    fig.scene.background = tuple(
        env.color_background
    )

    mesh1 = mlab.mesh(
        np.stack([
            env.corners[[0, 1], 0],
            env.corners[[3, 2], 0]
        ]),
        np.stack([
            env.corners[[0, 1], 1],
            env.corners[[3, 2], 1]
        ]),
        np.stack([
            env.corners[[0, 1], 2],
            env.corners[[3, 2], 2]
        ]),
        color=tuple(env.color_floor),
        figure=fig
    )
    mesh1.actor.property.lighting = False

    # DEBUGGED: elevation=90 and azimuth=180 means looking along
    #  the +x axis direction with +y direction leftward and
    #  +z direction upward, as intended.
    Retina.view_fig(
        loc=npy(loc),
        heading_deg=npy(heading_deg),
        fig=fig
    )

    # mlab.view(
    #     azimuth=(npy(self.agent_state.heading_deg) + 180.) % 360.,
    #     elevation=90.,
    #     distance=.01,
    #     focalpoint=npy(self.agent_state.loc),
    #     figure=fig
    # )


class GenerativeModel(ykt.BoundedModule):
    def transition(self, control: Control) -> AgentState:
        raise NotImplementedError()

    def measure(self, state: AgentState) -> Measurement:
        raise NotImplementedError()

    def measure_retinal_image(
            self, state: AgentState, retina: Retina, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError()


class GenerativeModelSingleEnv(GenerativeModel):
    def __init__(
            self,
            env: env_boundary.EnvBoundary,
            radius_corner=1.,
            self_loc_xy0=(0., 0.),
            heading0=(1., 0.),
            velocity_ego0=5.,
            self_height0=None, # 1.8,  # Bellmund et al. 2019
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
            tactile_sensor: TactileSensorTabularBinary = None,
            dt=1.,
            **kwargs  # to ignore extraneous kwargs
    ):
        """

        Renamed from 'Agent' in slam_ori_speed_clamp,
        since this model is independent of the policy, which is assumed
        observed.

        :type env: env_boundary.EnvBoundary
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
                    npt.tensor(self_loc_xy0),
                    npt.tensor([self_height0])
                ], -1),
                velocity_ego=npt.tensor(velocity_ego0),
                heading=torch.cat([
                    npt.tensor(heading0),
                    npt.tensor([0.])
                ], -1)
            )
        self.agent_state = state  # type: AgentState

        # Transition & observation noise
        self.noise_heading = npt.tensor(noise_heading)
        self.noise_velocity_ego = npt.tensor(noise_velocity_ego)
        self.noise_tangential = npt.tensor(noise_control_shift)

        self.blur_retina = npt.tensor(blur_retina)  # in degree
        self.noise_pixel = npt.tensor(noise_pixel)
        self.noise_obs_dloc = npt.tensor(noise_obs_dloc)
        self.noise_obs_dheading = npt.tensor(noise_obs_dheading)
        self.noise_obs_dvelocity_ego = npt.tensor(noise_obs_dvelocity_ego)

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
            dloc=npt.zeros_like(self.agent_state.dloc),
            dheading=npt.zeros_like(self.agent_state.dheading),
            tactile_sensor=tactile_sensor,
            # dvelocity_ego=npt.zeros_like(self.agent_state.dvelocity_ego),
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
                c, cw = env.contrast, env._contrast_btw_walls
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

        rot_yx = npt.tensor(rot_yx, min_ndim=0)
        rot_zy = npt.tensor(rot_zy, min_ndim=0)
        rot_xz = npt.tensor(rot_xz, min_ndim=0)

        rot_zy, rot_xz, rot_yx = npt.expand_batch(
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

        if torch.all(self.noise_heading == 0.):
            noise_heading = npt.zeros(self.n_dim)
        else:
            noise_heading = torch.cat([npt.mvnrnd(
                npt.zeros(self.n_dim - 1),
                self.noise_heading * npt.eye(self.n_dim - 1)
            ), npt.tensor([0.])], 0)
        v = npt.tensor([self.inertia_heading, 0., 0.]) \
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
        noise_velocity_ego = npt.normrnd((0., 0., 0.), self.noise_velocity_ego,
                                         sample_shape=control.velocity_ego.shape[:-1])
        s.velocity_ego = (
            s.velocity_ego * self.inertia_velocity_ego + control.velocity_ego + noise_velocity_ego
        )
        s.dvelocity_ego = s.velocity_ego - velocity_ego_prev

        velocity_RtFwUp = npt.permute2en(
            # s.velocity_ego is originally FwRtUp
            npt.permute2st(s.velocity_ego)
            * npt.tensor([1., -1., 1.])
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
                                  [:2,:]).T):
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
        #         npt.zeros(self.n_dim - 1),
        #         self.noise_obs_dloc * npt.eye(self.n_dim - 1)
        #     ), npt.zeros(1) # Assume zero motion along height
        # ], -1)
        # meas.dloc = geom.h2e(geom.rotate_to_align(
        #     geom.e2h(state.dloc),
        #     geom.e2h(state.heading)
        # )) + noise_obs_dloc
        #
        # noise_obs_dheading = torch.cat([npt.mvnrnd(
        #     npt.zeros(self.n_dim - 1),
        #     self.noise_obs_dheading * npt.eye(self.n_dim - 1)
        # ), npt.zeros(1)], -1)
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
            corners = npt.tensor(self.env.corners)
            dcorners = corners.clone()
            dcorners[:, :2] = dcorners[:, :2] - state.loc[None, :2]
            dcorners = geom.h2e(geom.rotate_to_align(
                geom.e2h(dcorners),
                geom.e2h(state.heading)
            ))
            # Consider height
            dcorners[:, 2] = corners[:, 2] - state.loc[2]
            dcorners_deg = npt.rad2deg(geom.ori2rad(dcorners, n_dim=3))

            dist = torch.sqrt((dcorners ** 2).sum(1))
            # noinspection PyTypeChecker
            apparent_radius_deg = npt.rad2deg(
                torch.asin(npt.tensor(self.radius_corner / dist,
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

                retina.image[:,:,0] = retina.image[:,:,0] + torch.exp(
                        distrib.Normal(
                            npt.tensor(0.),
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
                    ) for img11 in np2.permute2st(img)
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

            retina.image = npt.tensor(img, dtype=torch.double)

            # # CHECKED
            # retina.plot()
            # plt.show()

        else:
            raise ValueError()

        return retina.image


class GenerativeModelSingleEnvTabular(GenerativeModelSingleEnv,
                                      StateTable):
    def __init__(
            self,
            env: Union[env_boundary.EnvBoundary, EnvTable],
            tactile_sensor: TactileSensorTabularBinary = None,
            **kwargs
    ):
        """

        :param env:
        :param kwargs:

        --- StateTable
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

        StateTable.__init__(self, env=env, **kwargs)
        GenerativeModelSingleEnv.__init__(
            self, env=env, tactile_sensor=tactile_sensor, **kwargs)

        self.cache_img_given_state: Cache = None
        self.dict_img_given_state = None

        # Lazy precomputation
        self._img_given_state = None
        self._log_img_given_state = None
        self._sum_img_given_state = None


    def get_dict_file_img_given_state(self) -> Dict[str, str]:
        return np2.shorten_dict({
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

        if self._img_given_state is not None and not to_force_render:
            i_state = self.get_i_state(loc=state.loc,
                                       heading_deg=state.heading_deg)
            image = self._img_given_state[i_state, :]

            if noise_pixel is not None and noise_pixel > 0:
                image = npt.tensor(
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
        image = self.img_given_state[i_state]
        if gain_retina is None:
            gain_retina = self.gain_retina
        if gain_retina is None or gain_retina == 0:
            pass
        else:
            image = distrib.poisson.Poisson(
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
        rate_retina = (distrib.Poisson(
            self.img_given_state[i_states] * self.dt * gain
        ).sample() / self.dt / gain)

        tactile_input = self.tactile_sensor.sense_vec(i_states)

        return rate_retina, tactile_input

    @property
    def img_given_state(self) -> torch.Tensor:
        """

        :return: img_given_state[state, x, y, c]
        """
        if self._img_given_state is None:
            # print('before computing img_given_state')  # CHECKED

            # Precompute spatial tuning curve of each pixel (img_given_state)
            # Note that noise is defined by the noise of the image.
            # Noise parameter is currently not defined inside this filter.
            # To include retinal noise, I'll have to give the renderer the
            # retinal noise parameter defined in this class.

            loaded = False
            cache = self.cache_img_given_state
            if cache is not None:
                try:
                    img_given_state = self.dict_img_given_state[
                        cache.fullpath]
                    loaded = True
                except KeyError:
                    try:
                        img_given_state = cache.getdict(['img_given_state'])[0]
                        assert img_given_state is not None
                        loaded = True
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

                # NOTE: Cache in gen as well and use in
                #   gen.measure_retinal_image()

                # except Exception as err:
                #     # CHECKED
                #     import sys, traceback
                #     print(sys.exc_info()[1])
                #     print(traceback.format_exc())
                #     raise err

                if cache is not None:
                    cache.set({'img_given_state': img_given_state})
                    cache.save()

            self._img_given_state = img_given_state

            # print('after computing img_given_state')  # CHECKED

            if (self.dict_img_given_state is not None
                and cache is not None
            ):
                self.dict_img_given_state[
                    cache.fullpath] = self._img_given_state
                cache.clear()  # to avoid memory leak

        # if self.gen.img_given_state is None:
        #     self.gen.img_given_state = self._img_given_state

        return self._img_given_state

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

    @img_given_state.setter
    def img_given_state(self, v: torch.Tensor):
        self._img_given_state = npt.tensor(v)

    @property
    def log_img_given_state(self):
        # # CHECKED
        # print('Observer2D.log_img_given_state(): observer ID: %d' % id(self))
        if self._log_img_given_state is None:
            self._log_img_given_state = torch.log(self.img_given_state)
        return self._log_img_given_state

    @property
    def sum_img_given_state(self):
        if self._sum_img_given_state is None:
            self._sum_img_given_state = torch.sum(
                self.img_given_state,
                dim=[v for v in range(1, self.img_given_state.ndim)]
            )
        return self._sum_img_given_state



GenSingleTab = GenerativeModelSingleEnvTabular


class Observer(ykt.BoundedModule):
    def set_prior(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def update(self, meas: Measurement, control: Control,
               **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def prediction_step(self, control: Control, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def measurement_step(self, meas: Measurement,
                         control: Control = None, **kwargs
                         ) -> torch.Tensor:
        raise NotImplementedError()

    def get_dict_file(self) -> Dict[str, str]:
        raise NotImplementedError()

    @staticmethod
    def merge_dict_file_true_bel(obs_bel: 'Observer', obs_tru: 'Observer'):
        dict_file1s = {
            k: obs.get_dict_file() for k, obs in [
                ('true', obs_tru),
                ('believed', obs_bel),
            ]
        }
        dict_file = {**dict_file1s['true']}
        for k, v in dict_file1s['believed'].items():
            if v != dict_file1s['true'][k]:
                dict_file[k[:-1] + k[-1].upper()] = v

        dict_file = {
            **dict_file,
            'eb': obs_bel.env.name, 'et': obs_tru.env.name}
        dict_file = np2.rmkeys(dict_file, ['ev', 'eV'])

        return dict_file


class Observer2D(Observer, StateTable):
    def __init__(
            self,
            gen: Union[type, GenerativeModelSingleEnvTabular],
            # NOTE: now copied from the generative model
            # x_max=None,
            # dx=2.5,
            # ddeg=45.,
            # to_use_inner=False,
            #
            # NOTE: consider delegating gen's
            # # noise_obs_shift=.1,
            # # noise_obs_rotate=.95,
            # noise_control_shift=(2.5, 2.5),

            noise_control_shift,  # =(0., 0.),
            noise_control_rotate_pconc,  # =.999,
            noise_control_shift_per_speed,  # =(0.1, 0.),
            noise_shift_kind='g',
            max_speed=5,
            max_noise_speed=5,
            requires_grad=False,
            duration_retina=1.,
            to_use_lapse=False,
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
        #     'Cannot compute visual likelihood - do not depcopy(gen) before ' \
        #     'feeding to Observer2D - construct separately instead!'

        Observer.__init__(self)
        StateTable.__init__(self, env=gen.env)

        # gen should be assigned after StateTable.__init__()
        # to allow calling set_state_table(gen)
        self.gen = gen  # type:GenerativeModelSingleEnvTabular
        # it's not removed by Module.__init__()
        self.set_state_table(self.gen)

        self.to_use_lapse = to_use_lapse

        self.gain_retina = duration_retina

        # Dense representation
        self.p_state0 = npt.zeros(
            self.nx, self.ny, self.n_heading
        )

        # Prior: uniform (sparse representation)
        self.p_state_incl = npt.ones(self.n_state_incl) / self.n_state_incl

        # # Self-motion noise parameters: not used for now
        # self.noise_obs_shift = nn.Parameter(npt.tensor([
        #     noise_obs_shift]), True)
        # self.noise_obs_rotate = nn.Parameter(npt.tensor([
        #     noise_obs_rotate]), True)

        # Control noise parameters
        self.noise_control_shift = ykt.BoundedParameter(
            npt.tensor(noise_control_shift).expand([2]),
            -1e-6, 40., requires_grad=requires_grad
        )

        self.noise_control_shift_per_speed = ykt.BoundedParameter(
            npt.tensor(noise_control_shift_per_speed).expand([2]),
            -1e-6, 40., requires_grad=requires_grad
        )

        self.noise_control_rotate_pconc = ykt.BoundedParameter(
            npt.tensor(noise_control_rotate_pconc),
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
                float(np2.npy(self.noise_control_shift_per_speed[0])),
        }

    def get_dict_file(self) -> Dict[str, str]:
        return np2.shorten_dict({
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
    def tactile_sensor(self) -> TactileSensorTabularBinary:
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

    @property
    def img_given_state(self) -> torch.Tensor:
        """
        :return: [state, x, y, color]
        """
        return self.gen.img_given_state

    @property
    def log_img_given_state(self) -> torch.Tensor:
        return self.gen.log_img_given_state

    @property
    def sum_img_given_state(self) -> torch.Tensor:
        return self.gen.sum_img_given_state

    @img_given_state.setter
    def img_given_state(self, v: torch.Tensor):
        self.gen.img_given_state = v

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
                  set_p_state=True
                  ) -> torch.Tensor:

        if i_state is not None:
            loc0, heading_deg0 = self.get_loc_heading_by_i_state(i_state)
            sigma0 = eps
            heading_pconc0 = .99

        p_xy = npt.sumto1(torch.exp(distrib.MultivariateNormal(
            loc=npt.tensor(loc0),
            covariance_matrix=npt.eye(2) * sigma0,
        ).log_prob(torch.stack([self.x0,
                                self.y0], -1))))
        # plt.imshow(npy(p_xy[:, :, 0]).T)
        # plt.show()

        p_ori = npt.vmpdf_prad_pconc(
            self.headings_deg / 360.,
            # DEBUGGED: convert deg to prad (0 to 1)
            npt.tensor([heading_deg0 / 360.]),
            npt.tensor([heading_pconc0]))
        # plt.plot(npy(self.headings_deg), npy(p_ori))
        # plt.show()

        p_state0 = p_xy * p_ori[None, None, :]
        p_state0[~self.state_incl] = 0.
        p_state0 = p_state0 / p_state0.sum()

        if set_p_state:
            self.p_state0 = p_state0
        return p_state0

    @staticmethod
    def get_p_s_true_given_s_belief(
            p_s_belief_given_s_true: np.ndarray, p_stationary: np.ndarray
    ) -> np.ndarray:
        return np2.sumto1(
            p_s_belief_given_s_true * p_stationary[:, None]
            , axis=1).T

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
            p_state0[self.state_incl] = npt.sumto1(
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
        i_state_true: torch.Tensor,
        control: torch.Tensor,
        **kwargs,
    ):
        """

        :param i_state_true: [batch]
        :param control: [batch, (forwrad, rightward, dheading)]
        :return:
        """
        batch_shape = i_state_true.shape

        p_state_incl = npt.zeros(
            batch_shape + self.p_state_incl.shape)
        p_state_incl[..., i_state_true] = 1.

        assert np.prod(list(batch_shape)) == 1, \
            'not implemented for batch_shape != [1] ' \
            'in prediction_step()'
        control1 = Control(
            dheading_deg=control[0, 2],
            velocity_ego=control[0, :2])

        # TODO: use params of generative model rather than the observation model
        p_state_incl = self.prediction_step(
            control=control1,
            p_state_incl=p_state_incl,
            return_p_state_incl=True,
            **kwargs)

        # sample transition from prediction
        i_state_true1 = distrib.Categorical(
            probs=p_state_incl).sample()

        # print(p_state_incl.shape)
        # print(i_state_true1.shape)
        # print('--')
        return i_state_true1[None]  # scalar for now


    def prediction_step(
        self, control: Control, p_state0: torch.Tensor = None,
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
    ) -> torch.Tensor:
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
        if p_state0 is None:
            p_state0 = self.p_state0
            if p_state_incl is not None:
                p_state0[self.state_incl] = p_state_incl

        if noise_control_rotate_pconc is None:
            noise_control_rotate_pconc = self.noise_control_rotate_pconc[0]
        else:
            noise_control_rotate_pconc = noise_control_rotate_pconc + \
                                         npt.zeros_like(
                                             self.noise_control_rotate_pconc[0])

        if noise_control_shift is None:
            noise_control_shift = self.noise_control_shift[:]
        else:
            noise_control_shift = noise_control_shift + npt.zeros_like(
                self.noise_control_shift[:])

        if noise_control_shift_per_speed is None:
            noise_control_shift_per_speed = \
                self.noise_control_shift_per_speed[:]
        else:
            noise_control_shift_per_speed = \
                noise_control_shift_per_speed + npt.zeros_like(
                    self.noise_control_shift_per_speed[:])

        # --- Predict rotation
        # DEBUGGED: rotation should happen before jumping in that direction
        p_heading1_given_heading0 = npt.vmpdf_a_given_b(
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
                    return np.flip(v.T, 0) # order of .T & .flip is important!
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
                    return np.flip(v, 0).T # order of .T & .flip is important!
                else:
                    raise ValueError()
            
            # p_state[x, y, heading]
            # print(p_state.shape)
            p_state0 = p_state.clone()
            for i_heading, (heading1, p0, incl) in enumerate(zip(
                self.headings_deg,
                npy(npt.permute2st(p_state0)),
                npy(npt.permute2st(self.state_incl))
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

                p_state[..., i_heading] = npt.tensor(p22.copy())

                # --- sanity check
                assert p22.shape == p0.shape
                assert np2.issimilar(p22.sum(), p0.sum(), 1e-9)

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
                    axs = plt2.GridAxes(1, 4)
                    plt.sca(axs[0, 0])
                    # plt.plot(p0.sum(1), '.-')
                    imshow1(p0)
                    plt.title(f'{np2.joinformat(p0.shape)}')
                    
                    plt.sca(axs[0, 1])
                    # plt.plot(p_dxy_given_heading.sum(1), '.-')
                    imshow1(p_dxy_given_heading)
                    plt.title(f'{np2.joinformat(p_dxy_given_heading.shape)}')
                    
                    plt.sca(axs[0, 2])
                    # plt.plot(p11.sum(1), '.-')
                    imshow1(p11)
                    plt.title(f'{np2.joinformat(p1.shape)}')
    
                    plt.sca(axs[0, 3])
                    # plt.plot(p21.sum(1), '.-')
                    imshow1(p22)
                    plt.title(f'{np2.joinformat(p22.shape)}')
                    
                    # plt2.sameaxes(axs)
                    plt.show()
    
                if to_plot:  # CHECKED
                    axs = plt2.GridAxes(
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
                    plt2.box_off()
    
                    plt.sca(axs[1, 1])
                    h0 = plt.plot(
                        npt.sumto1(p_state0[:, p_state0.shape[1] // 2, i_heading]),
                        color='b')
                    h1 = plt.plot(
                        npt.sumto1(p_state[:, p_state.shape[1] // 2, i_heading]),
                        color='r', ls='--')
                    plt.title('along x')
                    plt.xlabel('x')
                    plt.ylabel(f'p(x | y=middle,\nheading={heading1})')
                    plt.legend([h0[0], h1[0]], ['bef', 'aft'],
                               **mplstyle.kw_legend)
                    plt2.box_off()
    
                    plt.sca(axs[1, 2])
                    plt.plot(
                        npt.sumto1(p_state0[p_state0.shape[0] // 2, :, i_heading]),
                        color='b')
                    plt.plot(
                        npt.sumto1(p_state[p_state.shape[0] // 2, :, i_heading]),
                        color='r', ls='--')
                    plt.title('along y')
                    plt.ylabel(f'p(y | x=middle,\nheading={heading1})')
                    plt.xlabel('y')
                    plt2.box_off()
    
                    file = locfile.get_file_fig(
                        'pred', {
                            'h': npy(heading1),
                            **dict(dict_file_plot)
                        })
                    mkdir4file(file)
                    plt2.savefig(file, dpi=150)
                    print(f'Saved to {file}')
                    plt.close(axs.figure.number)

            # --- make sure no state that's not included has nonzero probability
            assert npt.issimilar(p_state0.sum(), p_state.sum(), 1e-9)
            assert not p_state0[~self.state_incl].any()
            assert not p_state[~self.state_incl].any()

        elif self.noise_shift_kind == 'n':
            ndxy = int(min([
                np.ceil((self.max_speed + self.max_noise_speed * 2) / self.dx),
                max([self.nx_incl, self.ny_incl])
            ]))
            dxs1 = npt.arange(-ndxy, ndxy + 1) * self.dx
            dxs, dys = torch.meshgrid([dxs1, dxs1])
    
            # mu_loc_next_given_heading[heading, xy] = mu|heading
            mu_loc_next_given_heading = (
                npt.permute2st(control.velocity_ego)[0]
                * prad2unitvec(self.headings_deg / 360.)
            ) + (
                npt.permute2st(control.velocity_ego)[1]
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
            p_dxy_given_heading = npt.sumto1(torch.exp(npt.mvnpdf_log(
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

        p_state = npt.nan2v(npt.sumto1(p_state, [0, 1], keepdim=True))
        p_state = p_state * p_angle0
        p_state = npt.sumto1(p_state)

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
        :param s_true: [batch, (x, y, heading)]
        :param control: [batch, (forward, rightward, dheading, speed)]
        :return: s_true[batch, (x, y, heading)]
        """
        # s_true0 = npt.dclone(s_true)

        raise DeprecationWarning(
            'Not passing validation - use transition() instead')

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
        variance = npt.pconc2var(noise_control_rotate_pconc1)
        # can skip wrapping when stdev is small
        assert (variance < 0.25 ** 2).all(), \
            'we currently skip wrapping, so you need to use a small variance' \
            '= high pconc!'

        dheading_deg = (
            self.headings_deg[None, :]
            - (th + dth)[:, None]
            + 180.
        ) % 360. - 180.  # ranges from -180 to +180

        p_th = torch.softmax(npt.log_normpdf(
            dheading_deg / 360., 0.,
            (npt.tensor(variance) * self.dt).sqrt()
        ), 1)
        # p_th = npt.vmpdf_prad_pconc(
        #     self.headings_deg[None, :] / 360.,
        #     (th / 360. + dth / 360.)[:, None],
        #     npt.tensor(noise_control_rotate_pconc1)
        # )
        assert not torch.isnan(p_th).any()
        i_th = npt.categrnd(p_th)
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

                if not self.n_heading in [1, 2, 4]:
                    raise NotImplementedError()
                speed = (dxy_allo ** 2).sum(-1).sqrt()

                incl = speed >= 1e-6
                dxy_allo[~incl] = 0.
                
                if incl.any():
                    p = self.noise_shift_gampdf(
                        noise_control_shift_per_speed1[0],
                        speed[incl])
    
                    idx = npt.categrnd(probs=p)
                    
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
                
                noise_dxy = npt.normrnd(
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
        s_true = npt.empty_like(s_true)
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
                i_states = npt.arange(self.n_state_incl)
                incl = (self.heading_deg == th1)
                i_state_dst = i_states[incl][torch.argmin(
                    (xy1[0] - self.x[incl]).abs()
                    + (xy1[1] - self.y[incl]).abs()
                )]
            else:
                dx = np.round(np.cos(np2.deg2rad(npy(th1))), 6)
                dy = np.round(np.sin(np2.deg2rad(npy(th1))), 6)
                
                if dx != 0 and dy == 0:
                    i_states = npt.arange(self.n_state_incl)
                    incl = (
                        (self.heading_deg == th1)
                        & npt.issimilar(self.y, xy1[1]))
                    i_state_dst = i_states[incl][torch.argmin(
                        (xy1[0] - self.x[incl]) * np.sign(dx))]
                elif dx == 0 and dy != 0:
                    i_states = npt.arange(self.n_state_incl)
                    incl = (
                        (self.heading_deg == th1)
                        & npt.issimilar(self.x, xy1[0]))
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
                        f'xy00: {np2.joinformat(xy00)} '
                        f'xy0:  {np2.joinformat(xy0)}, '
                        f'xy1:  {np2.joinformat(xy1)}'
                        f's_true: {s_true}')

                    i_batch = 0
                    th1 = th[0]
                    rec = self.gen.agent_state.get_record()

                    axs = plt2.GridAxes(1, 3, widths=2, heights=2)
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
                        rec=np2.dictlist2listdict(rec)[0])
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
            incl = npt.rand(s_true.shape[0]) < lapse_rate
            if incl.any():
                s_true[incl] = npt.randint(
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
        p = npt.zeros([n_batch, nx])
        nonzero_speed = speed > 1e-6
        if nonzero_speed.any():
            p[nonzero_speed] = torch.softmax(
                npt.gamma_logpdf_ms(
                    dx[None], speed[nonzero_speed],
                    speed[nonzero_speed] * noise_control_shift_per_speed1[0]
                ), -1)

            # plt.plot(npy(p.flatten()))
            # plt.show()
            # print('--')

        p[speed <= 1e-6, 0] = 1
        p[speed <= 1e-6, 1:] = 0
        p = npt.sumto1(p, -1)
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
        return npt.prod_sumto1(p_state, vislik, dim=1)

    def measurement_step_vec(
        self,
        i_state_beliefs: torch.Tensor,
        rate_retina: torch.Tensor,
        tactile_input: torch.Tensor,
        duration_times_gain=1.,
        use_vison=True,
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

        loglik = npt.zeros_like(visloglik)
        if use_vison:
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

    def resample_particle_filter(
            self, i_states: torch.Tensor, p_states) -> torch.Tensor:
        """

        :param i_states: [batch, particle]
        :param p_states: [batch, particle]
        :return: i_states[batch, particle] resampled in proportion to p_states
            within each batch
        """
        n_particle = i_states.shape[-1]
        return torch.stack([
            i_states1[distrib.Categorical(probs=p_states1).sample([n_particle])]
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

        :param p_state: [batch, state]
        :param control: [batch, (dx_ego, dy_ego, dth, speed)]
        :param xs:
        :param ys:
        :param headings_deg:
        :param noise_control_rotate_pconc1:
        :param noise_control_shift_per_speed1:
        :param noise_control_shift1:
        :param n_particle:
        :return:
        """
        raise DeprecationWarning('Not tested! Use prediction_step() instead.')

        # --- First sample states from p_state (= resampling step)
        n_batch, n_state = p_state.shape

        # DEF: states[particle, batch]
        states = npt.categrnd(p_state, sample_shape=[n_particle]
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
        # p_state = npt.zeros_like(p_state)[None, :].expand(
        #     [n_particle] + list(p_state.shape)
        # ).reshape([n_batch * n_particle, -1])
        # p_state = npt.zeros_like(p_state)[None, :].repeat(
        #     [n_particle, 1, 1]).reshape([n_batch * n_particle, -1])
        # NOTE: using zeros() instead of repeat().reshape() saved time
        p_state = npt.aggregate([
            np.tile(np.arange(n_batch)[None, :], [n_particle, 1]).flatten(),
            npy(states).flatten().astype(int),
        ], np.ones([n_batch * n_particle]), size=[n_batch, n_state]
        ) / n_particle
        # size_p_state = np.prod(list(p_state.shape))
        # p_state = npt.zeros([
        #     n_batch * n_particle, size_p_state // n_batch
        # ])
        # p_state.scatter_add_(1, states[:, None],
        #                      npt.ones([n_batch * n_particle, 1]))
        # p_state = p_state.reshape([n_particle, n_batch, -1]).sum(0) / n_particle
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
        :type meas: Measurement
        :type control: Control
        :type p_state_incl: torch.Tensor
        :return: p_state_incl (updated)
        :rtype: torch.Tensor
        """
        if duration is None:
            duration = self.gain_retina
        if p_state_incl is None:
            p_state_incl = self.p_state_incl

        if not skip_visual:
            loglik = self.get_loglik_retina_vectorized(
                meas.retinal_image[None, :],
                duration=duration,
            )[0]
            # loglik = self.get_loglik_retina(meas.retinal_image)
            log_p_state = torch.log(p_state_incl) + loglik
            log_p_state = log_p_state - torch.max(log_p_state)
            p_state_incl = torch.exp(log_p_state)
            p_state_incl = p_state_incl / torch.sum(p_state_incl)

        if update_state:
            self.p_state_incl = p_state_incl

        return p_state_incl

    def get_loglik_retina(self, rate_img, duration=None):
        """
        Assume independent Poisson firing rate.
        See Dayan & Abbott Eq. 3.30
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
        loglik = npt.zeros(self.n_state_incl)
        for s, tuning_img in enumerate(self.img_given_state):
            loglik[s] = (
                torch.sum(
                    rate_img * (torch.log(tuning_img) + np.log(
                        duration))
                ) - self.sum_img_given_state[s]
            ) * duration
        return loglik

    def get_loglik_retina_vectorized(
            self, rate_imgs, duration=None,
            i_states_belief: torch.Tensor = None
    ):
        """
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
                    * (self.log_img_given_state[None, :]
                       + np.log(duration)),
                    [-3, -2, -1]
                ) - self.sum_img_given_state[None, :]  # [1, state_belief]
            ) * duration
        elif i_states_belief.ndim == 1:
            return (
                torch.sum(
                    # [state_true_batch, x, y, chn]
                    rate_imgs
                    #
                    # [state_beliefs_to_consider, x, y, chn]
                    * (self.log_img_given_state[i_states_belief]
                       + np.log(duration)),
                    [-3, -2, -1]
                ) - self.sum_img_given_state[i_states_belief]  # [state_belief]
            ) * duration
        elif i_states_belief.ndim == 2:
            return (
                torch.sum(
                    # [state_true_batch, x, y, chn]
                    rate_imgs[:, None, :]
                    #
                    # [state_true_batch, state_beliefs_to_consider, x, y, chn]
                    * (npt.indexshape(self.log_img_given_state, i_states_belief)
                       + np.log(duration)),
                    [-3, -2, -1]
                ) - npt.indexshape(
                    self.sum_img_given_state, i_states_belief)
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
        loc_xy_maxlik = npt.tensor([
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


def plot_p_loc_tru_bel(
    obs_tru: Observer2D,
    start_tru_xyd: np.ndarray,
    goal_tru_xyd: np.ndarray,
    p_marker_tru: Union[torch.Tensor, np.ndarray],
    obs_bel: Observer2D,
    p_marker_bel: Union[torch.Tensor, np.ndarray]
) -> plt2.GridAxes:
    """

    :param obs_tru:
    :param start_tru_xyd:
    :param goal_tru_xyd: [waypoint, (x, y, d)]
    :param p_marker_tru:
    :param obs_bel:
    :param p_marker_bel:
    :return: axs[row, col]
    """
    if goal_tru_xyd.ndim == 1:
        goal_tru_xyd = goal_tru_xyd[None]

    axs = plt2.GridAxes(
        1, 2, widths=2, heights=2, left=0.1, right=1,
        top=1.
    )
    for i_obs, (title, obs, p_marker) in enumerate(
        [
            ('true', obs_tru, p_marker_tru),
            ('believed', obs_bel, p_marker_bel)
        ]):
        plt.sca(axs.flatten()[i_obs])
        obs.plot_p_loc(
            obs.p_state_incl2p_state_loc_incl(p_marker),
            cmap='gray_r')
        obs.env.plot_walls()
        plt.xlabel('Location (m)')
        plt2.box_off(['left', 'top', 'right'])
        plt.title(title, pad=25)

        if title == 'true' or obs_tru.env.name == obs_bel.env.name:
            plt.plot(
                goal_tru_xyd[:, 0], goal_tru_xyd[:, 1], 'k-', lw=0.5
            )
            h_goal = plt.plot(
                *goal_tru_xyd[-1, :2], 'o', mfc='None', mec='c',
                label='goal',
            )
            h_start = plt.plot(
                *start_tru_xyd[:2],
                '+', mfc='None', mec='c', label='start',
            )
            plt.legend(
                [h_start[0], h_goal[0]], ['start', 'goal'],
                ncol=2, **mplstyle.kw_legend_upperoutside)
        elif title == 'believed':
            plt2.colorbar()
    return axs


def plot_p_dist_tru_bel(
    obs_tru: Observer2D,
    obs_bel: Observer2D,
    xyd_root_tru: np.ndarray,
    p_xyd_root_bel: np.ndarray,
    dist_tru: float,
    dists_bel: np.ndarray,
    p_dist_bel: np.ndarray,
    xyd_estimate_tru: np.ndarray,
    p_xyd_estimate_bel: np.ndarray,
    axs: plt2.GridAxes = None,
) -> plt2.GridAxes:
    """

    :param obs_tru:
    :param obs_bel:
    :param start_tru_xyd_test: [xyd]
    :param p_xyd_root_bel: [i_state_incl]
    :param dist_tru: (scalar)
    :param dists_bel: [i_dist]
    :param p_dist_bel: [i_dist]
    :param xyd_estimate_tru: [xyd]
    :param p_xyd_estimate_bel: [i_state_incl]
    :param axs:
    :return:
    """

    assert type(obs_tru.env.env) == type(obs_bel.env.env), \
        'true and believed envs should be the same to overlay beliefs on ' \
        'true environment!'
    if axs is None:
        axs = plt2.GridAxes(
            1, 2, widths=[2, 1], heights=2,
            wspace=0.75, left=0.1, right=1.5, top=.5)

    plt.sca(axs[0, 0])
    plt.title('location')
    for xyd, p_xyd, color in [
        (xyd_root_tru, p_xyd_root_bel, 'b'),
        (xyd_estimate_tru, p_xyd_estimate_bel, 'r'),
    ]:
        obs_tru.env.plot_walls()
        obs_bel.plot_p_loc(npt.tensor(p_xyd), cmap=plt2.cmap_alpha(color))
        obs_tru.gen.agent_state.set_state(loc_xy=xyd[:2], heading_deg=xyd[2])
        obs_tru.gen.agent_state.quiver(color=color)
    plt2.box_off(['left', 'top', 'right'])
    plt.xlabel('$\ell_\mathrm{x}$ (m)')

    plt.sca(axs[0, 1])
    plt.title('distance')

    h_bar = plt.barh(
        y=dists_bel,
        width=p_dist_bel,
        height=dists_bel[1] - dists_bel[0] if len(dists_bel) >= 2 else 1.,
        color='lightgray'
    )
    plt.xlabel('$\mathrm{P}(\mathrm{distance})$')
    # df = pd.DataFrame({
    #     'distance (m)': dists_bel,
    #     'p': p_dist_bel
    # })
    # sns.kdeplot(
    #     data=df, y='distance (m)',
    #     color='tab:purple', fill=True, linestyle='None')
    # plt.xlabel('density')
    plt2.detach_axis('y')
    plt.ylabel('distance (m)')
    plt2.box_off()

    dist_resp = np.linalg.norm(xyd_estimate_tru[:2] - xyd_root_tru[:2])
    hs = [
        plt.axhline(y, color=color, ls=linestyle, lw=0.5)
        for y, color, linestyle in [
            (dist_tru, 'k', '-'),
            (0, 'b', '--'),
            (dist_resp, 'r', '--'),
        ]]
    plt.legend(
        [h_bar] + hs, [
            "bel. pair. dist.", "true pair. dist.",
            'estimation start', 'estimation end'
        ],
        **mplstyle.kw_legend_rightoutside
    )
    return axs
