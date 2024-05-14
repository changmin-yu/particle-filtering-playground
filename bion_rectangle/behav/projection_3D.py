"""
See example usage in ProjectiveGeometry3D.demo_centroid()
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from importlib import reload
import os
from matplotlib.animation import FuncAnimation

import torch
from torch import nn

from bion_rectangle.utils import plt2
from bion_rectangle.utils import numpytorch as npt
from bion_rectangle.utils import argsutil

from bion_rectangle.behav import kalman_filter as pred

# plt2.use_interactive()
mpl.rcParams['figure.dpi'] = 100
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')

torch.set_default_tensor_type(torch.DoubleTensor) # To prevent underflow

#%%
reload(npt)
reload(plt2)
reload(pred)

enforce_tensor = npt.enforce_tensor
p2st = npt.permute2st
p2en = npt.permute2en
v2m = npt.vec2matmul
m2v = npt.matmul2vec
npy = npt.npy
npys = npt.npys

seed = 0
torch.manual_seed(seed)

# NOTE: Glossary:
# self_euc: [batch, n_dim]: Euclidean coordinates
# self_hom: [batch, n_dim + 1]: Homogenous coordinates
# x_euc, x_hom [batch, n_dim], [batch_n_dim + 1]: Same for landmarks
# self_rad: [batch, n_dim]: Radians - rotation around x, y, z axes
# TODO: self_qtn: [batch, n_dim + 1]: Quaternion that expresses radians

# % Always 3D - 2D is just a special case where self & all landmarks' z
# coordinates are the same.
class ProjectiveGeometry3DSimple(nn.Module):
    """Pure geometry without parsing hidden vectors"""
    def __init__(self):
        super().__init__()
        self.n_dim = 3


    def ____UTILS____(self):
        pass

    @property
    def n_dim_hom(self):
        """:rtype: int"""
        return self.n_dim + 1

    @property
    def n_dim_ori(self):
        """:rtype: int"""
        return self.n_dim

    @property
    def n_dim_self(self):
        """:rtype: int"""
        return self.n_dim_hom + self.n_dim_ori

    @property
    def n_dim_ret_hom(self):
        """:rtype: int"""
        return self.n_dim

    @property
    def n_dim_ret_euc(self):
        """:rtype: int"""
        return self.n_dim - 1

    def rad2ori(self, rad):
        mr = self.get_mat_rotation(rad)
        # Unit vector toward +1 in the homogeneous coordinates
        # (hence 1 at the end)
        ori = m2v(
            mr @ v2m(npt.tensor(
                [1.] + [0.] * (self.n_dim - 1) + [1.]
            ))
        )
        return self.h2e(ori)

    def ori2rad(self, ori, n_dim=2):
        ori = npt.p2st(ori, 1)
        if n_dim == 2:
            rot_rad = torch.cat([v.unsqueeze(-1) for v in [
                npt.zeros_like(ori[0]),  # ignore
                npt.zeros_like(ori[0]),  # ignore
                torch.atan2(ori[1], ori[0])   # y / x or x->y
            ]], -1)
        elif n_dim == 3:
            rot_rad = torch.cat([v.unsqueeze(-1) for v in [
                npt.zeros_like(ori[0]),  # ignore
                torch.atan2(
                    ori[2],
                    torch.sqrt(ori[0] ** 2 + ori[1] ** 2)
                ),
                torch.atan2(ori[1], ori[0])   # y / x or x->y
            ]], -1)
            # [[rad_x, rad_y, rad_z]]
            # i.e., [[pitch, roll, yaw]]
            # For now, constrain to x->y rotation
        else:
            raise ValueError()
        return rot_rad

    def euclidean2homogeneous(self, x_euc):
        """
        :type x_euc: torch.Tensor
        :rtype: torch.Tensor
        """
        n_dim_euc = x_euc.dim()
        shape_rest = x_euc.shape[:-1]
        return torch.cat([x_euc,
                          npt.ones(shape_rest + torch.Size([1]))],
                         dim=n_dim_euc - 1)
    e2h = euclidean2homogeneous

    def homogeneous2euclidean(self, x_hom):
        """
        :type x_hom: torch.Tensor
        :rtype: torch.Tensor
        """
        n_dim_euc = x_hom.dim()
        shape_last = x_hom.shape[-1]
        return x_hom.index_select(n_dim_euc-1, npt.arange(shape_last - 1)) \
            / x_hom.index_select(n_dim_euc-1, npt.tensor([shape_last - 1]))
    h2e = homogeneous2euclidean


    # conversion between radian, orientation, and quaternion is more
    # complicated than this.
    # See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    # def rad2quaternion(self, rad):
    #     return self.e2h(self.rad2ori(rad))
    # r2q = rad2quaternion
    #
    # def quaternion2rad(self, qtn):
    #     return self.ori2rad(self.h2e(qtn))
    # q2r = quaternion2rad


    def get_mat_translation(self, vec_trans_euc):
        """
        :type vec_trans_euc: torch.Tensor
        :rtype: torch.Tensor
        """
        m = p2st(torch.diag_embed(npt.ones(vec_trans_euc.shape[:-1]
                                             + torch.Size([self.n_dim + 1]))),
                 2)
        try:
            m[:self.n_dim,-1,:] = p2st(vec_trans_euc)
        except:
            print(m[:self.n_dim,-1,:].shape)
            print(p2st(vec_trans_euc).shape)
            raise ValueError('--')
        return p2en(m, 2)


    def get_mat_rotation(self, rot_rad, n_dim=2):
        """
        :type rot_rad: torch.Tensor
        :type n_dim: int
        :rtype: torch.Tensor
        """
        if n_dim == 2:
            siz = rot_rad.shape[:-1]
            rot_rad = p2st(rot_rad).unsqueeze(0).unsqueeze(0)
            rot_z = rot_rad[:, :, 2]  # [1, 1, batch]
            cos_z, sin_z = torch.cos(rot_z), torch.sin(rot_z)
            zeros = npt.zeros_like(cos_z)  # [1, 1, batch]
            ones = npt.ones_like(cos_z)  # [1, 1, batch]
            m_z = torch.cat([
                torch.cat([cos_z, -sin_z, zeros], 1),
                torch.cat([sin_z, cos_z, zeros], 1),
                torch.cat([zeros, zeros, ones], 1),
            ], 0)
            m = npt.zeros(torch.Size([self.n_dim + 1] * 2) + siz)
            m[-1,-1] = 1.
            m[:3,:3] = m_z

            return p2en(m, 2)
        else:
            siz = rot_rad.shape[:-1]
            rot_rad = p2st(rot_rad).unsqueeze(0).unsqueeze(0)
            rot_x = rot_rad[:,:,0] # [1, 1, batch]
            cos_x, sin_x = torch.cos(rot_x), torch.sin(rot_x)
            rot_y = rot_rad[:,:,1] # [1, 1, batch]
            cos_y, sin_y = torch.cos(rot_y), torch.sin(rot_y)
            rot_z = rot_rad[:,:,2] # [1, 1, batch]
            cos_z, sin_z = torch.cos(rot_z), torch.sin(rot_z)
            zeros = npt.zeros_like(cos_x) # [1, 1, batch]
            ones = npt.ones_like(cos_x) # [1, 1, batch]
            m_z = torch.cat([
                torch.cat([cos_z, -sin_z, zeros], 1),
                torch.cat([sin_z, cos_z, zeros], 1),
                torch.cat([zeros, zeros, ones], 1),
            ], 0)
            m_x = torch.cat([
                torch.cat([ones, zeros, zeros], 1),
                torch.cat([zeros, cos_x, -sin_x], 1),
                torch.cat([zeros, sin_x, cos_x], 1),
            ], 0)
            m_y = torch.cat([
                torch.cat([cos_y, zeros, sin_y], 1),
                torch.cat([zeros, ones, zeros], 1),
                torch.cat([-sin_y, zeros, cos_y], 1),
            ], 0)

            # roll first (m_y), then pitch (m_x), then yaw (m_z).
            # when expressed matrix multiplication, the order is reversed.
            m = npt.zeros(torch.Size([self.n_dim + 1] * 2) + siz)
            m[-1,-1] = 1.
            m[:3,:3] = p2st(p2en(m_z, 2) @ p2en(m_x, 2) @ p2en(m_y, 2), 2)
            return p2en(m, 2)


    def rotate_to_align(self, src_hom, dst_hom, add_dst=False):
        """
        Return src - dst (default) or src + dst (if add_dst=True)
        in terms of azimuth & elevation
        :type src_hom: torch.Tensor
        :type dst_hom: torch.Tensor
        :param add_dst: if False (default), return src - dst; if True,
        return src + dst (in terms of azimuth & elevation)
        :rtype: torch.Tensor
        """
        rad = self.ori2rad(dst_hom)
        if add_dst:
            rot = self.get_mat_rotation(rad)
        else:
            rot = self.get_mat_rotation(-rad)
        return (rot @ src_hom.unsqueeze(-1)).squeeze(-1)


class ProjectiveGeometry3D(ProjectiveGeometry3DSimple):
    def __init__(self):
        """
        :param n_dim: Euclidean dimensionality of the world.
            dimensionality of the homogeneous coordinate is n_dim + 1.
            dimensionality of the retina is n_dim - 1.
        """
        super().__init__()

        n_dim = 3
        self.n_dim = n_dim
        self.default_self_rad = npt.tensor([
            [0., 0., 0.5]
        ]) * npt.pi
        self.default_self_euc = npt.zeros(1, n_dim)

        # always observe in 1 lower dimension
        # e.g., 3D world & 2D retina
        self.n_dim_ret = n_dim - 1

        self.dist_retina = 1.
        # Field of view - how far to go in each dimension
        # per depth?
        self.fov_tan = torch.tan(npt.tensor([80., 80.]) / 180. * np.pi)


    def x_hom2n_landmark(self, x_hom):
        return x_hom.shape[-1] // self.n_dim_hom

    def loc_hom2mat(self, loc_hom):
        """
        Make the last two dimensions [landmark, dim_hom]
        """
        return loc_hom.reshape(
            loc_hom.shape[:-1] + torch.Size([-1, self.n_dim_hom])
        )

    def loc_mat2ten(self, loc_mat, n_dim_per_obj=4):
        return npt.unblock_diag(loc_mat,
                                size_block=torch.Size([n_dim_per_obj] * 2))

    def ret_hom2mat(self, ret_hom):
        """
        Make the last two dimensions [landmark, dim_ret_hom]
        """
        return ret_hom.reshape(
            ret_hom.shape[:-1] + torch.Size([-1, self.n_dim_ret_hom])
        )


    def parse_vec_hidden(self, vec_hidden, stack_objs=False):
        """
        :param vec_hidden: as returned by cat_vec_hidden()
        :return: self_loc_hom, self_rad, landmark_hom
        """
        ix_self_loc = self.n_dim_hom
        ix_self_ori = self.n_dim_hom + self.n_dim_ori

        self_loc_hom = p2en(p2st(vec_hidden)[:ix_self_loc,:])
        self_rad = p2en(p2st(vec_hidden)[ix_self_loc:ix_self_ori,:])
        landmark_hom = p2en(p2st(vec_hidden)[self.n_dim_self:,:])

        if stack_objs:
            landmark_hom = self.loc_hom2mat(landmark_hom)
        return self_loc_hom, self_rad, landmark_hom

    def parse_mat_hidden(self, mat_hidden, stack_objs=False):
        """
        :param mat_hidden: covariance matrix of the latents
        :return: self_loc_hom, self_rad, landmark_hom
        """
        ix_self_loc = self.n_dim_hom
        ix_self_ori = self.n_dim_self

        self_loc_hom = p2en(p2st(mat_hidden, 2)
                            [:ix_self_loc, :ix_self_loc, :],
                            2)
        self_rad = p2en(p2st(mat_hidden, 2)
                        [ix_self_loc:ix_self_ori, ix_self_loc:ix_self_ori, :],
                        2)
        landmark_hom = p2en(p2st(mat_hidden, 2)
                            [ix_self_ori:, ix_self_ori:, :],
                            2)

        if stack_objs:
            landmark_hom = self.loc_mat2ten(landmark_hom, self.n_dim_hom)
        return self_loc_hom, self_rad, landmark_hom

    def cat_vec_hidden(self, s_loc_hom, s_rad, x_hom, *args, **kwargs):
        # Concatenate into a hidden state vector [batch, dim]
        return torch.cat(
            npt.expand_upto_dim([s_loc_hom, s_rad, x_hom], -1),
            dim=-1
        )

    def objdim2ix(self, obj, dim):
        # obj: -2 = self_loc; -1 = self_rad; 0+ = landmark
        if obj == -2:
            return dim
        elif obj == -1:
            return self.n_dim_hom + dim
        else:
            return self.n_dim_self \
                + self.n_dim_hom * obj + dim

    def set_cov(self, cov_orig, obj0, dim0, obj1, dim1, cov_new):
        ix0 = self.objdim2ix(obj0, dim0)
        ix1 = self.objdim2ix(obj1, dim1)
        cov = p2st(cov_orig, 2)
        cov[ix0, ix1] = cov_new
        cov[ix1, ix0] = cov_new
        return p2en(cov, 2)

    def get_cov(self, cov_orig, obj0, dim0, obj1, dim1):
        ix0 = self.objdim2ix(obj0, dim0)
        ix1 = self.objdim2ix(obj1, dim1)
        return p2st(cov_orig, 2)[ix0,ix1]

    def ____OBSERVE____(self):
        pass


    def get_mat_allo2top_hom(self, self_hom, self_rad, n_landmark=1):
        """
        translate -self_loc and rotate -self_ori.
        :type self_hom: torch.Tensor
        :type self_rad: torch.Tensor
        :rtype: torch.Tensor
        """
        m = self.get_mat_rotation(self.default_self_rad - self_rad) \
                @ self.get_mat_translation(-self.h2e(self_hom))
        return npt.block_diag(npt.repeat_dim(m.unsqueeze(-3), n_landmark, -3))

    def get_mat_allo2ret_hom(self, self_hom, self_rad, x_hom):
        batch_siz = npt.max_shape([
            x_hom.shape[:-1],
            self_rad.shape[:-1],
            self_hom.shape[:-1]
        ])
        n_dim_batch = len(batch_siz)

        # Get egocentric view
        n_landmark = self.x_hom2n_landmark(x_hom)
        obs_mat = self.get_mat_allo2top_hom(
            self_hom, self_rad, n_landmark=n_landmark
        )

        # Compute distance for each landmark
        x_obs = p2st(m2v(obs_mat @ v2m(x_hom)))
        dist = x_obs[1::self.n_dim_hom,:] # [landmark, batch]
        # dist[dist < 1.] = np.nan #TODO: deal with unseen landmarks -> done with retina instead

        # dist[landmark, 1, 1, batch] <- dist[landmark, batch]
        dist = dist.unsqueeze(1).unsqueeze(1)

        # The part being scaled with distance
        # ret_mat[landmark, 2, 4, batch]
        dst_mat0 = npt.tensor([[1., 0., 0., 0.],
                                 [0., 0., 1., 0.]])
        dst_mat = npt.attach_dim(dst_mat0, 1, n_dim_batch).repeat_interleave(
            n_landmark, dim=0
        ) / dist

        # The part that keeps the homogeneous dimension (last dim)
        # hom_mat[landmark, 1, 4, batch]
        hom_mat0 = npt.tensor([[0., 0., 0., 1.]])
        hom_mat = npt.attach_dim(
            hom_mat0, 1, n_dim_batch
        ).repeat_interleave(
            n_landmark, dim=0
        ).repeat(torch.Size([1] * 3) + batch_siz)

        # ret_mat[batch, 3 * n_landmark, 4 * n_landmark]
        # <- ret_mat[batch, landmark, 3, 4]
        ret_mat = npt.block_diag(
            npt.p2en(torch.cat([dst_mat, hom_mat], dim=1), 3)
        )

        # Compose egocentric view & projection
        return ret_mat @ obs_mat

    def observe_allo2top_hom(self, self_hom, self_rad, x_hom):
        """
        Return retinal coordinates given landmark locations in homogeneous
        coordinates.
        :param self_hom: [batch_shape, dims_homogeneous]
        :param self_rad: [batch_shape, dims_homogeneous] # not sure
        :param x_hom: [batch_shape, dims_homogeneous]
        :return: retinal_loc: [batch_shape, dims_retinal]
        """
        obs_mat = self.get_mat_allo2top_hom(self_hom, self_rad)
        return m2v(obs_mat @ v2m(x_hom))

    def observe_allo2top_euc(self, self_euc, self_rad, x_euc):
        res = self.observe_allo2top_hom(
            self.e2h(self_euc), self_rad, self.e2h(x_euc)
        )
        return self.h2e(res)

    def observe_allo2ret_hom(self, self_hom, self_rad, x_hom):
        m = self.get_mat_allo2ret_hom(self_hom, self_rad, x_hom)
        x_ret = m2v(m @ v2m(x_hom))
        return x_ret

    def forward(self, self_hom, self_rad, x_hom):
        """
        Use homogeneous coordinates to return retinal observation
        :param self_hom: [batch_shape, dims_homogeneous]
        :param self_rad: [batch_shape, dims_homogeneous]
        :param x_hom: [batch_shape, dims_homogeneous] landmark locations
        :return: [batch_shape, dims_retinal_homogeneous]
        """
        return self.observe_allo2ret_hom(self_hom, self_rad, x_hom)

    def ____PLOT____(self):
        pass

    def plot_self(self, self_euc=None, self_rad=None,
                  self_mu=None,
                  self_sig=None,
                  markerfacecolor='w',
                  to_plot_obs_separately=False,
                  dims=(0, 1),
                  quiver_scale=.75,
                  head_scale=None):
        """
        :param self_euc: [batch_shape, dims_euclidean]
        :param self_rad: [batch_shape, dims_euclidean]
        :param self_mu:
        :param self_sig:
        :param markerfacecolor:
        :param to_plot_obs_separately:
        :param dims:
        :param quiver_scale:
        :param head_scale:
        :return:
        """
        if self_euc is None:
            self_euc = self.default_self_euc
        if self_rad is None:
            self_rad = self.default_self_rad
        self_ori = self.rad2ori(self_rad)

        dims = np.array(dims)
        self_euc = npy(self_euc[0, dims[[0, 1]]])
        self_ori = npy(self_ori[0, dims[[0, 1]]])

        if self_mu is None:
            self_mu = self_euc
        else:
            self_mu = npy(self_mu[0, dims[[0, 1]]])

        if self_sig is not None:
            self_sig = npy(self_sig[0, dims[[0, 1]], :])
            self_sig = self_sig[:, dims[[0, 1]]]
            plt2.plot_centroid(self_mu, self_sig,
                               color=[0.7, 0.7, 0.7],
                               add_axis=False, zorder=-1)
            if to_plot_obs_separately:
                plt.plot(self_mu[0], self_mu[1], 'x',
                         # markersize=8,
                         color='k', zorder=-1, alpha=0.25)

        if quiver_scale > 0:
            if head_scale is None:
                head_scale = quiver_scale / .75
            plt.quiver(self_euc[0], self_euc[1],
                       self_ori[0], self_ori[1]) # ,
                       # scale=1./quiver_scale, scale_units='xy',
                       # headlength=10 * head_scale,
                       # headwidth=8 * head_scale,
                       # headaxislength=8 * head_scale)

        plt.plot(self_euc[0], self_euc[1], 'o',
                 markerfacecolor=markerfacecolor,
                 markeredgecolor='k')

    def plot_allo_ego_retina(self,
                             self_euc, self_rad,
                             landmark_orig,
                             self_mu=None, landmark_mu=None,
                             self_sig=None, landmark_sig=None,
                             color='b',
                             fill_marker=True,
                             # xlim=None, ylim=None,
                             xlim=(-5, 5), ylim=(-2, 8),
                             xlim_ego=None, ylim_ego=None,
                             xlim_ret=None, ylim_ret=None,
                             to_plot_obs_separately=False,
                             to_plot='allo'):
        # to_plot: 'allo', 'ego', 'retina', 'cov', or 'precision'

        if self_mu is None:
            self_mu = self.e2h(self_euc)
        if landmark_mu is None:
            landmark_mu = self.e2h(landmark_orig)
        if xlim_ego is None:
            xlim_ego = xlim
        if ylim_ego is None:
            ylim_ego = xlim_ego
        if xlim_ret is None:
            xlim_ret = xlim
        if ylim_ret is None:
            ylim_ret = xlim_ret

        def beautify_plot(ylim1=None, xlim1=None):
            plt.axis('equal')
            # plt.axis('square')
            if ylim1 is None:
                ylim1 = ylim
            if xlim1 is None:
                xlim1 = xlim
            plt.xlim(xlim1)
            plt.ylim(ylim1)
            plt.xticks([xlim1[0], 0, xlim1[1]])
            plt.yticks([ylim1[0], 0, ylim1[1]])


        # % Compute retinal projection
        x_ego = self.observe_allo2top_euc(self_euc, self_rad, landmark_orig)
        r_plot = npy(x_ego)

        # Egocentric, frontoparallel view (z vs x: "pinhole camera model")
        # https://en.wikipedia.org/wiki/Pinhole_camera_model
        # except there's no inversion, and FOV is assumed to be constrained
        # by the location of the image plane
        # (which is in *front* of the origin in this implementation.)
        x_ret = self.observe_allo2ret_hom(
            self.e2h(self_euc),
            self_rad,
            self.e2h(landmark_orig)
        )
        r_plot = npy(x_ret)
        hs = []

        if fill_marker == 1:
            markerfacecolor = color
            markeredgecolor = color
        elif fill_marker == -1:
            markerfacecolor = 'None'
            markeredgecolor = 'None'
        else:
            markerfacecolor = 'w'
            markeredgecolor = color

        # Plot
        if to_plot == 'allo':
            if landmark_sig is not None:
                for i_landmark in range(landmark_mu.shape[0]):
                    mu1 = landmark_mu[i_landmark, :2]
                    sig1 = landmark_sig[i_landmark, :2, :2]
                    plt2.plot_centroid(*npys(mu1, sig1),
                                       color=color,
                                       alpha=0.3,
                                       add_axis=False,
                                       zorder=-1)
                    if to_plot_obs_separately:
                        plt.plot(
                            *npys(mu1[0], mu1[1]), 'x',
                            color=color,
                            # markersize=8,
                            zorder=-1,
                            alpha=0.25
                        )

            l_plot = npy(landmark_orig)
            h1 = plt.plot(
                l_plot[:,0], l_plot[:,1], 'o',
                mfc=markerfacecolor,
                mec=markeredgecolor,
                zorder=0
            )
            hs.append(h1)

            beautify_plot(ylim)
            self.plot_self(self_euc, self_rad,
                           self_mu=self_mu, self_sig=self_sig,
                           to_plot_obs_separately=to_plot_obs_separately)
            plt.xlabel('$x$')
            plt.ylabel('$y$', rotation=0, va='center')
            plt.title('Allocentric')

        elif to_plot == 'ego':
            # Egocentric, top-down view (y vs x)
            h1 = plt.plot(
                r_plot[:,0], r_plot[:,1], 'o',
                mfc = markerfacecolor,
                mec = markeredgecolor
            )
            hs.append(h1)
            beautify_plot(ylim_ego, xlim_ego)

            # plt.plot(0, 0, 'ko')
            # self.plot_self()

            plt.axhline(0, linestyle=':', color=0.7 + np.zeros(3), zorder=2)
            plt.axvline(0, linestyle=':', color=0.7 + np.zeros(3), zorder=2)
            plt.title('Egocentric')
            plt2.hide_ticklabels('y')
            plt.xlabel('$x$')

        elif to_plot == 'retina':
            h1 = plt.plot(
                r_plot[:,0], r_plot[:,1], 'o',
                mfc = markerfacecolor,
                mec = markeredgecolor
            )
            hs.append(h1)
            self.plot_self(dims=(0,2))
            plt.axis('equal')
            # plt.axis('square')
            plt.plot()
            beautify_plot(ylim_ret, xlim_ret)
            plt.title('Retina')
            plt.axhline(0, linestyle=':', color=0.7 + np.zeros(3), zorder=-1)
            plt.axvline(0, linestyle=':', color=0.7 + np.zeros(3), zorder=-1)
            plt.xlabel('$x$')
            plt.ylabel('$z$', rotation=0, labelpad=10, va='center')
            plt2.hide_ticklabels('y')

        else:
            raise ValueError('%s is not supported!' % to_plot)

        return hs, x_ego

    def plot_state(self, mu, sigma,
                   obs=None,
                   plot_filled=None,
                   plot_obs=True,
                   colors=None,
                   **kwargs):
        mu_s_loc_hom, mu_s_rad, mu_x_hom = self.parse_vec_hidden(
            mu, stack_objs=True
        )
        sig_s_loc_hom, _, sig_x_hom = self.parse_mat_hidden(
            sigma, stack_objs=True
        )
        if obs is None:
            obs = mu
        obs_s_loc_hom, obs_s_rad, obs_x_hom = self.parse_vec_hidden(
            obs, stack_objs=True
        )

        # fig = plt.figure('centroids')
        # plt2.subfigureRC(4,4,3,1,fig)

        n_landmark = mu_x_hom.shape[-2]

        if colors is None:
            colors = ['r', 'b', 'g', 'y']

        if plot_filled is None:
            plot_filled = [True] * n_landmark

        for i_landmark in range(n_landmark):
            kw = argsutil.kwdefault(
                kwargs,
                self_sig=sig_s_loc_hom,
                landmark_sig=sig_x_hom[0, [i_landmark], :self.n_dim,
                             :self.n_dim],
                color=colors[i_landmark % len(colors)],
            )
            self.plot_allo_ego_retina(
                self_euc=self.h2e(obs_s_loc_hom[[0], :]),
                self_rad=mu_s_rad[[0], :],
                landmark_orig=self.h2e(obs_x_hom[0, [i_landmark], :]),
                self_mu=self.h2e(mu_s_loc_hom[[0], :]),
                landmark_mu=self.h2e(mu_x_hom[0, [i_landmark], :]),
                fill_marker=plot_filled[i_landmark],
                **kw
            )
        # plt.show()
        # print('---')

    def ____SSM_INTERFACE____(self):
        pass

    def hid2retinal(self, vec_hid, control=None):
        """
        Converts the prior mode (after prediction step) to the expected
        observation.
        Wrapper needed for get_jacobian.
        """
        # return self(*self.parse_vec_hidden(vec_hid))
        return self.observe_allo2ret_hom(*self.parse_vec_hidden(vec_hid))

    def get_ret_n_jac(self, vec_hid):
        """
        Currently only processes the first batch, due to restriction in
        npt.get_jacobian.
        :param vec_hid: [1, n_dim_hid]
        :return: x_ret_hom, jac
        x_ret_hom: [1, n_dim_ret]
        jac[1, n_dim_ret, n_dim_hid]
        """

        x_ret_hom = self.hid2retinal(vec_hid)
        n_out = x_ret_hom.shape[-1]
        jac = npt.get_jacobian(self.hid2retinal, vec_hid[[0], :], n_out)
        return x_ret_hom, jac

    def build_ssm(self, s_loc, s_rad, x_euc):
        pass
        # return ssm

    def ____DEMO____(self):
        pass

    def get_demo_spec(self, preset='loc'):
        if preset == 'loc':
            # x[landmark, dim] # All objects are on the floor (z=-3)
            z_floor = -3.
            x_euc = npt.tensor([
                [-3., 3., z_floor],
                [0., 3., z_floor],
                [+3., 3., z_floor],
                [-3., 6., z_floor],
                [0., 6., z_floor],
                [+3., 6., z_floor]
            ])

            # Looking from a constant height (z=0.)
            s_loc = npt.tensor([
                [0., 0., 1.],
                [0., 0.5, 1.],
                [0., 1., 1.],
                [0., 1., 1.],
                [0., 1., 1.],
                [0., 1., 1.],
                [0., 1., 1.],
                [-0.5, 1.5, 1.],
                [-1, 2., 1.],
            ]) + npt.tensor([[0., 0., 0.]])

            # [rad_x, rad_y, rad_z]
            s_rad = npt.tensor([
                [0., 0., 2. / 4.],
                [0., 0., 2. / 4.],
                [0., 0., 2. / 4.],
                [0., 0., 2.25 / 4.],
                [0., 0., 2.5 / 4.],
                [0., 0., 2.75 / 4.],
                [0., 0., 3. / 4.],
                [0., 0., 3. / 4.],
                [0., 0., 3. / 4.],
            ]) * npt.pi

        elif preset == 'ellipse_horz':
            x_euc, _, _ = self.get_demo_spec(preset='loc')
            n_frame = 5
            s_loc = npt.zeros(n_frame, 3)
            s_loc[:,0] = npt.linspace(-3., 3., n_frame)
            s_rad = npt.zeros(n_frame, 3)
            s_rad[:,-1] = 2./4. * npt.pi

        else:
            raise ValueError()

        return x_euc, s_loc, s_rad

    def demo_space(self):
        x_euc, s_loc, s_rad = self.get_demo_spec()

        # Basic tests
        x_hom = self.euclidean2homogeneous(x_euc)
        print(x_hom)
        x_euc = self.homogeneous2euclidean(x_hom)
        print(x_euc)

        i_obs = 1
        obs_mat = self.get_mat_allo2top_hom(
            self.e2h(s_loc[[i_obs],:]),
            s_rad[[i_obs], :],
        )
        print(obs_mat)

        x_ret = self.observe_allo2top_euc(
            s_loc[[i_obs], :], s_rad[[i_obs], :], x_euc
        )
        # x_ret = self.h2e(p2en(obs_mat @ p2st(self.e2h(x_euc[[0], :]))))
        print(x_ret)
        print(x_ret.shape)

        s_loc = torch.cat([s_loc, s_loc.flip(0)], 0)
        s_rad = torch.cat([s_rad, s_rad.flip(0)], 0)
        n_frame = s_loc.shape[0]

        pth = 'Data_SLAM/obs_projective'
        if not os.path.exists(pth):
            os.mkdir(pth)
        colors = ['b', 'r', 'g']
        fig_name = 'allo_ego_retina'
        fig = plt.figure(fig_name, figsize=(6, 2.3))
        plt2.subfigureRC(4,6,3,5, fig=fig)

        def plot_frame(frame):
            plt.clf()
            for i_landmark in range(x_euc.shape[0]):
                self.plot_allo_ego_retina(s_loc[[frame],:],
                                          s_rad[[frame],:],
                                          x_euc[[i_landmark], :],
                                          colors[i_landmark % len(colors)])
            plt.suptitle('Frame %d' % frame)

        fname = 'plt=anim+nd=3'
        file = os.path.join(pth, fname + '.gif')
        # noinspection PyTypeChecker
        anim = FuncAnimation(fig, plot_frame, frames=np.arange(n_frame),
                             interval=200)
        anim.save(file, dpi=150, writer='imagemagick')
        print('Saved to %s' % file)

    def demo_space_vectorized(self):
        # %
        x_euc, s_loc, s_rad = self.get_demo_spec()
        n_landmark = x_euc.shape[0]
        i_landmarks = npt.arange(n_landmark)
        frames = [0]
        x_hom = self.e2h(x_euc[i_landmarks,:])
        s_hom = self.e2h(s_loc[frames,:])
        s_rad1 = s_rad[frames,:]
        s_loc1 = s_loc[frames,:]

        obs_mat = self.get_mat_allo2top_hom(s_hom, s_rad)
        print(obs_mat)

        obs_mat1 = obs_mat[frames,:,:]

        hom2ret_mat = self.get_mat_allo2ret_hom(self.e2h(s_loc1), s_rad1, x_hom)
        print(hom2ret_mat)
        x_ret2 = m2v(hom2ret_mat @ v2m(x_hom))
        print(x_ret2)

        x_euc, s_loc, s_rad = self.get_demo_spec()

        # Basic tests
        x_hom = self.euclidean2homogeneous(x_euc)
        print(x_hom)
        x_euc = self.homogeneous2euclidean(x_hom)
        print(x_euc)

        i_obs = 1
        obs_mat = self.get_mat_allo2top_hom(self.e2h(s_loc[[i_obs],:]),
                                            s_rad[[i_obs], :])
        print(obs_mat)

        x_ret = self.observe_allo2top_euc(s_loc[[i_obs], :],
                                          s_rad[[i_obs], :],
                                          x_euc)
        print(x_ret)
        print(x_ret.shape)

        s_loc = torch.cat([s_loc, s_loc.flip(0)], 0)
        s_rad = torch.cat([s_rad, s_rad.flip(0)], 0)
        n_frame = s_loc.shape[0]

        pth = 'Data_SLAM/obs_projective'
        if not os.path.exists(pth):
            os.mkdir(pth)
        colors = ['b', 'r', 'g']
        fig_name = 'allo_ego_retina'
        fig = plt.figure(fig_name, figsize=(6, 2.3))
        plt2.subfigureRC(4,6,3,5, fig=fig)

        def plot_frame(frame):
            plt.clf()
            for i_landmark in range(x_euc.shape[0]):
                self.plot_allo_ego_retina(s_loc[[frame],:],
                                          s_rad[[frame],:],
                                          x_euc[[i_landmark], :],
                                          colors[i_landmark % len(colors)])
            plt.suptitle('Frame %d' % frame)

        fname = 'plt=anim_vec+nd=3'
        file = os.path.join(pth, fname + '.gif')
        # noinspection PyTypeChecker
        anim = FuncAnimation(fig, plot_frame, frames=np.arange(n_frame),
                             interval=200)
        anim.save(file, dpi=150, writer='imagemagick')
        print('Saved to %s' % file)

    def demo_centroid(self):
        #
        x_euc, s_loc, s_rad = self.get_demo_spec('ellipse_horz')
        # ssm = self.build_ssm(s_loc, s_rad, x_euc)

        n_landmark = x_euc.shape[0]
        i_landmarks = npt.arange(n_landmark) # npt.tensor([0, 1, 2]) #
        frame = npt.tensor([0])
        x_hom = self.e2h(x_euc[i_landmarks,:])
        s_hom = self.e2h(s_loc[frame,:])
        s_rad1 = s_rad[frame,:]

        # # %% Simulate the first frame - just observation
        vec_hid = self.cat_vec_hidden(s_loc_hom=s_hom,
                                      s_rad=s_rad1,
                                      x_hom=x_hom.view(1, -1))

        x_ret_hom, jac = self.get_ret_n_jac(vec_hid)

        n_dim_hidden = jac.shape[-1] # == vec_hid.shape[-1]
        n_dim_ret = jac.shape[-2]

        mu_prior = vec_hid
        sigma_prior = torch.diag(npt.tensor(
            [1e-4] * self.n_dim_self
            + [1., 1., 0.1, 1e-2] * n_landmark
        ))[None,:]

        ssm = pred.ExtendedKalmanFilter(
            ndim_hidden=n_dim_hidden,
            ndim_obs=n_dim_ret,
            ndim_control=n_dim_hidden,
            mu_prior=mu_prior,
            # setting to an arbitrary value to see if it is preserved
            # Also avoiding those behind retina for now
            # mu_prior=npt.zeros(n_dim_hidden),
            sigma_prior=sigma_prior,
            # setting to an arbitrary value to see if it is preserved
            meas_fun=self.hid2retinal,
            meas_jacobian=jac,
            meas_noise=torch.eye(n_dim_ret) * 0.1 ** 2,
            tran_noise=npt.zeros(n_dim_hidden, n_dim_hidden),
        ) # type: pred.ExtendedKalmanFilter
        ssm.update(obs=x_ret_hom)

        # #%% Plot results
        plt.imshow(jac, cmap='bwr', vmin=-2., vmax=2.)
        plt.colorbar()
        plt.show()

        self.plot_state(ssm.mu, ssm.sigma)
        plt.show()

        #
        return ssm

class Proj3DOri(ProjectiveGeometry3D):
    """
    ori_type:
    'rad': radians around x, y, z axes
    'ori': a homogeneous vector (1,0,0,1) after such ortations

    Uses the orientation vector (a unit vector in 3D homogeneous coordinates),
    rather than radians (as in ProjectiveGeometry3D), as the hidden
    representation for self orientation.
    This representation allows for diffusion in the orientation vector space,
    which should work fine as long as the orientation vector is not too close to
    the north or south pole, and as long as there is no 'roll',
    which should be a reasonable assumption for terrestrial animals.
    (For bats, this would indeed be a problem.
    For them, use quaternions instead of the orientation vector:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    Then we should also consider propagating uncertainty about quaternions:
    https://stats.stackexchange.com/questions/119780/what-does-the-covariance-of-a-quaternion-mean
    )
    """
    def __init__(self):
        super().__init__()

    @property
    def n_dim_ori(self):
        """:rtype: int"""
        return self.n_dim + 1

    def parse_vec_hidden(self, vec_hidden, stack_objs=False, ori_type='ori'):
        """
        :param vec_hidden:
        :param stack_objs:
        :param ori_type: 'rad': radians around x, y, z axes; 'ori': a
        homogeneous vector (1,0,0,1) after such ortations
        :return: (self_loc_hom, self_rad_or_ori_hom, landmark_hom)
        """
        self_loc_hom, self_ori_hom, landmark_hom = super().parse_vec_hidden(
            vec_hidden, stack_objs=stack_objs
        )
        if ori_type == 'rad':
            return self_loc_hom, self.ori2rad(self.h2e(self_ori_hom)), \
                   landmark_hom
        elif ori_type == 'ori':
            return self_loc_hom, self_ori_hom, landmark_hom
        else:
            raise ValueError()

    def parse_mat_hidden(self, mat_hidden, stack_objs=False, ori_type='ori'):
        """
        :param mat_hidden: covariance matrix of the latents
        :return: self_loc_hom, self_rad, landmark_hom
        """
        if ori_type == 'rad':
            raise NotImplementedError()
        elif ori_type == 'ori':
            return super().parse_mat_hidden(mat_hidden, stack_objs=stack_objs)
        else:
            raise ValueError()

    def cat_vec_hidden(self, s_loc_hom, s_ori_hom, x_hom, *args,
                       ori_type='ori', **kwargs):
        # Concatenate into a hidden state vector [batch, dim]
        if ori_type == 'rad':
            s_ori_hom = self.e2h(self.rad2ori(s_ori_hom))
        elif ori_type == 'ori':
            s_ori_hom = s_ori_hom
        else:
            raise ValueError()

        return torch.cat(
            npt.expand_upto_dim([s_loc_hom, s_ori_hom, x_hom], -1),
            dim=-1
        )

    def ____OBSERVE___(self):
        pass

    def get_mat_ori2y(self, self_ori_hom, n_dim=None):
        """
        :param self_ori_hom: orientation vector in the homogeneous coordinate.
        z is assumed to be zero for now.

        :type rot_rad: torch.Tensor
        :type n_dim: int
        :rtype: torch.Tensor
        """

        siz = self_ori_hom.shape[:-1]
        self_ori = p2st(self_ori_hom)[:, None, :]

        len_z = (self_ori[0:2,:] ** 2).sum(0, keepdim=True).sqrt()
        cos_z = self_ori[[0],:] / len_z
        sin_z = self_ori[[1],:] / len_z

        # th_z_rot = (pi/2 - th_z)
        # Change cos and sin accordingly
        cos_z_rot = sin_z
        sin_z_rot = cos_z

        zeros = npt.zeros_like(cos_z) # [1, 1, batch]
        ones = npt.ones_like(cos_z) # [1, 1, batch]
        m_z = torch.cat([
            torch.cat([cos_z_rot, -sin_z_rot, zeros], 1),
            torch.cat([sin_z_rot, cos_z_rot, zeros], 1),
            torch.cat([zeros, zeros, ones], 1),
        ], 0)
        m = npt.zeros(torch.Size([self.n_dim + 1] * 2) + siz)
        m[-1,-1,:] = 1.
        m[:3,:3,:] = m_z
        return p2en(m, 2)

    def get_mat_allo2top_hom(self, self_hom, self_ori,
                             n_landmark=1):
        """
        translate -self_loc and rotate -self_ori.
        After transformation, self_loc becomes (0,0,0) and self_ori becomes
        (0,1,0) in the orientation vector
        :type self_hom: torch.Tensor
        :type self_rad: torch.Tensor
        :rtype: torch.Tensor
        """

        m = self.get_mat_ori2y(self_ori) \
                @ self.get_mat_translation(-self.h2e(self_hom))
        return npt.block_diag(npt.repeat_dim(m.unsqueeze(-3), n_landmark, -3))

    def get_mat_allo2ret_hom(self, self_hom, self_ori, x_hom,
                             ori_type='ori'):
        """

        :param self_hom:
        :param self_ori:
        :param x_hom:
        :param ori_type:
        :return: (mat_allo2ret_hom, dist)
        mat_allo2ret_hom[batch, ...]
        dist[batch, landmark]
        """

        batch_siz = npt.max_shape([
            x_hom.shape[:-1],
            self_ori.shape[:-1],
            self_hom.shape[:-1]
        ])
        n_dim_batch = len(batch_siz)

        # Get egocentric view
        n_landmark = self.x_hom2n_landmark(x_hom)
        obs_mat = self.get_mat_allo2top_hom(
            self_hom, self_ori, n_landmark=n_landmark
        )

        # Compute distance for each landmark
        x_obs = p2st(m2v(obs_mat @ v2m(x_hom)))
        dist = x_obs[1::self.n_dim_hom,:] # [landmark, batch]

        if torch.any(torch.isnan(dist)):
            print('dist is NaN!')
            print('---')

        # dist[landmark, 1, 1, batch] <- dist[landmark, batch]
        dist = dist.unsqueeze(1).unsqueeze(1)

        # The part being scaled with distance
        # ret_mat[landmark, 2, 4, batch]
        dst_mat0 = npt.tensor([[1., 0., 0., 0.],
                                 [0., 0., 1., 0.]])
        dst_mat = npt.attach_dim(dst_mat0, 1, n_dim_batch).repeat_interleave(
            n_landmark, dim=0
        ) / dist

        # The part that keeps the homogeneous dimension (last dim)
        # hom_mat[landmark, 1, 4, batch]
        hom_mat0 = npt.tensor([[0., 0., 0., 1.]])
        hom_mat = npt.attach_dim(
            hom_mat0, 1, n_dim_batch
        ).repeat_interleave(
            n_landmark, dim=0
        ).repeat(torch.Size([1] * 3) + batch_siz)

        # ret_mat[batch, 3 * n_landmark, 4 * n_landmark]
        # <- ret_mat[batch, landmark, 3, 4]
        ret_mat = npt.block_diag(
            npt.p2en(torch.cat([dst_mat, hom_mat], dim=1), 3)
        )

        # Compose egocentric view & projection
        return ret_mat @ obs_mat, dist.squeeze(1).squeeze(1).t()

    def observe_allo2ret_hom(self, self_hom, self_ori, x_hom,
                             return_dist=False):
        m, dist = self.get_mat_allo2ret_hom(self_hom, self_ori, x_hom)
        x_ret = m2v(m @ v2m(x_hom))
        if return_dist:
            return x_ret, dist
        else:
            return x_ret

    def hid2retinal(self, vec_hid, control=None, return_dist=False):
        """
        Converts the prior mode (after prediction step) to the expected
        observation.
        Wrapper needed for get_jacobian.
        """
        if control is not None:
            NotImplementedError()
        return self.observe_allo2ret_hom(
            *self.parse_vec_hidden(vec_hid, ori_type='ori'),
            return_dist=return_dist
        )

    get_ret = hid2retinal

    def get_meas_jac(self, vec_hid, x_ret_hom):
        n_out = x_ret_hom.shape[-1]

        jac = npt.get_jacobian(self.hid2retinal, vec_hid[[0], :], n_out)
        return jac

    def is_seen(self, x_ret_hom, dist):
        is_far_enough = dist[0, :] > self.dist_retina
        n_landmark = x_ret_hom.shape[-1] // (self.n_dim_ret + 1)
        x_ret_mat = x_ret_hom.reshape(
            [n_landmark, (self.n_dim_ret + 1)]
        )[:, :2]
        tan_fov = x_ret_mat / self.dist_retina
        # tan_fov = x_ret_mat / dist[0,:][:,None]
        is_within_fov = torch.all(
            torch.abs(tan_fov) < self.fov_tan[None,:],
            dim=1
        )
        # print(dist)
        # print(tan_fov.t())
        # print(is_within_fov[None,:])
        # print(is_far_enough[None,:])
        # print('--')
        return is_far_enough & is_within_fov

    def get_ret_n_jac(self, vec_hid, meas_noise=None):
        """
        Currently only processes the first batch, due to restriction in
        npt.get_jacobian.
        :param vec_hid: [1, n_dim_hid]
        :return: x_ret_hom, jac
        x_ret_hom: [1, n_dim_ret]
        jac[n_dim_ret, n_dim_hid]
        """

        # dist[batch, landmark]
        x_ret_hom, dist = self.hid2retinal(vec_hid, return_dist=True)

        if meas_noise is not None:
            x_ret_hom += meas_noise

        jac = self.get_meas_jac(vec_hid, x_ret_hom)

        if torch.any(torch.isnan(jac)):
            print('jac is NaN!')
            print('--')

        not_seen = ~self.is_seen(x_ret_hom, dist)

        n_dim_hid = jac.shape[-1]
        hid_not_seen = npt.zeros(n_dim_hid, dtype=torch.bool)
        hid_not_seen[self.n_dim_self:] = not_seen.repeat_interleave(
            self.n_dim_hom
        )
        jac[:, hid_not_seen] = 0.
        return x_ret_hom, jac

    def ____PLOT____(self):
        pass

    def plot_self(self, self_euc=None, self_ori=None,
                  self_mu=None,
                  self_sig=None,
                  markerfacecolor='w',
                  to_plot_obs_separately=False,
                  dims=(0, 1),
                  quiver_scale=.75,
                  width=None,
                  head_scale=None):
        """
        :param self_euc: [batch_shape, dims_euclidean]
        :param self_rad: [batch_shape, dims_euclidean]
        :param self_mu:
        :param self_sig:
        :param markerfacecolor:
        :param to_plot_obs_separately:
        :param dims:
        :param quiver_scale:
        :param head_scale:
        :return:
        """
        if self_euc is None:
            self_euc = self.default_self_euc
        if self_ori is None:
            self_rad = self.default_self_rad
            self_ori = self.rad2ori(self_rad)

        dims = np.array(dims)
        self_euc = npy(self_euc[0, dims[[0, 1]]])
        self_ori = npy(self_ori[0, dims[[0, 1]]])

        if self_mu is None:
            self_mu = self_euc
        else:
            self_mu = npy(self_mu[0, dims[[0, 1]]])

        if self_sig is not None:
            self_sig = npy(self_sig[0, dims[[0, 1]], :])
            self_sig = self_sig[:, dims[[0, 1]]]
            plt2.plot_centroid(self_mu, self_sig,
                               color=[0.7, 0.7, 0.7],
                               add_axis=False, zorder=-1)
            if to_plot_obs_separately:
                plt.plot(self_mu[0], self_mu[1], 'x',
                         # markersize=8,
                         color='k', zorder=-1, alpha=0.25)

        if quiver_scale > 0:
            if head_scale is None:
                head_scale = quiver_scale / .75
            h_quiver = plt.quiver(
                self_euc[0], self_euc[1],
                self_ori[0], self_ori[1],
                # scale=1./quiver_scale, scale_units='height',
                # width=width,
                # headwidth=4,
                # headlength=6 * width,
                # headwidth=8 * head_scale,
                # headaxislength=4 * width
            )
        else:
            h_quiver = None

        h = plt.plot(self_euc[0], self_euc[1], 'o',
                 markerfacecolor=markerfacecolor,
                 markeredgecolor='k')
        return h, h_quiver

    def plot_allo_ego_retina(self,
                             self_euc, self_ori,
                             landmark_orig,
                             self_mu=None, landmark_mu=None,
                             self_sig=None, landmark_sig=None,
                             color='b',
                             fill_marker=True,
                             # xlim=None, ylim=None,
                             xlim=(-5, 5), ylim=(-2, 8),
                             xlim_ego=None, ylim_ego=None,
                             xlim_ret=None, ylim_ret=None,
                             to_plot_obs_separately=False,
                             to_plot='allo'):
        # to_plot: 'allo', 'ego', 'retina', 'cov', or 'precision'

        if self_mu is None:
            self_mu = self.e2h(self_euc)
        if landmark_mu is None:
            landmark_mu = self.e2h(landmark_orig)
        if xlim_ego is None:
            xlim_ego = xlim
        if ylim_ego is None:
            ylim_ego = xlim_ego
        if xlim_ret is None:
            xlim_ret = xlim
        if ylim_ret is None:
            ylim_ret = xlim_ret

        def beautify_plot(ylim1=None, xlim1=None):
            plt.axis('equal')
            # plt.axis('square')
            if ylim1 is None:
                ylim1 = ylim
            if xlim1 is None:
                xlim1 = xlim
            plt.xlim(xlim1)
            plt.ylim(ylim1)
            plt.xticks([xlim1[0], 0, xlim1[1]])
            plt.yticks([ylim1[0], 0, ylim1[1]])


        # % Compute retinal projection
        x_ego = self.observe_allo2top_euc(self_euc, self_ori, landmark_orig)
        r_plot = npy(x_ego)

        # Egocentric, frontoparallel view (z vs x: "pinhole camera model")
        # https://en.wikipedia.org/wiki/Pinhole_camera_model
        # except there's no inversion, and FOV is assumed to be constrained
        # by the location of the image plane
        # (which is in *front* of the origin in this implementation.)
        x_ret = self.observe_allo2ret_hom(
            self.e2h(self_euc),
            self_ori,
            self.e2h(landmark_orig)
        )
        r_plot = npy(x_ret)
        hs = []

        if fill_marker == 1:
            markerfacecolor = color
            markeredgecolor = color
        elif fill_marker == -1:
            markerfacecolor = 'None'
            markeredgecolor = 'None'
        else:
            markerfacecolor = 'w'
            markeredgecolor = color

        # Plot
        if to_plot == 'allo':
            if landmark_sig is not None:
                for i_landmark in range(landmark_mu.shape[0]):
                    mu1 = landmark_mu[i_landmark, :2]
                    sig1 = landmark_sig[i_landmark, :2, :2]
                    plt2.plot_centroid(*npys(mu1, sig1),
                                       color=color,
                                       alpha=0.3,
                                       add_axis=False,
                                       zorder=-1)
                    if to_plot_obs_separately:
                        plt.plot(
                            *npys(mu1[0], mu1[1]), 'x',
                            color=color,
                            # markersize=8,
                            zorder=-1,
                            alpha=0.25
                        )

            l_plot = npy(landmark_orig)
            h1 = plt.plot(
                l_plot[:,0], l_plot[:,1], 'o',
                mfc=markerfacecolor,
                mec=markeredgecolor,
                zorder=0
            )
            hs.append(h1)

            beautify_plot(ylim)
            self.plot_self(self_euc, self_ori,
                           self_mu=self_mu, self_sig=self_sig,
                           to_plot_obs_separately=to_plot_obs_separately)
            plt.xlabel('$x$')
            plt.ylabel('$y$', rotation=0, va='center')
            plt.title('Allocentric')

        elif to_plot == 'ego':
            # Egocentric, top-down view (y vs x)
            h1 = plt.plot(
                r_plot[:,0], r_plot[:,1], 'o',
                mfc = markerfacecolor,
                mec = markeredgecolor
            )
            hs.append(h1)
            beautify_plot(ylim_ego, xlim_ego)

            # plt.plot(0, 0, 'ko')
            # self.plot_self()

            plt.axhline(0, linestyle=':', color=0.7 + np.zeros(3), zorder=2)
            plt.axvline(0, linestyle=':', color=0.7 + np.zeros(3), zorder=2)
            plt.title('Egocentric')
            plt2.hide_ticklabels('y')
            plt.xlabel('$x$')

        elif to_plot == 'retina':
            h1 = plt.plot(
                r_plot[:,0], r_plot[:,1], 'o',
                mfc = markerfacecolor,
                mec = markeredgecolor
            )
            hs.append(h1)
            self.plot_self(dims=(0,2))
            plt.axis('equal')
            # plt.axis('square')
            plt.plot()
            beautify_plot(ylim_ret, xlim_ret)
            plt.title('Retina')
            plt.axhline(0, linestyle=':', color=0.7 + np.zeros(3), zorder=-1)
            plt.axvline(0, linestyle=':', color=0.7 + np.zeros(3), zorder=-1)
            plt.xlabel('$x$')
            plt.ylabel('$z$', rotation=0, labelpad=10, va='center')
            plt2.hide_ticklabels('y')

        else:
            raise ValueError('%s is not supported!' % to_plot)

        return hs, x_ego

    def plot_state(self, mu, sigma,
                   obs=None,
                   plot_filled=None,
                   plot_obs=True,
                   colors=None,
                   **kwargs):
        mu_s_loc_hom, mu_s_ori, mu_x_hom = self.parse_vec_hidden(
            mu, stack_objs=True
        )
        sig_s_loc_hom, _, sig_x_hom = self.parse_mat_hidden(
            sigma, stack_objs=True
        )
        if obs is None:
            obs = mu
        obs_s_loc_hom, obs_s_rad, obs_x_hom = self.parse_vec_hidden(
            obs, stack_objs=True
        )

        # fig = plt.figure('centroids')
        # plt2.subfigureRC(4,4,3,1,fig)

        n_landmark = mu_x_hom.shape[-2]

        if colors is None:
            colors = ['r', 'b', 'g', 'y']

        if plot_filled is None:
            plot_filled = [True] * n_landmark

        for i_landmark in range(n_landmark):
            kw = argsutil.kwdefault(
                kwargs,
                self_sig=sig_s_loc_hom,
                landmark_sig=sig_x_hom[0, [i_landmark], :self.n_dim,
                             :self.n_dim],
                color=colors[i_landmark % len(colors)],
            )
            self.plot_allo_ego_retina(
                self_euc=self.h2e(obs_s_loc_hom[[0], :]),
                self_ori=mu_s_ori[[0], :],
                landmark_orig=self.h2e(obs_x_hom[0, [i_landmark], :]),
                self_mu=self.h2e(mu_s_loc_hom[[0], :]),
                landmark_mu=self.h2e(mu_x_hom[0, [i_landmark], :]),
                fill_marker=plot_filled[i_landmark],
                **kw
            )
        # plt.show()
        # print('---')

class Proj3DOriSpeed(Proj3DOri):
    def __init__(self):
        super().__init__()
        self.n_dim_speed = 1

    @property
    def n_dim_self(self):
        """:rtype: int"""
        return self.n_dim_hom + self.n_dim_ori + self.n_dim_speed

    def cat_vec_hidden(self, s_loc_hom, s_ori_hom, x_hom, s_speed, *args,
                       ori_type='ori', **kwargs):
        # Concatenate into a hidden state vector [batch, dim]
        if ori_type == 'rad':
            s_ori_hom = self.e2h(self.rad2ori(s_ori_hom))
        elif ori_type == 'ori':
            s_ori_hom = s_ori_hom
        else:
            raise ValueError()

        return torch.cat(
            npt.expand_upto_dim([s_loc_hom, s_ori_hom, s_speed, x_hom], -1),
            dim=-1
        )

class Proj3DGeom(object):
    """
    Purely contain geometry; variable dimension(s) first for easy
    indexing.
    """
    def __init__(self):
        self.n_dim = 3

    def ori2rad(self, ori):
        """
        :param ori: [x, y, z] tensors: first dim is xyz, the rest is
        batch. Can be either homogeneous or euclidean (scale invariant)
        :type ori: torch.Tensor
        :return: pitch, roll, yaw = [rad_x, rad_y, rad_z] tensors:
        first dim is xyz (or, z/y, x/z, y/x), the rest is batch.
        :rtype torch.Tensor
        """
        rot_rad = torch.empty_like(ori)
        rot_rad[0] = torch.atan2(ori[2], ori[1])
        rot_rad[1] = torch.atan2(ori[0], ori[2])
        rot_rad[2] = torch.atan2(ori[1], ori[0])
        return rot_rad

    def get_mat_rotataion(self, rot_rad):
        """
        :param rot_rad: [n_dim, batch] (homogeneous or euclidean)
        :type rot_rad: torch.Tensor
        :return rot_mat[n_dim+1, n_dim+1, batch] (in homogeneous
        coordinate)
        :rtype: torch.Tensor
        """
        siz = rot_rad.shape[1:]
        rot_rad = torch.unsqueeze(torch.unsqueeze(rot_rad, 1), 1)
        rot_x = rot_rad[0] # [1, 1, batch]
        cos_x, sin_x = torch.cos(rot_x), torch.sin(rot_x)
        rot_y = rot_rad[1] # [1, 1, batch]
        cos_y, sin_y = torch.cos(rot_y), torch.sin(rot_y)
        rot_z = rot_rad[2] # [1, 1, batch]
        cos_z, sin_z = torch.cos(rot_z), torch.sin(rot_z)
        zeros = npt.zeros_like(cos_x) # [1, 1, batch]
        ones = npt.ones_like(cos_x) # [1, 1, batch]
        m_z = torch.cat([
            torch.cat([cos_z, -sin_z, zeros], 1),
            torch.cat([sin_z, cos_z, zeros], 1),
            torch.cat([zeros, zeros, ones], 1),
        ], 0)
        m_x = torch.cat([
            torch.cat([ones, zeros, zeros], 1),
            torch.cat([zeros, cos_x, -sin_x], 1),
            torch.cat([zeros, sin_x, cos_x], 1),
        ], 0)
        m_y = torch.cat([
            torch.cat([cos_y, zeros, sin_y], 1),
            torch.cat([zeros, ones, zeros], 1),
            torch.cat([-sin_y, zeros, cos_y], 1),
        ], 0)

        # roll first (m_y), then pitch (m_x), then yaw (m_z).
        # when expressed matrix multiplication, the order is reversed.
        m = npt.zeros(torch.Size([self.n_dim + 1] * 2) + siz)
        m[-1,-1] = 1.
        m[:3,:3] = p2st(p2en(m_z, 2) @ p2en(m_x, 2) @ p2en(m_y, 2), 2)
        return m


    def rad2ori(self, rad):
        """
        :param rad: [rot_around_x, rot_around_y, rot_around_z]
        = [atan(z/y), atan(x/z), atan(y/x)]
        :type rad: torch.Tensor
        :return: ori: unit vector toward rad in the Euclidean coord.
        :rtype: torch.Tensor
        """

        rot_mat = self.get_mat_rotataion(rad)
        # Unit vector toward +1 in the homogeneous coordinates
        # (hence 1 at the end)
        ori_hom = npt.matvecmul0(
            rot_mat,
            self.e2h(npt.tensor([1.] + [0.] * (self.n_dim - 1)))
        )
        return self.h2e(ori_hom)


    def euc2hom(self, euc):
        return torch.cat([euc, npt.ones_like(euc)], 0)
    e2h = euc2hom


    def hom2euc(self, hom):
        return hom[:self.n_dim] / hom[self.n_dim]
    h2e = hom2euc


    def rotate_to_align(self, src_hom, dst_hom, add_dst=False):
        """
        :type src_hom: torch.Tensor
        :type dst_hom: torch.Tensor
        :param add_dst: if False (default), return src - dst; if True,
        return src + dst (in terms of azimuth & elevation)
        :rtype: torch.Tensor
        """
        rad = self.ori2rad(dst_hom)
        if add_dst:
            rot = self.get_mat_rotataion(rad)
        else:
            rot = self.get_mat_rotataion(-rad)
        return npt.matvecmul0(rot, src_hom)



#%%
if __name__ == '__main__':
    #
    # #%%
    # geom = ProjectiveGeometry3D() # type: ProjectiveGeometry3D
    # self = geom
    #
    # #%%
    # # geom.demo_space_vectorized()
    # ssm = geom.demo_centroid()

    # %%
    geom = Proj3DOri()  # type: Proj3DOri
    self = geom
    ssm = geom.demo_centroid()

    #%%
    x_obj = npt.tensor([[1., 0., 0., 1.]])
    self_ori = npt.tensor([[-1., 0., 0., 1.]])
    m = geom.get_mat_ori2y(self_ori)
    print(m)
    print(m @ v2m(self_ori))
    print(m @ v2m(x_obj))
#%%
