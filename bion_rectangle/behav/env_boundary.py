#  Copyright (c) 2020  Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
import matplotlib as mpl
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from importlib import reload
from typing import Type, Tuple

# import torch
from collections import OrderedDict as odict

from bion_rectangle.utils import np2
from bion_rectangle.utils import plt2
from bion_rectangle.utils import colors_given_contrast
from bion_rectangle.utils import numpytorch as npt

# torch.set_default_tensor_type(torch.DoubleTensor) # To prevent underflow

#%%
reload(npt)
reload(plt2)
# reload(pred)
# reload(proj3d)

enforce_tensor = npt.enforce_tensor
p2st = npt.permute2st
p2en = npt.permute2en
v2m = npt.vec2matmul
m2v = npt.matmul2vec
npy = npt.npy
npys = npt.npys

# # UNUSED
# pth_root = '../Data_SLAM/a04_deform'
# pth_out = os.path.join(pth_root, 'env_boundary')
# cacheutil.mkdir(pth_out)
# if not os.path.exists(pth_out):
#     os.mkdir(pth_out)
# pth_cache = os.path.join(pth_out, 'cache')
# if not os.path.exists(pth_cache):
#     os.mkdir(pth_cache)


class EnvSpace:
    """
    Summary:
    ------------
    Abstract class for confirming whether the states are continuous.
    """
    pass


class EnvBoundary(EnvSpace):
    name = ''
    reference = ''

    def __init__(
        self,
        contrast=0.2,
        contrast_btw_walls=None,
        use_color=True,
        # scale ratio between Krupic et al. 2015 (rat)
        # and Bellmund et al. 2020 (human)
        # self_height / height_wall = 0.48 (cf. Krupic: 0.0804)
        height_wall=3.75,  # Following Bellmund et al. 2019
        self_height=1.8,  # Following Bellmund et al. 2019
        x_max=40.,
        y_max=None,
        width=None,
        height=None,
        species='human',
        dx=5.,
        dy=None,
        instance_name='',
        **kwargs  # ignore unused arguments
    ):
        """

        :param contrast:
        :param contrast_btw_walls:
        :param use_color:
        :param height_wall: 3.75,  # Following Bellmund et al. 2019
        :param self_height: 1.8,  # Following Bellmund et al. 2019
        :param x_max:
        :param width:
        :param height:
        :param species: 'human'|'rat'
        :param instance_name: .name is for class, instance_name is for instance
          (e.g., for caching)
        """
        self.instance_name = instance_name

        self.self_height = self_height
        self.corners = np.zeros((0, 2))  # corners[:, (x, y)]
        self.corners_inner = np.zeros((0, 2))
        self.polygon = mpl.path.Path(self.corners[:, :2])
        self.polygon_inner = mpl.path.Path(self.corners_inner[:, :2])
        self.height_wall = height_wall
        self.self_height = self_height
        self.x_max = x_max
        if y_max is None:
            y_max = x_max
        self.y_max = y_max
        self.dx = dx
        if dy is None:
            dy = dx
        self.dy = dy

        if width is None:
            width = x_max * 2
        if height is None:
            height = x_max * 2
        self.width = width
        self.height = height

        # The four walls and ceiling & floor are vertices of a regular
        # octahedron in the RGB space around gray (0.5, 0.5, 0.5),
        # so that every pair of adjacent surfaces have the same distance
        # from each other.
        self.use_color=use_color
        self.contrast = contrast
        self._contrast_btw_walls = contrast_btw_walls
        # self._contrast = 0.2
        # self._contrast_btw_walls = 0.2
        # self.set_contrast(contrast, contrast_btw_walls)

        self.species=species

    @property
    def x_max_corners(self):
        return np.amax(self.corners[:, 0])

    @property
    def x_min_corners(self):
        return np.amin(self.corners[:, 0])

    @property
    def y_max_corners(self):
        return np.amax(self.corners[:, 1])

    @property
    def y_min_corners(self):
        return np.amin(self.corners[:, 1])

    @property
    def width_corners(self):
        return self.x_max_corners - self.x_min_corners

    @property
    def height_corners(self):
        return self.y_max_corners - self.y_min_corners

    def get_kw_init(self) -> dict:
        return {
            k: self.__getattribute__(k) for k in [
                'contrast',
                'contrast_btw_walls',
                'use_color',
                'height_wall',
                'self_height',
                'x_max',
                'width',
                'height',
                'species',
                'offset_inner',
                'dx',
                'instance_name',
            ]
        }

    def get_xlim(self, margin=0.05):
        return np.array([-self.x_max, self.x_max]) * 0.9 * (1 + margin)

    def get_ylim(self, margin=0.05):
        return np.array([-self.y_max, self.y_max]) * 0.9 * (1 + margin)
        # return np.array([
        #     np.amin(self.corners[:, 1]), np.amax(self.corners[:, 1])
        # ]) * (1. + margin)

    def get_xlim_tight(self, margin=0.1):
        return np.array([
            np.amin(self.corners[:, 0]), np.amax(self.corners[:, 0])
        ]) * (1. + margin)

    def get_ylim_tight(self, margin=0.1):
        return np.array([
            np.amin(self.corners[:, 1]), np.amax(self.corners[:, 1])
        ]) * (1. + margin)

    # @property
    # def contrast(self):
    #     return self._contrast
    #
    # @contrast.setter
    # def contrast(self, v):
    #     self.set_contrast(contrast=v)
    #
    @property
    def contrast_btw_walls(self):
        if self._contrast_btw_walls is None:
            return self.contrast
        else:
            return self._contrast_btw_walls

    @contrast_btw_walls.setter
    def contrast_btw_walls(self, v):
        self._contrast_btw_walls = v

    # def set_contrast(self, contrast=None, contrast_btw_walls=None):
    #     if contrast is not None:
    #         self._contrast = contrast
    #     if contrast_btw_walls is not None:
    #         self._contrast_btw_walls = contrast_btw_walls
    #
    #     if self.use_color:
    #         # # floor: gray
    #         # self.color_floor = 0.5 + np.zeros(3)
    #         # #
    #         # # ceiling: gray
    #         # self.color_background = 0.5 + np.zeros(3)
    #         # #
    #         # # walls: colors from a regular tetrahedron,
    #         # # such that any pair of colors are mixed, the mix doesn't
    #         # # resemble any other pair, and any triple of colors mixed
    #         # # doesn't result in the other color.
    #         # self.color_walls = 0.5 + np.array([
    #         #     [1., 1., 1.], # light gray
    #         #     [-1., 1., -1.], # red
    #         #     [-1., -1., 1.], # blue
    #         #     [-1., 1., -1.], # green
    #         # ]) * self.contrast_btw_walls / 2
    #
    #         colors = self.get_colors()
    #         self.color_floor, self.color_background, self.color_walls = (
    #             colors[c] for c in ['floor', 'background', 'walls']
    #         )
    #
    #     else:
    #         self.color_floor = np.zeros(3) + 0.5 + self._contrast / 2.
    #         self.color_background = np.zeros(3) + 0.5 + self._contrast / 2.
    #         self.color_walls = np.zeros([4, 3]) + 0.5 - self._contrast / 2.

    @property
    def colors_all(self) -> np.ndarray:
        """

        :return: color[(ground, ceiling, wall0, wall1, wall2, wall3), (R, G, B)]
            0 <= color < 1
        """
        return colors_given_contrast.colors_given_contrast(
            contrast_btw_walls=self.contrast_btw_walls,
            contrast_ground=self.contrast
        )

    @property
    def color_walls(self) -> np.ndarray:
        if self.use_color:
            return 0.5 + np.array([
                [1., 0., 0.], # bright red
                [0., 1., 0.], # bright green
                [-1., 0., 0.], # dark green
                [0., -1., 0.], # dark red
            ]) * self.contrast_btw_walls / 2

            # TODO: use constant-sum colors instead of the above
            # return self.colors_all[colors_given_contrast.WALLS]
        else:
            return np.zeros([4, 3]) + 0.5 - self.contrast / 2.

    @property
    def color_background(self):
        if self.use_color:
            # return self.colors_all[colors_given_contrast.CEILING]
            return 0.5 + np.array([0., 0., 1.]) * self.contrast / 2
        else:
            return np.zeros(3) + 0.5 + self.contrast / 2.

    @property
    def color_floor(self):
        if self.use_color:
            # return self.colors_all[colors_given_contrast.GROUND]
            return 0.5 + np.array([0., 0., -1.]) * self.contrast / 2
        else:
            return np.zeros(3) + 0.5 + self.contrast / 2.


    # def get_colors(self, contrast=None, contrast_btw_walls=None):
    #     if contrast is None:
    #         contrast = self.contrast
    #     if contrast_btw_walls is None:
    #         contrast_btw_walls = self.contrast_btw_walls
    #     return {
    #         'floor': 0.5 + np.array([0., 0., -1.]) * contrast / 2,
    #         'background': 0.5 + np.array([0., 0., 1.]) * contrast / 2,
    #         'walls': 0.5 + np.array([
    #             [1., 0., 0.], # bright red
    #             [0., 1., 0.], # bright green
    #             [-1., 0., 0.], # dark green
    #             [0., -1., 0.], # dark red
    #         ]) * contrast_btw_walls / 2
    #     }

    def is_inside(
        self, xy: np.ndarray,
        x_incl: str = 'all',
        use_inner=True,
        margin_x_half=None,
    ) -> np.ndarray:
        """
        :type xy: np.ndarray
        :param xy: [point, (x, y)]
        :param x_incl: 'all'(default)|'left'|'right'
        :param use_inner: use polygon_inner
            (e.g., when subjects cannot approach
            the walls beyond a certain margin)
        :rtype: np.ndarray(dtype=bool)
        :return: inside[point], x_half
        """
        siz0 = xy.shape
        xy = np.reshape(xy, [-1, 2])

        if use_inner:
            polygon = self.polygon_inner
        else:
            polygon = self.polygon
        if margin_x_half is None:
            # DEBUGGED: to avoid introducing bias by assigning
            #   half point always to either half
            margin_x_half = self.dx / 4  # PARAM

        is_inside1 = polygon.contains_points(xy)

        # disallow being too close to the wall
        is_inside1 = is_inside1 & ((
                np.abs(xy[:, 0] - np.amin(self.corners[:, 0])) >= 1e-3
        ) & (
                np.abs(xy[:, 0] - np.amax(self.corners[:, 0])) >= 1e-3
        ) & (
                np.abs(xy[:, 1] - np.amin(self.corners[:, 1])) >= 1e-3
        ) & (
                np.abs(xy[:, 1] - np.amax(self.corners[:, 1])) >= 1e-3
        ))

        # x_half = float(np.nanmedian(xy[is_inside1, 0]))  # type: float
        if x_incl == 'all':
            pass
        elif x_incl == 'left':
            is_inside1 = is_inside1 & (
                xy[:, 0] < self.get_x_half() - margin_x_half)
        elif x_incl == 'right':
            is_inside1 = is_inside1 & (
                xy[:, 0] > self.get_x_half() + margin_x_half)
        else:
            raise ValueError()

        return np.reshape(is_inside1, siz0[:-1])

    def get_x_half(self) -> float:
        return 0.

    def plot_walls(
        self,
        *args, contrast=1.,
        scale=(1., 1.),
        use_inner=False,
        bevel_outer_scale=1.1,
        bevel_auto_lim=True,
        mode='bevel',
        corners=None,
        **kwargs
    ):
        """

        :param args:
        :param contrast:
        :param scale:
        :param use_inner:
        :param mode: 'line'|'bevel'
        :param kwargs:
        :return:
        """

        if corners is None:
            if use_inner:
                corners = self.corners_inner
            else:
                corners = self.corners

        h = []
        for corner0, corner1, color in \
                zip(corners, np.roll(corners, -1, 0), self.color_walls):

            color = np.clip(
                (color - 0.5) / self.contrast
                * contrast + 0.5, 0., 1.)

            if mode == 'line':
                h.append(plt.plot(
                    np.array([corner0[0], corner1[0]]) * scale[0],
                    np.array([corner0[1], corner1[1]]) * scale[1],
                    *args,
                    **{'color': color, **kwargs}
                ))
            elif mode == 'bevel':
                h1 = Polygon(
                    np.stack([
                        np.r_[
                            corner0[0] * scale[0],
                            corner1[0] * scale[0],
                            corner1[0] * scale[0] * bevel_outer_scale,
                            corner0[0] * scale[0] * bevel_outer_scale,
                        ],
                        np.r_[
                            corner0[1] * scale[1],
                            corner1[1] * scale[1],
                            corner1[1] * scale[1] * bevel_outer_scale,
                            corner0[1] * scale[1] * bevel_outer_scale,
                        ],
                    ], -1),
                    *args,
                    closed=True,
                    **{
                        **kwargs, **{
                            'facecolor': color,
                            'linewidth': 0,
                        }
                    }
                    )
                ax = plt.gca()
                ax.add_patch(h1)
                h.append(h1)
            else:
                raise ValueError()

        if mode == 'bevel' and bevel_auto_lim:
            plt.xlim(
                np.amin(corners[:, 0]) * bevel_outer_scale,
                np.amax(corners[:, 0]) * bevel_outer_scale
            )
            plt.ylim(
                np.amin(corners[:, 1]) * bevel_outer_scale,
                np.amax(corners[:, 1]) * bevel_outer_scale
            )

        return h
        # return plt.plot(
        #     np.concatenate([corners[:, 0], corners[[0], 0]], 0),
        #     np.concatenate([corners[:, 1], corners[[0], 1]], 0),
        #     *args, **kwargs
        # )

    # def set_species(self, species):
    #     if species == self.species:
    #         return
    #     else:
    #         if species == 'human':
    #             factor = 1. / self.rat_over_human
    #             self.reference = 'Bellmund2020'
    #         elif species == 'rat':
    #             factor = self.rat_over_human
    #             self.reference = 'Krupic2015'
    #         else:
    #             raise ValueError()
    #
    #     for v in (
    #         'coerners', 'corners_inner', 'trapez_len',
    #         'trapez_short', 'trapez_long', 'offset', 'offset2',
    #         'corners', 'corners_inner'
    #     ):
    #         self.__dict__[v] *= factor
    #
    #     self.polygon = mpl.path.Path(self.corners[:,:2])
    #     self.polygon_inner = mpl.path.Path(self.corners_inner[:, :2])
    #     self.species = species

    def is_crossing_boundary(
            self, xy_src: np.ndarray, xy_dst: np.ndarray
    ) -> np.ndarray:
        """

        :param xy_src: [..., xy]
        :param xy_dst: [..., xy]
        :return: crossing_boundary[...]
        """
        corners = np2.prepend_dim(
            np.r_[self.corners, self.corners[[0]]][..., :2],
            xy_src.ndim - 1)
        crossing_boundary = np.any(np2.intersect(
            xy_src[..., None, :],
            xy_dst[..., None, :],
            corners[..., :-1, :],
            corners[..., 1:, :]
        ), -1)  # crossing any boundary
        return crossing_boundary


# ratio of the sizes of the square in Krupic / Bellmund studies
HUMAN_HEIGHT_BELLMUND = 1.8                # added as was raising an error in env_derdikman.py
RAT_OVER_HUMAN = 0.9 / 40.27               # when running main_fig_ideal_obs_hairpin.py
RAT_OVER_HUMAN_APPROX = 0.02
BIN_SIZE_KRUPIC = 0.025


class Rectangle(EnvBoundary):
    name='rect'
    def __init__(
            self,
            width: float,
            height: float,
            offset_inner=0.,
            offset_xy=(0., 0.),
            dx=0.05,
            **kwargs
    ):
        super().__init__(
            width=width, height=height,
            dx=dx,
            **kwargs
        )

        self.offset_xy = np.array(offset_xy)

        self.corners = np.array([
            (0.5 * width, -0.5 * height, 0),
            (0.5 * width, 0.5 * height, 0),
            (-0.5 * width, 0.5 * height, 0),
            (-0.5 * width, -0.5 * height, 0),
        ]) + np.array([list(offset_xy) + [0.]])

        self.corners_inner = np.array([
            (+width / 2. - offset_inner,
             -height / 2. + offset_inner, 0),
            (+width / 2. - offset_inner,
             +height / 2. - offset_inner, 0),
            (-width / 2. + offset_inner,
             +height / 2. - offset_inner, 0),
            (-width / 2. + offset_inner,
             -height / 2. + offset_inner, 0),
        ]) + np.array([list(offset_xy) + [0.]])

        self.polygon = mpl.path.Path(self.corners[:, :2])
        self.polygon_inner = mpl.path.Path(self.corners_inner[:, :2])

    def get_xlim(self, margin=0.) -> Tuple[float, float]:
        x_range = self.x_max - self.x_min
        return (
            self.x_max - x_range * (1 + margin / 2),
            self.x_min + x_range * (1 + margin / 2)
        )

    def get_ylim(self, margin=0.) -> Tuple[float, float]:
        y_range = self.y_max - self.y_min
        return (
            self.y_max - y_range * (1 + margin / 2),
            self.y_min + y_range * (1 + margin / 2)
        )

    @property
    def shape(self):
        return self.width, self.height


def ____OKeefe_Burgess_1996___():
    pass


class OB96(Rectangle):
    name = 'None'
    reference = 'OKeefeBurgess1996'

    def __init__(
            self,
            **kwargs
    ):
        kwargs = {
            'self_height': 1.8 * RAT_OVER_HUMAN,
            'x_max': 1.5 / 2,
            'height_wall': 0.61,
            'species': 'rat',
            'offset_inner': 0.,
            **kwargs
        }
        super().__init__(**kwargs)


class SmallSquare(OB96):
    name = 'SS'

    def __init__(self, **kwargs):
        kwargs = {
            'width': 0.61,
            'height': 0.61,
            **kwargs
        }
        super().__init__(**kwargs)


class LargeSquare(OB96):
    name = 'LS'

    def __init__(self, **kwargs):
        kwargs = {
            'width': 1.22,
            'height': 1.22,
            **kwargs
        }
        super().__init__(**kwargs)


class HorizontalRectangle(OB96):
    name = 'HR'

    def __init__(self, **kwargs):
        kwargs = {
            'width': 1.22,
            'height': 0.61,
            **kwargs
        }
        super().__init__(**kwargs)


class VerticalRectangle(OB96):
    name = 'VR'

    def __init__(self, **kwargs):
        kwargs = {
            'width': 0.61,
            'height': 1.22,
            **kwargs
        }
        super().__init__(**kwargs)


odict_OB96 = odict([
    ('SS', SmallSquare),
    ('HR', HorizontalRectangle),
    ('VR', VerticalRectangle),
    ('LS', LargeSquare),
])

