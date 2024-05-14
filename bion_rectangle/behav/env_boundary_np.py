from typing import Optional, Tuple
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.pyplot as plt

from collections import OrderedDict

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from bion_rectangle.utils.colors_given_contrast import colors_given_contrast
from bion_rectangle.utils.np2 import prepend_dim, intersect


HUMAN_HEIGHT_BELLMUND = 1.8
RAT_OVER_HUMAN = 0.9 / 40.27
RAT_OVER_HUMAN_APPROX = 0.02
BIN_SIZE_KRUPIC = 0.025


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
        contrast: float = 0.2,
        contrast_btw_walls: Optional[float] = None,
        use_color: bool = True,
        # scale ratio between Krupic et al. 2015 (rat)
        # and Bellmund et al. 2020 (human)
        # self_height / height_wall = 0.48 (cf. Krupic: 0.0804)
        wall_height: float = 3.75,  # Following Bellmund et al. 2019
        agent_height: float = 1.8,  # Following Bellmund et al. 2019
        x_max: float = 40.,
        y_max: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        species: str = "human",
        dx: float = 5.,
        dy: Optional[float] = None,
        instance_name: str = "",
        **kwargs  # ignore unused arguments
    ):
        self.instance_name = instance_name
        
        self.agent_height = agent_height
        self.corners = np.zeros((0, 2))
        self.corners_inner = np.zeros((0, 2))
        self.polygon = Path(self.corners[:, :2])
        self.polygon_inner = Path(self.corners_inner[:, :2])
        self.wall_height = wall_height
        
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
        
        self.use_color = use_color
        self.contrast = contrast
        self.contrast_btw_walls = contrast_btw_walls
        
        self.species = species
        
    @property
    def x_max_corners(self):
        return np.max(self.corners[:, 0])

    @property
    def x_min_corners(self):
        return np.min(self.corners[:, 0])
    
    @property
    def y_max_corners(self):
        return np.max(self.corners[:, 1])

    @property
    def y_min_corners(self):
        return np.min(self.corners[:, 1])
    
    @property
    def width_corners(self):
        return self.x_max_corners - self.x_min_corners
    
    @property
    def height_corners(self):
        return self.y_max_corners - self.y_min_corners

    def get_init_kwargs(self):
        return {
            k: self.__getattribute__(k) for k in [
                'contrast',
                'contrast_btw_walls',
                'use_color',
                'wall_height',
                'agent_height',
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
    
    def get_xlim_tight(self, margin=0.1):
        return np.array([np.min(self.corners[:, 0]), np.max(self.corners[:, 0])]) * (1 + margin)
    
    def get_ylim_tight(self, margin=0.1):
        return np.array([np.min(self.corners[:, 1]), np.max(self.corners[:, 1])]) * (1 + margin)
    
    @property
    def _contrast_btw_walls(self):
        if self.contrast_btw_walls is None:
            return self.contrast
        return self.contrast_btw_walls
    
    @_contrast_btw_walls.setter
    def _contrast_btw_walls(self, v: float):
        self.contrast_btw_walls = v
        
    @property
    def colors_all(self):
        return colors_given_contrast(
            contrast_btw_walls=self._contrast_btw_walls, 
            contrast_ground=self.contrast, 
        )
    
    @property
    def color_walls(self):
        if self.use_color:
            return 0.5 + np.array([
                [1., 0., 0.], # bright red
                [0., 1., 0.], # bright green
                [-1., 0., 0.], # dark green
                [0., -1., 0.], # dark red
            ]) * self._contrast_btw_walls / 2
        
        return np.zeros([4, 3]) + 0.5 - self.contrast / 2.
    
    @property
    def color_background(self):
        if self.use_color:
            return 0.5 + np.array([0., 0., 1.]) * self.contrast / 2
        return np.zeros(3) + 0.5 + self.contrast / 2
    
    @property
    def color_floor(self):
        if self.use_color:
            return 0.5 + np.array([0., 0., -1.]) * self.contrast / 2
        return np.zeros(3) + 0.5 + self.contrast / 2.
    
    def is_inside(
        self, 
        xy: np.ndarray, 
        x_incl: str = "all",
        use_inner: bool = True, 
        margin_x_half: Optional[float] = None, 
    ):
        size0 = xy.shape
        xy = xy.reshape((-1, 2))
        
        if use_inner:
            polygon = self.polygon_inner
        else:
            polygon = self.polygon
        
        if margin_x_half is None:
            margin_x_half = self.dx / 4
        
        is_inside_1 = polygon.contains_points(xy)
        
        is_inside_1 = is_inside_1 & (
            (np.abs(xy[:, 0] - np.min(self.corners[:, 0])) >= 1e-3) & 
            (np.abs(xy[:, 0] - np.max(self.corners[:, 0])) >= 1e-3) & 
            (np.abs(xy[:, 1] - np.min(self.corners[:, 1])) >= 1e-3) & 
            (np.abs(xy[:, 1] - np.max(self.corners[:, 1])) >= 1e-3)
        )
        
        if x_incl == "all":
            pass
        elif x_incl == "left":
            is_inside_1 = is_inside_1 & (xy[:, 0] < self.get_x_half() - margin_x_half)
        elif x_incl == "right":
            is_inside_1 = is_inside_1 & (xy[:, 0] > self.get_xlim_tight() + margin_x_half)
        else:
            raise ValueError
        
        return is_inside_1.reshape(size0[:-1])
    
    def get_x_half(self):
        return 0.0
    
    def plot_walls(
        self, 
        *args, 
        contrast: float=  1., 
        scale: Tuple[float] = (1., 1.), 
        use_inner: bool = False, 
        bevel_outer_scale: float = 1.1, 
        bevel_auto_lim: bool = True, 
        mode: str = "bevel", 
        corners: Optional[np.ndarray] = None, 
        **kwargs, 
    ):
        if corners is None:
            if use_inner:
                corners = self.corners_inner
            else:
                corners = self.corners
        
        h = []
        
        for c0, c1, color in zip(corners, np.roll(corners, -1, 0), self.color_walls):
            color = np.clip((color - 0.5) / self.contrast * contrast + 0.5, 0.0, 1.0)
            
            if mode == "line":
                h.append(
                    plt.plot(
                        np.array([c0[0], c1[0]]) * scale[0], 
                        np.array([c0[1], c1[1]]) * scale[1], 
                        *args, 
                        **{"color": color, **kwargs}
                    )
                )
            elif mode == "bevel":
                h1 = Polygon(
                    np.stack([
                        np.r_[
                            c0[0] * scale[0], 
                            c1[0] * scale[0], 
                            c1[0] * scale[0] * bevel_outer_scale, 
                            c0[0] * scale[0] * bevel_outer_scale, 
                        ], 
                        np.r_[
                            c0[1] * scale[1], 
                            c1[1] * scale[1], 
                            c1[1] * scale[1] * bevel_outer_scale, 
                            c0[1] * scale[1] * bevel_outer_scale, 
                        ]
                    ], axis=-1), 
                    *args, 
                    closed=True, 
                    **{
                        **kwargs, 
                        **{
                            "facecolor": color, 
                            "linewidth": 0.0, 
                        }
                    }
                )
                ax = plt.gca()
                ax.add_patch(h1)
                h.append(h1)
            else:
                raise ValueError
            
        if mode == "bevel" and bevel_auto_lim:
            plt.xlim(
                np.min(corners[:, 0]) * bevel_outer_scale, 
                np.max(corners[:, 0]) * bevel_outer_scale
            )
            plt.ylim(
                np.min(corners[:, 1]) * bevel_outer_scale, 
                np.max(corners[:, 1]) * bevel_outer_scale
            )
        
        return h
    
    def is_crossing_boundary(self, xy_src: np.ndarray, xy_dst: np.ndarray):
        corners = prepend_dim(
            np.r_[self.corners, self.corners[[0]]][..., :2], 
            xy_src.ndim - 1
        )
        crossing_boundary = np.any(intersect(
            xy_src[..., None, :], 
            xy_dst[..., None, :], 
            corners[..., :-1, :], 
            corners[..., 1:, :], 
        ), -1)
        
        return crossing_boundary
    
    
class Rectangle(EnvBoundary):
    name = "rectangle"
    def __init__(
        self, 
        width: float, 
        height: float, 
        offset_inner: float = 0., 
        offset_xy: Tuple[float] = (0., 0.), 
        dx: float = 0.05, 
        **kwargs
    ):
        super().__init__(width=width, height=height, dx=dx, **kwargs)
        
        self.offset_xy = offset_xy
        
        self.corners = np.array([
            (0.5 * width, -0.5 * height, 0), 
            (0.5 * width, 0.5 * height, 0), 
            (-0.5 * width, 0.5 * height, 0), 
            (-0.5 * width, -0.5 * height, 0), 
        ]) + np.array([list(offset_xy) + [0.]])
        
        self.corners_inner = np.array([
            (width / 2. - offset_inner, - height / 2. + offset_inner, 0), 
            (width / 2. - offset_inner, height / 2. - offset_inner, 0), 
            (-width / 2. + offset_inner, height / 2. - offset_inner, 0), 
            (-width / 2. + offset_inner, -height / 2. + offset_inner, 0), 
        ]) + np.array([list(offset_xy) + [0.]])
        
        self.polygon = Path(self.corners[:, :2])
        self.polygon_inner = Path(self.corners_inner[:, :2])
    
    def get_xlim(self, margin: float = 0.0):
        x_range = self.x_max - self.x_min
        return (
            self.x_max - x_range * (1 + margin / 2), 
            self.x_min + x_range * (1 + margin / 2), 
        )
    
    def get_ylim(self, margin: float = 0.0):
        y_range = self.y_max - self.y_min
        return (
            self.y_max - y_range * (1 + margin / 2), 
            self.y_min + y_range * (1 + margin / 2), 
        )
    
    @property
    def shape(self):
        return self.width, self.height
    

class OKeefeBurgess1996(Rectangle):
    name = "None"
    reference = "OKeefeBurgess1996"
    
    def __init__(self, **kwargs):
        kwargs = {
            "agent_height": 1.8 * RAT_OVER_HUMAN, 
            "x_max": 1.5 / 2, 
            "wall_height": 0.61, 
            "species": "rat", 
            "offset_inner": 0.0, 
            **kwargs
        }
        super().__init__(**kwargs)
    

class SmallSquare(OKeefeBurgess1996):
    name = "SmallSquare"
    
    def __init__(self, **kwargs):
        kwargs = {
            "width": 0.61, 
            "height": 0.61, 
            **kwargs
        }
        super().__init__(**kwargs)


class LargeSquare(OKeefeBurgess1996):
    name = "LargeSquare"
    
    def __init__(self, **kwargs):
        kwargs = {
            "width": 1.22, 
            "height": 1.22, 
            **kwargs
        }
        super().__init__(**kwargs)
        
        
class HorizontalRectangle(OKeefeBurgess1996):
    name = "HorizontalRectangle"
    
    def __init__(self, **kwargs):
        kwargs = {
            "width": 1.22, 
            "height": 0.61, 
            **kwargs
        }
        super().__init__(**kwargs)


class VerticalRectangle(OKeefeBurgess1996):
    name = "VerticalRectangle"
    
    def __init__(self, **kwargs):
        kwargs = {
            "width": 0.61, 
            "height": 1.22, 
            **kwargs
        }
        super().__init__(**kwargs)



OB96 = OrderedDict([
    ("SmallSquare", SmallSquare), 
    ("LargeSquare", LargeSquare), 
    ("HorizontalRectangle", HorizontalRectangle),
    ("VerticalRectangel", VerticalRectangle),
])
