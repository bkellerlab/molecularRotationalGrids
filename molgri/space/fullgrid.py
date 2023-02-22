from copy import copy

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from scipy.spatial import SphericalVoronoi
from scipy.constants import pi
import pandas as pd
import matplotlib.path as mplPath

from molgri.constants import SMALL_NS
from molgri.space.rotobj import SphereGridFactory
from molgri.molecules.parsers import TranslationParser, GridNameParser
from molgri.paths import PATH_OUTPUT_FULL_GRIDS
from molgri.space.utils import norm_per_axis, normalise_vectors, angle_between_vectors
from molgri.wrappers import save_or_use_saved


class FullGrid:

    def __init__(self, b_grid_name: str, o_grid_name: str, t_grid_name: str, use_saved: bool = True):
        """
        A combination object that enables work with a set of grids. A parser that

        Args:
            b_grid_name: body rotation grid (should be made into a 4D sphere grid)
            o_grid_name: origin rotation grid (should be made into a 3D sphere grid)
            t_grid_name: translation grid
        """
        b_grid_name = GridNameParser(b_grid_name, "b")
        self.b_rotations = SphereGridFactory.create(alg_name=b_grid_name.get_alg(), N=b_grid_name.get_N(),
                                                    dimensions=4, use_saved=use_saved)
        o_grid_name = GridNameParser(o_grid_name, "o")
        self.o_rotations = SphereGridFactory.create(alg_name=o_grid_name.get_alg(), N=o_grid_name.get_N(),
                                                    dimensions=3, use_saved=use_saved)
        self.o_positions = self.o_rotations.get_grid_as_array()
        self.t_grid = TranslationParser(t_grid_name)
        self.use_saved = use_saved
        self.save_full_grid()

    def get_name(self):
        o_name = self.o_rotations.get_name(with_dim=False)
        b_name = self.b_rotations.get_name(with_dim=False)
        return f"o_{o_name}_b_{b_name}_t_{self.t_grid.grid_hash}"

    def get_radii(self):
        return self.t_grid.get_trans_grid()

    def get_between_radii(self):
        radii = self.get_radii()
        # increments = self.t_grid.get_increments()
        # delta_radii = list(increments[1:])
        # delta_radii.append(delta_radii[-1])
        # delta_radii = np.array(delta_radii)
        #previous_radii = radii-radii[0]
        #between_radii = np.cbrt(2*radii**3-previous_radii**3) # equal volume
        #between_radii = radii + np.cbrt(2) * delta_radii    # equal volume

        increases = self.t_grid.get_increments() / 2
        between_radii = radii + increases

        # constant_delta = delta_radii[-1]
        # f = 1/constant_delta *(-radii[0] + 2**(2/3)*constant_delta +2**(1/3)*constant_delta+constant_delta)
        # print("f=", f)
        # between_radii  = radii + f*constant_delta
        return between_radii

    def get_body_rotations(self) -> Rotation:
        return Rotation.from_quat(self.b_rotations.get_grid_as_array())

    def get_position_grid(self) -> NDArray:
        """
        Get a 'product' of o_grid and t_grid so you can visualise points in space at which COM of the second molecule
        will be positioned. Important: Those are points on spheres in real space.

        Returns:
            an array of shape (len_o_grid, len_t_grid, 3) in which the elements of result[0] have the first
            rotational position, each line at a new (increasing) distance, result[1] the next rotational position,
            again at all possible distances ...
        """
        dist_array = self.get_radii()
        o_grid = self.o_positions
        num_dist = len(dist_array)
        num_orient = len(o_grid)
        result = np.zeros((num_dist, num_orient, 3))
        for i, dist in enumerate(dist_array):
            result[i] = np.multiply(o_grid, dist)
            norms = norm_per_axis(result[i])
            assert np.allclose(norms, dist), "In a position grid, all vectors in i-th 'row' should have the same norm!"
        result = np.swapaxes(result, 0, 1)
        return result

    @save_or_use_saved
    def get_flat_position_grid(self):
        pos_grid = self.get_position_grid()
        pos_grid = np.swapaxes(pos_grid, 0, 1)
        return pos_grid.reshape((-1, 3))

    def save_full_grid(self):
        np.save(f"{PATH_OUTPUT_FULL_GRIDS}position_grid_{self.get_name()}", self.get_position_grid())

    def get_full_voronoi_grid(self):
        try:
            return FullVoronoiGrid(self)
        except AttributeError:
            return None

    def point2cell_position_grid(self, points_vector: NDArray):
        # determine index within a layer - the layer grid point to which the point vectors are closest
        rot_points = self.o_rotations.get_grid_as_array()
        # this automatically select the one of angles that is < pi
        angles = angle_between_vectors(points_vector, rot_points)
        indices_within_layer = np.argmin(angles, axis=1)

        # determine radii of cells
        norms = norm_per_axis(points_vector)
        layers = np.zeros((len(points_vector),))
        vor_radii = self.get_between_radii()

        # find the index of the layer to which each point belongs
        for i, norm in enumerate(norms):
            for j, vor_rad in enumerate(vor_radii):
                # because norm keeps the shape of the original array
                if norm[0] < vor_rad:
                    layers[i] = j
                    break
            else:
                layers[i] = np.NaN

        layer_len = len(rot_points)
        indices = layers * layer_len + indices_within_layer
        return indices

    def points2cell_scipy(self, points_vector: NDArray):
        cdist = scipy.spatial.distance.cdist(points_vector, self.get_flat_position_grid())
        return np.argmin(cdist, axis=1)



class FullVoronoiGrid:

    def __init__(self, full_grid: FullGrid):
        self.full_grid = full_grid
        self.use_saved = self.full_grid.use_saved
        self.flat_positions = self.full_grid.get_flat_position_grid()
        self.all_sv = None
        self.get_voronoi_discretisation()

    def get_name(self):
        return f"voronoi_{self.full_grid.get_name()}"

    def _change_voronoi_radius(self, sv: SphericalVoronoi, new_radius):
        sv.radius = new_radius
        sv.vertices = normalise_vectors(sv.vertices, length=new_radius)
        sv.points = normalise_vectors(sv.points, length=new_radius)
        # important that it's a copy!
        return copy(sv)

    @save_or_use_saved
    def get_voronoi_discretisation(self):
        if self.all_sv is None:
            unit_sph_voronoi = self.full_grid.o_rotations.get_spherical_voronoi_cells()
            between_radii = self.full_grid.get_between_radii()
            self.all_sv = [self._change_voronoi_radius(unit_sph_voronoi, r) for r in between_radii]
        return self.all_sv

    @save_or_use_saved
    def find_voronoi_vertices_of_point(self, point_index: int):
        my_point = Point(point_index, self)

        vertices = my_point.get_vertices()
        return vertices

    def get_division_area(self, index_1: int, index_2: int, print_message=True):
        point_1 = Point(index_1, self)
        point_2 = Point(index_2, self)

        # if they are sideways neighbours
        if np.isclose(point_1.d_to_origin, point_2.d_to_origin):
            # vertices_above
            vertices_1a = point_1.get_vertices_above()
            vertices_2a = point_2.get_vertices_above()
            r_larger = np.linalg.norm(vertices_1a[0])
            set_vertices_1a = set([tuple(v) for v in vertices_1a])
            set_vertices_2a = set([tuple(v) for v in vertices_2a])
            intersection_a = set_vertices_1a.intersection(set_vertices_2a)
            # vertices below
            vertices_1b = point_1.get_vertices_below()
            r_smaller = np.linalg.norm(vertices_1b[0])
            if len(intersection_a) != 2:
                if print_message:
                    print(f"Points {index_1} and {index_2} are not neighbours.")
                return
            else:
                intersection_list = list(intersection_a)
                theta = angle_between_vectors(np.array(intersection_list[0]), np.array(intersection_list[1]))
                return theta / 2 * (r_larger ** 2 - r_smaller ** 2)
        # if point_1 right above point_2
        # both points on the same ray
        if np.allclose(point_1.get_normalised_point(), point_2.get_normalised_point()):
            if point_1.index_radial == point_2.index_radial + 1:
                return point_2.get_area_above()
            elif point_2.index_radial == point_1.index_radial + 1:
                return point_1.get_area_above()
        if print_message:
            print(f"Points {index_1} and {index_2} are not neighbours.")
        return

    def get_volume(self, index):
        point = Point(index, self)
        return point.get_cell_volume()

    def get_all_voronoi_volumes(self):
        N = len(self.flat_positions)
        volumes = np.zeros((N,))
        for i in range(0, N):
            volumes[i] = self.get_volume(i)
        return volumes


class Point:

    def __init__(self, index_position_grid, full_sv: FullVoronoiGrid):
        self.full_sv = full_sv
        self.index_position_grid = index_position_grid
        self.point = self.full_sv.flat_positions[index_position_grid]
        self.d_to_origin = np.linalg.norm(self.point)
        self.index_radial = self._find_index_radial()
        self.index_within_sphere = self._find_index_within_sphere()

    def get_normalised_point(self):
        return normalise_vectors(self.point, length=1)

    def _find_index_radial(self):
        point_radii = self.full_sv.full_grid.get_radii()
        for i, dist in enumerate(point_radii):
            if np.isclose(dist, self.d_to_origin):
                return i
        else:
            raise ValueError("The norm of the point not close to any of the radii.")

    def _find_index_within_sphere(self):
        radial_index = self.index_radial
        num_o_rot = self.full_sv.full_grid.o_rotations.N
        return self.index_position_grid - num_o_rot * radial_index

    def _find_index_sv_above(self):
        for i, sv in enumerate(self.full_sv.all_sv):
            if sv.radius > self.d_to_origin:
                return i
        else:
            # the point is outside the largest voronoi sphere
            return None

    def get_sv_above(self):
        return self.full_sv.all_sv[self._find_index_sv_above()]

    def get_radius_above(self):
        sv_above = self.get_sv_above()
        return sv_above.radius

    def get_radius_below(self):
        sv_below = self.get_sv_below()
        if sv_below is None:
            return 0.0
        else:
            return sv_below.radius

    def get_sv_below(self):
        index_above = self._find_index_sv_above()
        if index_above != 0:
            return self.full_sv.all_sv[index_above-1]
        else:
            return None

    def get_area_above(self):
        sv_above = self.get_sv_above()
        areas = sv_above.calculate_areas()
        return areas[self.index_within_sphere]

    def get_area_below(self):
        sv_below = self.get_sv_below()
        if sv_below is None:
            return 0.0
        else:
            areas = sv_below.calculate_areas()
            return areas[self.index_within_sphere]

    def get_vertices_above(self):
        sv_above = self.get_sv_above()
        regions = sv_above.regions[self.index_within_sphere]
        vertices_above = sv_above.vertices[regions]
        return vertices_above

    def get_vertices_below(self):
        sv_below = self.get_sv_below()

        if sv_below is None:
            vertices_below = np.zeros((1, 3))
        else:
            regions = sv_below.regions[self.index_within_sphere]
            vertices_below = sv_below.vertices[regions]

        return vertices_below

    def get_vertices(self):
        vertices_above = self.get_vertices_above()
        vertices_below = self.get_vertices_below()

        return np.concatenate((vertices_above, vertices_below))

    def get_cell_volume(self):
        radius_above = self.get_radius_above()
        radius_below = self.get_radius_below()
        area_above = self.get_area_above()
        area_below = self.get_area_below()
        volume = 1/3 * (radius_above * area_above - radius_below * area_below)
        return volume


class ConvergenceFullGridO:

    def __init__(self, b_grid_name: str, t_grid_name: str,  o_alg_name: str, N_set = None, use_saved=False, **kwargs):
        if N_set is None:
            N_set = SMALL_NS
        self.N_set = N_set
        self.alg_name = o_alg_name
        self.use_saved = use_saved
        self.list_full_grids = self.create(b_grid_name=b_grid_name, t_grid_name=t_grid_name,  o_alg_name=o_alg_name,
                                           N_set=self.N_set, use_saved=use_saved, **kwargs)

    def get_name(self):
        b_name = self.list_full_grids[0].b_rotations.get_name(with_dim=False)
        t_name = self.list_full_grids[0].t_grid.grid_hash
        return f"convergence_o_{self.alg_name}_b_{b_name}_t_{t_name}"

    @classmethod
    def create(cls,  b_grid_name: str, t_grid_name: str,  o_alg_name: str, N_set, **kwargs) -> list:
        list_full_grids = []
        for N in N_set:
            fg = FullGrid(b_grid_name=b_grid_name, o_grid_name=f"{o_alg_name}_{N}", t_grid_name=t_grid_name, **kwargs)
            list_full_grids.append(fg)
        return list_full_grids

    @save_or_use_saved
    def get_voronoi_volumes(self):
        data = []
        for N, fg in zip(self.N_set, self.list_full_grids):
            vor_radius = list(fg.get_between_radii())
            vor_radius.insert(0, 0)
            vor_radius = np.array(vor_radius)
            fvg = fg.get_full_voronoi_grid()
            ideal_volumes = 4/3 * pi * (vor_radius[1:]**3 - vor_radius[:-1]**3) / N
            real_volumes = fvg.get_all_voronoi_volumes()
            for i, volume in enumerate(real_volumes):
                layer = i//N
                data.append([N, layer, ideal_volumes[i//N], volume])
        df = pd.DataFrame(data, columns=["N", "layer", "ideal volume", "Voronoi cell volume"])
        return df




if __name__ == "__main__":
    full_grid = FullGrid(b_grid_name="cube3D_16", o_grid_name="ico_15", t_grid_name="[1, 2, 3]")
    # fvg = full_grid.get_full_voronoi_grid()
    # print(fvg)
    points = np.array([[-1, 3, 2], [-0.5, -0.5, 1], [22, 8, 4], [1, 1, 222]])
    print(full_grid.point2cell_position_grid(points))