from copy import copy

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from scipy.spatial import SphericalVoronoi

from molgri.space.rotobj import SphereGridFactory
from molgri.molecules.parsers import TranslationParser, GridNameParser
from molgri.paths import PATH_OUTPUT_FULL_GRIDS
from molgri.space.utils import norm_per_axis, normalise_vectors, angle_between_vectors


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
        self.save_full_grid()

    def get_full_grid_name(self):
        o_name = self.o_rotations.get_standard_name(with_dim=False)
        b_name = self.b_rotations.get_standard_name(with_dim=False)
        return f"o_{o_name}_b_{b_name}_t_{self.t_grid.grid_hash}"

    def get_radii(self):
        return self.t_grid.get_trans_grid()

    def get_between_radii(self):
        radii = self.get_radii()
        increases = self.t_grid.get_increments() / 2
        return radii+increases

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

    def get_flat_position_grid(self):
        return self.get_position_grid().reshape((-1, 3))

    def save_full_grid(self):
        np.save(f"{PATH_OUTPUT_FULL_GRIDS}position_grid_{self.get_full_grid_name()}", self.get_position_grid())

    def get_voranoi_discretisation(self):

        def change_voranoi_radius(sv: SphericalVoronoi, new_radius):
            sv.radius = new_radius
            sv.vertices = normalise_vectors(sv.vertices, length=new_radius)
            sv.points = normalise_vectors(sv.points, length=new_radius)
            # important that it's a copy!
            return copy(sv)

        unit_sph_voranoi = self.o_rotations.get_spherical_voranoi_cells()
        between_radii = self.get_between_radii()
        radius_sph_voranoi = [change_voranoi_radius(unit_sph_voranoi, r) for r in between_radii]
        return radius_sph_voranoi

    def get_division_area(self, index_1: int, index_2: int):
        positions = self.get_flat_position_grid()
        all_sv = self.get_voranoi_discretisation()

        point_1 = positions[index_1]
        point_2 = positions[index_2]
        radius_1 = np.linalg.norm(point_1)
        radius_2 = np.linalg.norm(point_2)
        # TODO: determine if neighbours
        # if at same radius, you need the sideways area
        if np.allclose(radius_1, radius_2):
            selected_sv = None
            for i, sv in enumerate(all_sv):
                if sv.radius > radius_1:
                    selected_sv = i
                    break
            # if you don't finish with a break, you failed to find sv with sufficient radius
            else:
                raise OverflowError("At least one of the points has radius larger than largest SphericalVoranoi")
            # if this sv is not the smallest, calculate area as the difference of two circle cut-outs
            theta = angle_between_vectors(all_sv[i].#TODO: find vertices)
            r_larger = all_sv[i].radius
            if i != 0:
                r_smaller = all_sv[i-1].radius
            else:
                r_smaller = 0
            return theta / 2 * (r_larger**2 - r_smaller**2)
        elif radius_1 > radius_2:
            # find bottom area of point 1
        else:
            # find bottom area of point 2
