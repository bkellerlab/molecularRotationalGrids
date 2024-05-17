"""
Full discretisation of space in spherical layers.

The module fullgrid combines a linear translation grid with two spherical grids: discretisation of approach vectors
(3D sphere, orientation grid) and of internal rotations of the second body (4D half-sphere of quaternions, body grid).
The three grids are commonly referred to as t_grid, o_grid and b_grid.

Position grid is a product of t_grid and o_grid and represents a set of spherical points in 3D space that are repeated
at different radii. Based on a position grid it is possible to create a Voronoi discretisation of 3D space using
identical (up to radius) layers of Voronoi surfaces between layers of grid points and connecting them with ray points
from the origin to are vertices of Voronoi cells. Methods to calculate volumes, areas and distances between centers of
such discretisation sof position space are provided.

Objects:
 - FullGrid combines 3D sphere grid, 4D sphere grid and a translation grid
 - FullVoronoiGrid extends FullGrid with methods to evaluate distances/areas/volumes of cells
 - Point represents a single point in FullVoronoiGrid and implements helper functions like identifying vertices of cell
 - ConvergenceFullGridO provides plotting data by creating a range of FullGrids with different N of o_grid points
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
from numpy._typing import ArrayLike
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from scipy.constants import pi
from scipy.sparse import bmat, coo_array, diags
import pandas as pd

from molgri.constants import SMALL_NS
from molgri.space.rotobj import SphereGrid3DFactory, SphereGrid3Dim, SphereGrid4DFactory
from molgri.space.translations import TranslationParser, get_between_radii
from molgri.naming import GridNameParser
from molgri.space.utils import norm_per_axis, angle_between_vectors
#from molgri.space.voronoi import PositionVoronoi
from molgri.wrappers import save_or_use_saved


class FullGrid:

    """
    A combination object that enables work a combination of three grids (provided by their names)

    Args:
        b_grid_name: body rotation grid (a 4D sphere grid of quaternions used to generate orientations)
        o_grid_name: origin rotation grid (a 3D sphere grid used to create approach vectors)
        t_grid_name: translation grid (a linear grid used to determine distances to origin)
    """

    def __init__(self, b_grid_name: str, o_grid_name: str, t_grid_name: str, use_saved: bool = True):

        """
        Args:
            b_grid_name: of the form 'ico_17'
            o_grid_name: of the form 'cube4D_12'
            t_grid_name: of the form '[1, 3, 4.5]'
            use_saved: try to obtain saved data if possible
        """
        # this is supposed to be a scaling factor to make metric of SO(3) comparable to R3
        # will be applied as self.factor**3 to volumes, self.factor**2 to areas and self.factor to distances
        self.factor = 2
        self.b_grid_name = b_grid_name
        self.o_grid_name = o_grid_name
        self.t_grid_name = t_grid_name
        b_grid_name = GridNameParser(b_grid_name, "b")
        self.b_rotations = SphereGrid4DFactory.create(alg_name=b_grid_name.get_alg(), N=b_grid_name.get_N(),
                                                      use_saved=use_saved)
        self.position_grid = PositionGrid(o_grid_name=o_grid_name, t_grid_name=t_grid_name,
                                          use_saved=use_saved)
        self.use_saved = use_saved

    def __getattr__(self, name):
        """ Enable forwarding methods to self.position_grid, so that from FullGrid you can access all properties and
         methods of PositionGrid too."""
        return getattr(self.position_grid, name)

    def __len__(self):
        """The length of the full grid is a product of lengths of all sub-grids"""
        return self.b_rotations.get_N() * len(self.get_position_grid())

    def get_b_N(self) -> int:
        """
        Get number of points in quaternion grid.
        """
        return self.b_rotations.get_N()

    def get_o_N(self) -> int:
        """
        Get number of points in positions-on-a-sphere grid.
        """
        return self.o_rotations.get_N()

    def get_t_N(self) -> int:
        """
        Get number of points in translation grid.
        """
        return len(self.t_grid.get_trans_grid())

    def get_quaternion_index(self, full_grid_indices: NDArray = None) -> NDArray:
        """
        For given indices of full grid, return to which position in quaternion/b-grid they belong

        Args:
            full_grid_indices: if None, the entire fullgrid indices are used
        """
        # this returns 0, 1, 2, ... n_b, 0, 1, 2, ... n_b  ... <- repeated n_t*n_o times
        repeated_natural_num = np.tile(np.arange(self.get_b_N()), self.get_t_N()*self.get_o_N())
        if full_grid_indices is None:
            full_grid_indices = np.arange(len(self))
        return repeated_natural_num[full_grid_indices]

    def get_position_index(self, full_grid_indices: NDArray = None) -> NDArray:
        """
        For given indices of full grid, return to which position in position grid they belong.
        """
        # this returns 0, 0, 0  ... 0, 1, 1, 1, ... 1, 2 ... <- every number repeated n_b times
        repeated_natural_num = np.repeat(np.arange(self.get_t_N()*self.get_o_N()), self.get_b_N())
        if full_grid_indices is None:
            full_grid_indices = np.arange(len(self))
        return repeated_natural_num[full_grid_indices]


    def get_position_grid(self):
        return self.position_grid

    @save_or_use_saved
    def get_total_volumes(self):
        """
        Return 6D volume of a particular point obtained from multiplying position space region and orientation space
        region.

        """
        pos_volumes = self.get_position_grid().get_all_position_volumes()
        ori_volumes = self.b_rotations.get_spherical_voronoi().get_voronoi_volumes()

        all_volumes = []
        for o_rot in pos_volumes:
            for b_rot in ori_volumes:
                all_volumes.append(o_rot*(self.factor**3)*b_rot)
        return all_volumes

    def get_between_radii(self):
        return get_between_radii(self.t_grid.get_trans_grid())

    def get_adjacency_of_orientation_grid(self) -> coo_array:
        return self.b_rotations.get_voronoi_adjacency(only_upper=True, include_opposing_neighbours=True)

    def get_name(self) -> str:
        """Name that is appropriate for saving."""
        b_name = self.b_rotations.get_name(with_dim=False)
        return f"b_{b_name}_{self.get_position_grid().get_name()}"

    def get_body_rotations(self) -> Rotation:
        """Get a Rotation object (may encapsulate a list of rotations) from the body grid."""
        return Rotation.from_quat(self.b_rotations.get_grid_as_array())

    @save_or_use_saved
    def get_full_grid_as_array(self) -> NDArray:
        """
        Return an array of shape (n_t*n_o_n_b, 7) where for every sequential step of pt, the first 3 coordinates
        describe the position in position space, the last four give the orientation in a form of a quaternion.

        Firstly, for the first element of position grid, all orientations (quaternions) will be included. Then,
        we move onto the next element of position grid and again include all orientations. As a consequence:
            result[0:N_b], result[N_b:2*N_b] ... first three coordinates will always be the same vector from origin
            result[::N_t*N_o], result[1::N_t*N_o] ... will always be at the same last four coordinates (orientation)
        """
        result = np.full((len(self), 7), np.nan)
        position_grid = self.position_grid.get_position_grid_as_array()
        quaternions = self.b_rotations.get_grid_as_array(only_upper=True)
        current_index = 0
        for o_rot in position_grid:
            for b_rot in quaternions:
                # coordinates are (x, y, z, q0, q1, q2, q3)
                result[current_index][:3] = o_rot
                result[current_index][3:] = b_rot
                current_index += 1
        return result

    @save_or_use_saved
    def get_full_prefactors(self):
        """
        Get the sparse array A_ij/(V_i*h_ij)
        Returns:

        """
        all_volumes = np.array(self.get_total_volumes())
        all_surfaces = self.get_full_borders()
        all_distances = self.get_full_distances()
        prefactor_matrix = all_surfaces
        prefactor_matrix = prefactor_matrix.tocoo()
        prefactor_matrix.data /= all_distances.tocoo().data

        # Divide every row of transition_matrix with the corresponding volume
        # self.transition_matrix /= all_volumes[:, None]
        prefactor_matrix.data /= all_volumes[prefactor_matrix.row]
        return prefactor_matrix

    @save_or_use_saved
    def get_full_adjacency(self):
        return self._get_N_N(sel_property="adjacency")

    @save_or_use_saved
    def get_full_distances(self):
        return self._get_N_N(sel_property="center_distances")

    @save_or_use_saved
    def get_full_borders(self):
        return self._get_N_N(sel_property="border_len")

    def _get_N_N(self, sel_property="adjacency"):
        full_sequence = self.get_full_grid_as_array()
        n_total = len(full_sequence)
        n_o = self.o_rotations.get_N()
        n_b = self.b_rotations.get_N()
        n_t = self.t_grid.get_N_trans()

        position_adjacency = self.position_grid._get_N_N_position_array(sel_property=sel_property).toarray()
        if n_b > 1:
            orientation_adjacency = self.b_rotations.get_spherical_voronoi()._calculate_N_N_array(sel_property=sel_property)
        else:
            orientation_adjacency = coo_array([[False]], shape=(1, 1))

        row = []
        col = []
        values = []

        for i, line in enumerate(position_adjacency):
            for j, el in enumerate(line):
                if el:
                    for k in range(n_b):
                        row.append(n_b * i + k)
                        col.append(n_b * j + k)
                        values.append(el)

        if sel_property == "adjacency":
            dtype = bool
            my_factor = 1
        elif sel_property == "border_len":
            dtype = float
            my_factor = self.factor**2
        elif sel_property == "center_distances":
            dtype = float
            my_factor = self.factor
        else:
            dtype = float
            my_factor = 1

        same_orientation_neighbours = coo_array(([v*my_factor for v in values], (row, col)), shape=(n_total, n_total),
                                                dtype=dtype)

        # along the diagonal blocks of size n_o*n_t that are neighbours exactly if their quaternions are neighbours
        if n_t * n_o > 1:
            my_blocks = [orientation_adjacency]
            my_blocks.extend([None, ] * (n_t * n_o))
            my_blocks = my_blocks * (n_t * n_o)
            my_blocks = my_blocks[:-(n_t * n_o)]
            my_blocks = np.array(my_blocks, dtype=object)
            my_blocks = my_blocks.reshape((n_t * n_o), (n_t * n_o))
            same_position_neighbours = bmat(my_blocks, dtype=float) #block_array(my_blocks, dtype=dtype, format="coo")
        else:
            return coo_array(orientation_adjacency)
        all_neighbours = same_position_neighbours + same_orientation_neighbours
        return all_neighbours




class PositionGrid:

    def __init__(self, o_grid_name: str, t_grid_name: str, use_saved: bool = True):
        """
        This is derived from FullGrid and contains methods that are connected to position grid.
        """
        o_grid_name = GridNameParser(o_grid_name, "o")
        self.o_rotations = SphereGrid3DFactory.create(alg_name=o_grid_name.get_alg(), N=o_grid_name.get_N(),
                                                      use_saved=use_saved)
        self.o_positions = self.o_rotations.get_grid_as_array()
        self.t_grid = TranslationParser(t_grid_name)
        self.use_saved = use_saved

    def __len__(self):
        """The length of the full grid is a product of lengths of all sub-grids"""
        return self.o_rotations.get_N() * self.t_grid.get_N_trans()

    def get_t_grid(self) -> TranslationParser:
        return self.t_grid

    def get_o_grid(self) -> SphereGrid3Dim:
        return self.o_rotations

    def get_name(self) -> str:
        """Name that is appropriate for saving."""
        o_name = self.o_rotations.get_name(with_dim=False)
        return f"o_{o_name}_t_{self.t_grid.grid_hash}"

    def get_radii(self) -> NDArray:
        """
        Get the radii at which points are positioned. Result is in Angstroms.
        """
        return self.t_grid.get_trans_grid()

    @save_or_use_saved
    def get_position_grid_as_array(self) -> NDArray:
        """
        Get a position grid that is not structured layer-by-layer but is simply a 2D array of shape (N_t*N_o, 3) where
        N_t is the length of translation grid and N_o the length of orientation grid.
        """
        return _t_and_o_2_positions(o_property=self.get_o_grid().get_grid_as_array(only_upper=False),
                                         t_property=self.get_radii())

    def get_all_position_volumes(self) -> NDArray:
        # o grid has the option to get size of areas -> need to be divided by 3 and multiplied with radius^3 to get
        # volumes in the first shell, later shells need previous shells subtracted
        radius_above = get_between_radii(self.t_grid.get_trans_grid())
        radius_below = np.concatenate(([0, ], radius_above[:-1]))
        s_vor = self.get_o_grid().get_spherical_voronoi()
        area = s_vor.get_voronoi_volumes()
        # but of rad 2?????
        cumulative_volumes = (_t_and_o_2_positions(o_property=area/3, t_property=radius_above**3) -
                              _t_and_o_2_positions(o_property=area/3, t_property=radius_below**3))
        return cumulative_volumes


    def _get_N_N_position_array(self, sel_property="adjacency"):
        flat_pos_grid = self.get_position_grid_as_array()
        n_points = len(flat_pos_grid)  # equals n_o*n_t
        n_o = self.o_rotations.get_N()
        n_t = self.t_grid.get_N_trans()
        between_radii = get_between_radii(self.t_grid.get_trans_grid())

        # First you have neighbours that occur from being at subsequent radii and the same ray
        # Since the position grid has all orientations at first r, then all at second r ...
        # the points index_center and index_center+n_o will
        # always be neighbours, so we need the off-diagonals by n_o and -n_o
        # Most points have two neighbours this way, first and last layer have only one

        if sel_property == "adjacency":
            my_diags = (True,)
            my_dtype = bool
            # within a layer
            neig = self.o_rotations.get_voronoi_adjacency(only_upper=False, include_opposing_neighbours=False).toarray()
            # what to multipy with in higher layers
            multiply = np.ones(n_t)
        elif sel_property == "border_len":         # TODO TODO TODO
            my_diags  = []
            radius_1_areas = self.o_rotations.get_spherical_voronoi().get_voronoi_volumes()
            # last layer doesn't have a border up
            for layer_i, radius in enumerate(between_radii[:-1]):
                my_diags.extend(radius_1_areas * radius**2)  # todo: test
            my_dtype = float
            # neighbours in the same layer -> this is the arch above (or angle in radians since unit sphere)
            neig = self.o_rotations.get_cell_borders().toarray()
            # area is angle/2pi  * pi r**2
            subtracted_radii = np.array([0, *between_radii[:-1]])
            multiply = between_radii ** 2 / 2 - subtracted_radii**2/2  # need to subtract area of previous level
        elif sel_property == "center_distances":
            # n_o elements will have the same distance
            increments = self.t_grid.get_increments()
            my_diags = _t_and_o_2_positions(np.ones(len(self.o_rotations)), increments) # todo: test
            my_dtype = float
            # within a layer -> this is the arch at the level of points
            neig = self.o_rotations.get_center_distances(only_upper=False,
                                                                          include_opposing_neighbours=False).toarray()
            multiply = self.get_radii()
        else:
            raise ValueError(f"Not recognised argument property={sel_property}")

        same_ray_neighbours = diags(my_diags, offsets=n_o, shape=(n_points, n_points), dtype=my_dtype,
                                          format="coo")
        same_ray_neighbours += diags(my_diags, offsets=-n_o, shape=(n_points, n_points), dtype=my_dtype,
                                          format="coo")

        # Now we also want neighbours on the same level based on Voronoi discretisation
        # We first focus on the first n_o points since the set-up repeats at every radius

        # can't create Voronoi grid with <= 4 points, but then they are just all neighbours (except with itself)
        # if n_o <= 4:
        #     neig = np.ones((n_o, n_o), dtype=my_dtype) ^ np.eye(n_o, dtype=my_dtype)
        # else:
        # neig = self.o_rotations.get_voronoi_adjacency(only_upper=False, include_opposing_neighbours=False).toarray()

        # in case there are several translation distances, the array neig repeats along the diagonal n_t times
        if n_t > 1:
            my_blocks = [neig]
            my_blocks.extend([None,] * n_t)
            my_blocks = my_blocks * n_t
            my_blocks = my_blocks[:-n_t]
            my_blocks = np.array(my_blocks, dtype=object)
            my_blocks = my_blocks.reshape(n_t, n_t)
            same_radius_neighbours = bmat(my_blocks, dtype=float)

            for ind_n_t in range(n_t):
                smallest_row = ind_n_t*n_o <= same_radius_neighbours.row
                largest_row = same_radius_neighbours.row < (ind_n_t+1)*n_o
                smallest_column = ind_n_t * n_o <= same_radius_neighbours.col
                largest_column = same_radius_neighbours.col < (ind_n_t + 1) * n_o
                mask = smallest_row & largest_row & smallest_column & largest_column
                same_radius_neighbours.data[mask] *= multiply[ind_n_t]
        else:
            same_radius_neighbours = coo_array(neig) * multiply
        all_neighbours = same_ray_neighbours + same_radius_neighbours
        return all_neighbours

    def get_adjacency_of_position_grid(self) -> coo_array:
        """
        Get a position grid adjacency matrix of shape (n_t*n_o, n_t*n_o) based on the numbering of flat position matrix.
        Two indices of position matrix are neigbours if
            index_center) one is directly above the other, or
            ii) they are voronoi neighbours at the same radius

        Returns:
            a diagonally-symmetric boolean sparse matrix where entries are True if neighbours and False otherwise

        """
        return self._get_N_N_position_array(sel_property="adjacency")

    def get_borders_of_position_grid(self) -> coo_array:
        return self._get_N_N_position_array(sel_property="border_len")

    def get_distances_of_position_grid(self) -> coo_array:
        return self._get_N_N_position_array(sel_property="center_distances")



class ConvergenceFullGridO:
    
    """
    This object is used to study the convergence of properties as the number of points in the o_grid of FullGrid 
    changes. Mostly used to plot how voronoi volumes, calculation times ... converge.
    """

    def __init__(self, b_grid_name: str, t_grid_name: str,  o_alg_name: str, N_set: tuple = None,
                 use_saved: bool = False, **kwargs):
        if N_set is None:
            N_set = SMALL_NS
        self.N_set = N_set
        self.alg_name = o_alg_name
        self.use_saved = use_saved
        self.list_full_grids = self.create(b_grid_name=b_grid_name, t_grid_name=t_grid_name,  o_alg_name=o_alg_name,
                                           N_set=self.N_set, use_saved=use_saved, **kwargs)

    def get_name(self) -> str:
        """Get name for saving."""
        b_name = self.list_full_grids[0].b_rotations.get_name(with_dim=False)
        t_name = self.list_full_grids[0].t_grid.grid_hash
        return f"convergence_o_{self.alg_name}_b_{b_name}_t_{t_name}"

    @classmethod
    def create(cls,  b_grid_name: str, t_grid_name: str,  o_alg_name: str, N_set: tuple, **kwargs) -> list:
        """
        Build the FullGrid object for each of the given Ns in N_set. This is the create function that is only
        called by __init__.
        """
        list_full_grids = []
        for N in N_set:
            fg = FullGrid(b_grid_name=b_grid_name, o_grid_name=f"{o_alg_name}_{N}", t_grid_name=t_grid_name, **kwargs)
            list_full_grids.append(fg)
        return list_full_grids

    @save_or_use_saved
    def get_voronoi_volumes(self) -> pd.DataFrame:
        """
        Get all volumes of cells for all choices of self.N_set in order to create convergence plots.
        """
        data = []
        for N, fg in zip(self.N_set, self.list_full_grids):
            N_real = fg.o_rotations.get_N()
            vor_radius = list(fg.get_between_radii())
            vor_radius.insert(0, 0)
            vor_radius = np.array(vor_radius)
            fvg = fg.get_full_voronoi_grid()
            ideal_volumes = 4/3 * pi * (vor_radius[1:]**3 - vor_radius[:-1]**3) / N_real
            real_volumes = fvg.get_all_voronoi_volumes()
            for i, volume in enumerate(real_volumes):
                layer = i//N_real
                data.append([N_real, layer, ideal_volumes[i//N_real], volume])
        df = pd.DataFrame(data, columns=["N", "layer", "ideal volume", "Voronoi cell volume"])
        return df


def _t_and_o_2_positions(o_property, t_property):
    """
    Helper function to systematically combine t_grid and o_grid. We always do this in the same way:
        - firstly, all orientations at first distance
        - then all orientations at second distance
        - ....

    Outputs an array of len n_o*n_t, can have shape 1 or higher depending on the property

    Returns:

    """
    n_t = len(t_property)
    n_o = len(o_property)


    # eg coordinates
    if len(o_property.shape) > 1:
        tiled_o = np.tile(o_property, reps=(n_t, 1))
        tiled_t = np.repeat(t_property, n_o)[:, np.newaxis]
        result = tiled_o * tiled_t
    else:
        tiled_o = np.tile(o_property, reps=n_t)
        tiled_t = np.repeat(t_property, n_o)[np.newaxis, :]
        result = (tiled_o * tiled_t)[0]
    assert len(result) == n_o*n_t
    return result




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    o_grid = 1000
    HALF_HYPERSPHERE_SURFACE = pi ** 2


    available_fgs = [FullGrid(o_grid_name=f"{o_grid}", b_grid_name="1", t_grid_name="[0.5]", use_saved=True),
                     FullGrid(o_grid_name="1", b_grid_name=f"{o_grid}", t_grid_name="[0.5]", use_saved=True),
                     #FullGrid(o_grid_name="1", b_grid_name="1", t_grid_name="linspace(0.1, 1.1, 1000)",
                     # use_saved=True)
                     ]

    fig, ax = plt.subplots(len(available_fgs), 3, figsize=(15, len(available_fgs) * 5))
    for i, my_full_grid in enumerate(available_fgs):
        all_volumes = my_full_grid.get_total_volumes()
        if i==0 or i==1:
            ideal_volume_o_total = 4/ 3 * pi * my_full_grid.get_between_radii()[0]**3 *HALF_HYPERSPHERE_SURFACE
            ideal_volume_o = ideal_volume_o_total / o_grid #HALF_HYPERSPHERE_SURFACE *
            ax[i][0].vlines(x=ideal_volume_o, ymin=0, ymax=0.25, colors="red")
        sns.histplot(all_volumes, ax=ax[i][0], stat="probability")
        all_borders = my_full_grid.get_full_borders()
        #print(my_full_grid.position_grid.get_borders_of_position_grid())
        sns.histplot(all_borders.data, ax=ax[i][1], stat="probability")
        all_dist = my_full_grid.get_full_distances()
        sns.histplot(all_dist.data, ax=ax[i][2], bins=30, stat="probability")
        ax[0][0].set_title("Volumes")
        ax[0][1].set_title("Border areas")
        ax[0][2].set_title("Center distances")
    plt.show()




