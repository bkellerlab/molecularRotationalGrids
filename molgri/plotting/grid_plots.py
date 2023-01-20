from typing import List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.constants import pi
from scipy.spatial import geometric_slerp

from molgri.constants import UNIQUE_TOL, DIM_SQUARE, DIM_LANDSCAPE
from molgri.molecules.parsers import XVGParser, FullGridNameParser, PtParser
from molgri.paths import PATH_OUTPUT_FULL_GRIDS, PATH_INPUT_BASEGRO, PATH_OUTPUT_PT, PATH_INPUT_ENERGIES, \
    PATH_OUTPUT_PLOTS
from molgri.plotting.abstract import Plot3D, AbstractPlot, AbstractMultiPlot
from molgri.plotting.analysis_plots import points_up_to_Ns, test_or_create_Ns
from molgri.space.analysis import vector_within_alpha
from molgri.space.cells import voranoi_surfaces_on_stacked_spheres, voranoi_surfaces_on_sphere
from molgri.space.polytopes import Polytope, IcosahedronPolytope, Cube3DPolytope
from molgri.space.rotobj import build_grid_from_name
from molgri.space.utils import norm_per_axis, normalise_vectors, cart2sphA


class GridPlot(Plot3D):

    def __init__(self, data_name, *, style_type: list = None, plot_type: str = "grid", **kwargs):
        """
        This class is used for plots and animations of grids.

        Args:
            data_name: in the form algorithm_N e.g. randomQ_60
            style_type: a list of style properties like ['empty', 'talk', 'half_dark']
            plot_type: change this if you need unique name for plots with same data_name
            **kwargs:
        """
        if style_type is None:
            style_type = ["talk"]
        super().__init__(data_name, style_type=style_type, plot_type=plot_type, **kwargs)
        self.grid = self._prepare_data()

    def _prepare_data(self) -> np.ndarray:
        my_grid = build_grid_from_name(self.data_name, use_saved=False, print_warnings=False).get_grid()
        return my_grid

    def _plot_data(self, color="black", s=30, **kwargs):
        self.sc = self.ax.scatter(*self.grid.T, color=color, s=s)

    def create(self, animate_seq=False, **kwargs):
        if "empty" in self.style_type:
            pad_inches = -0.2
        else:
            pad_inches = 0
        x_max_limit = kwargs.pop("x_max_limit", 1)
        y_max_limit = kwargs.pop("y_max_limit", 1)
        z_max_limit = kwargs.pop("z_max_limit", 1)
        super(GridPlot, self).create(equalize=True, x_max_limit=x_max_limit, y_max_limit=y_max_limit,
                                     z_max_limit=z_max_limit, pad_inches=pad_inches, azim=30, elev=10, projection="3d",
                                     **kwargs)
        if animate_seq:
            self.animate_grid_sequence()

    def animate_grid_sequence(self):
        """
        Animate how a grid is constructed - how each individual point is added.

        WARNING - I am not sure that this method always displays correct order/depth coloring - mathplotlib
        is not the most reliable tool for 3d plots and it may change the plotting order for rendering some
        points above others!
        """

        def update(i):
            current_colors = np.concatenate([facecolors_before[:i], all_white[i:]])
            self.sc.set_facecolors(current_colors)
            self.sc.set_edgecolors(current_colors)
            return self.sc,

        facecolors_before = self.sc.get_facecolors()
        shape_colors = facecolors_before.shape
        all_white = np.zeros(shape_colors)

        self.ax.view_init(elev=10, azim=30)
        ani = FuncAnimation(self.fig, func=update, frames=len(facecolors_before), interval=5, repeat=False)
        writergif = PillowWriter(fps=3, bitrate=-1)
        # noinspection PyTypeChecker
        ani.save(f"{self.ani_path}{self.data_name}_{self.plot_type}_ord.gif", writer=writergif, dpi=400)


class PositionGridPlot(GridPlot):

    def __init__(self, data_name, style_type=None, cell_lines=False, plot_type="positions", **kwargs):
        self.cell_lines = cell_lines
        super().__init__(data_name, style_type=style_type, plot_type=plot_type, **kwargs)

    def _prepare_data(self) -> np.ndarray:
        points = np.load(f"{PATH_OUTPUT_FULL_GRIDS}{self.data_name}.npy")
        return points

    def _plot_data(self, color="black", **kwargs):
        points = self._prepare_data()
        points = np.swapaxes(points, 0, 1)
        for i, point_set in enumerate(points):
            self.sc = self.ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2], c=color)
        if self.cell_lines:
            self._draw_voranoi_cells(points)

    def create(self, **kwargs):
        max_norm = np.max(norm_per_axis(self.grid))
        super(PositionGridPlot, self).create(x_max_limit=max_norm, y_max_limit=max_norm, z_max_limit=max_norm, **kwargs)

    def _draw_voranoi_cells(self, points):
        svs = voranoi_surfaces_on_stacked_spheres(points)
        for i, sv in enumerate(svs):
            sv.sort_vertices_of_regions()
            t_vals = np.linspace(0, 1, 2000)
            # plot Voronoi vertices
            self.ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g')
            # indicate Voronoi regions (as Euclidean polygons)
            for region in sv.regions:
                n = len(region)
                for j in range(n):
                    start = sv.vertices[region][j]
                    end = sv.vertices[region][(j + 1) % n]
                    norm = np.linalg.norm(start)
                    result = geometric_slerp(normalise_vectors(start), normalise_vectors(end), t_vals)
                    self.ax.plot(norm * result[..., 0], norm * result[..., 1], norm * result[..., 2], c='k')


class TrajectoryEnergyPlot(Plot3D):

    def __init__(self, data_name: str, plot_type="trajectory", plot_points=False, plot_surfaces=True,
                 selected_Ns=None, **kwargs):
        self.energies = None
        self.property = "Trajectory positions"
        self.unit = r'$\AA$'
        self.selected_Ns = selected_Ns
        self.N_index = 0
        self.plot_points = plot_points
        self.plot_surfaces = plot_surfaces
        super().__init__(data_name, plot_type=plot_type, **kwargs)

    def add_energy_information(self, path_xvg_file, property_name="Potential"):
        self.plot_type += "_energies"
        file_parser = XVGParser(path_xvg_file)
        self.property, property_index = file_parser.get_column_index_by_name(column_label=property_name)
        self.unit = file_parser.get_y_unit()
        self.energies = file_parser.all_values[:, property_index]

    def _prepare_data(self) -> pd.DataFrame:
        try:
            split_name = self.data_name.split("_")
            path_m1 = f"{PATH_INPUT_BASEGRO}{split_name[0]}"
            path_m2 = f"{PATH_INPUT_BASEGRO}{split_name[1]}"
            try:
                gnp = FullGridNameParser("_".join(split_name[2:]))
                num_b = gnp.get_num_b_rot()
                num_o = gnp.get_num_o_rot()
            except AttributeError:
                print("Warning! Trajectory name not in standard format. Will not be able to perform convergence tests.")
                num_b = 1
                num_o = 1
        except AttributeError:
            raise ValueError("Cannot use the name of the XVG file to find the corresponding trajectory. "
                             "Please rename the XVG file to the same name as the XTC file.")
        path_topology = f"{PATH_OUTPUT_PT}{self.data_name}.gro"
        path_trajectory = f"{PATH_OUTPUT_PT}{self.data_name}.xtc"
        my_parser = PtParser(path_m1, path_m2, path_topology, path_trajectory)
        my_data = []
        for i, molecules in enumerate(my_parser.generate_frame_as_molecule()):
            mol1, mol2 = molecules
            com = mol2.get_center_of_mass()
            try:
                current_E = self.energies[i]
            except TypeError:
                current_E = 0
            my_data.append([*np.round(com), current_E])
        my_df = pd.DataFrame(my_data, columns=["x", "y", "z", f"{self.property} {self.unit}"])
        # if multiple shells present, warn and only use the closest one.
        num_t = len(my_df) // num_b // num_o
        if num_t != 1:
            print("Warning! The pseudotrajectory has multiple shells/translation distances. 2D/3D visualisation is "
                  "most suitable for single-shell visualisations. Only the data from the first shell will be used "
                  "in the visualisation.")
            my_df = get_only_one_translation_distance(my_df, num_t)
        # of all points with the same position, select only the orientation with the smallest energy
        df_extract = groupby_min_body_energy(my_df, target_column=f"{self.property} {self.unit}", N_b=num_b)
        # now potentially reduce the number of orientations tested
        self.selected_Ns = test_or_create_Ns(num_o, self.selected_Ns)
        points_up_to_Ns(df_extract, self.selected_Ns, target_column=f"{self.property} {self.unit}")
        return df_extract

    def _plot_data(self, **kwargs):
        my_df = self._prepare_data()
        # determine min and max of the color dimension
        up_to_N_elements = my_df[my_df[f"{self.selected_Ns[self.N_index]}"].notnull()]
        all_positions = up_to_N_elements[["x", "y", "z"]].to_numpy()
        all_energies = up_to_N_elements[f"{self.selected_Ns[self.N_index]}"].to_numpy()
        # sort out what not unique
        _, indices = np.unique(all_positions.round(UNIQUE_TOL), axis=0, return_index=True)
        all_positions = np.array([all_positions[index] for index in sorted(indices)])
        all_energies = np.array([all_energies[index] for index in sorted(indices)])
        # TODO: enable colorbar even if not plotting points
        if self.energies is None:
            self.ax.scatter(*all_positions.T, c="black")
        else:
            if self.plot_surfaces:
                try:
                    self._draw_voranoi_cells(all_positions, all_energies)
                except AssertionError:
                    print("Warning! Sperichal Voranoi cells plot could not be produced. Likely all points are "
                          "not at the same radius. Will create a scatterplot instead.")
                    self.plot_points = True
            cmap = ListedColormap((sns.color_palette("coolwarm", 256).as_hex()))
            im = self.ax.scatter(*all_positions.T, c=all_energies, cmap=cmap)
            # cbar = self.fig.colorbar(im, ax=self.ax)
            # cbar.set_label(f"{self.property} {self.unit}", fontsize=20)
            if not self.plot_points:
                im.set_visible(False)
            self.ax.set_title(r"$N_{rot}$ " + f"= {self.selected_Ns[self.N_index]}", fontsize=30)
            self.ax.view_init(elev=10, azim=60)

    def _draw_voranoi_cells(self, points, colors):
        sv = voranoi_surfaces_on_sphere(points)
        norm = matplotlib.colors.Normalize(vmin=colors.min(), vmax=colors.max())
        cmap = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        cmap.set_array([])
        fcolors = cmap.to_rgba(colors)
        sv.sort_vertices_of_regions()
        for n in range(0, len(sv.regions)):
            region = sv.regions[n]
            polygon = Poly3DCollection([sv.vertices[region]], alpha=1)
            polygon.set_color(fcolors[n])
            self.ax.add_collection3d(polygon)

    def create(self, *args, title=None, **kwargs):
        if title is None:
            title = f"{self.property} {self.unit}"
        super().create(*args, equalize=True, title=title, **kwargs)


class HammerProjectionTrajectory(TrajectoryEnergyPlot, AbstractPlot):

    def __init__(self, data_name: str, plot_type="hammer", figsize=DIM_SQUARE, **kwargs):
        sns.set_context("talk")
        super().__init__(data_name, plot_type=plot_type, plot_surfaces=False, plot_points=True, figsize=figsize,
                         **kwargs)

    def _plot_data(self, **kwargs):

        my_df = self._prepare_data()
        # determine min and max of the color dimension
        up_to_N_elements = my_df[my_df[f"{self.selected_Ns[self.N_index]}"].notnull()]
        all_positions = up_to_N_elements[["x", "y", "z"]].to_numpy()
        all_energies = up_to_N_elements[f"{self.selected_Ns[self.N_index]}"].to_numpy()
        # sort out what not unique
        # _, indices = np.unique(all_positions.round(UNIQUE_TOL), axis=0, return_index=True)
        # all_positions = np.array([all_positions[index] for index in sorted(indices)])
        # all_energies = np.array([all_energies[index] for index in sorted(indices)])
        all_positions = cart2sphA(all_positions)
        x = all_positions[:, 2]
        y = all_positions[:, 1]
        if self.energies is None:
            self.ax.scatter(*all_positions.T, c="black")
        else:
            cmap = ListedColormap((sns.color_palette("coolwarm", 256).as_hex()))
            im = self.ax.scatter(x, y, c=all_energies, cmap=cmap)
        self.ax.set_xticklabels([])

    def create(self, *args, **kwargs):
        AbstractPlot.create(self, *args, projection="hammer", **kwargs)


class TrajectoryEnergyMultiPlot(AbstractMultiPlot):

    def __init__(self, list_plots: List[TrajectoryEnergyPlot], figsize=None, **kwargs):
        if figsize is None:
            figsize = (DIM_LANDSCAPE[1]*4, DIM_LANDSCAPE[1])
        super().__init__(list_plots, figsize=figsize, **kwargs)

    def create(self, *args, **kwargs):
        super().create(*args, projection="3d", **kwargs)


class HammerProjectionMultiPlot(AbstractMultiPlot):

    def __init__(self, list_plots: List[HammerProjectionTrajectory], plot_type="hammer", figsize=None,
                 n_rows=1, n_columns=5, **kwargs):
        if figsize is None:
            figsize = (DIM_LANDSCAPE[1]*n_columns, DIM_LANDSCAPE[1]*n_rows)
        super().__init__(list_plots, plot_type=plot_type, n_rows=n_rows, n_columns=n_columns, figsize=figsize)

    def create(self, *args, projection="hammer", **kwargs):
        super().create(*args, projection=projection, **kwargs)


class GridColoredWithAlphaPlot(GridPlot):
    def __init__(self, data_name, vector: np.ndarray, alpha_set: list, plot_type: str = "colorful_grid", **kwargs):
        super().__init__(data_name, plot_type=plot_type, **kwargs)
        self.alpha_central_vector = vector
        self.alpha_set = alpha_set
        self.alpha_set.sort()
        self.alpha_set.append(pi)

    def _plot_data(self, color=None, **kwargs):
        # plot vector
        self.ax.scatter(*self.alpha_central_vector, marker="x", c="k", s=30)
        # determine color palette
        cp = sns.color_palette("Spectral", n_colors=len(self.alpha_set))
        # sort points which point in which alpha area
        already_plotted = []
        for i, alpha in enumerate(self.alpha_set):
            possible_points = np.array([vec for vec in self.grid if tuple(vec) not in already_plotted])
            within_alpha = vector_within_alpha(self.alpha_central_vector, possible_points, alpha)
            selected_points = [tuple(vec) for i, vec in enumerate(possible_points) if within_alpha[i]]
            array_sel_points = np.array(selected_points)
            self.sc = self.ax.scatter(*array_sel_points.T, color=cp[i], s=30)  # , s=4)
            already_plotted.extend(selected_points)
        self.ax.view_init(elev=10, azim=30)


def create_trajectory_energy_multiplot(data_name, Ns=None, animate_rot=False):
    list_single_plots = []
    max_index = 5 if Ns is None else len(Ns)
    for i in range(max_index):
        tep = TrajectoryEnergyPlot(data_name, plot_points=False, plot_surfaces=True, selected_Ns=Ns, style_type=["talk"])
        tep.N_index = i
        tep.add_energy_information(f"{PATH_INPUT_ENERGIES}{data_name}")
        list_single_plots.append(tep)
    TrajectoryEnergyMultiPlot(list_single_plots, n_columns=max_index, n_rows=1).create_and_save(animate_rot=animate_rot,
                                                                                                elev=10, azim=60)


def create_hammer_multiplot(data_name, Ns=None):
    list_single_plots = []
    max_index = 5 if Ns is None else len(Ns)
    for i in range(max_index):
        tep = HammerProjectionTrajectory(data_name, selected_Ns=Ns, style_type=["talk"])
        tep.N_index = i
        tep.add_energy_information(f"{PATH_INPUT_ENERGIES}{data_name}")
        list_single_plots.append(tep)
    HammerProjectionMultiPlot(list_single_plots, n_columns=max_index, n_rows=1).create_and_save()


class PolytopePlot(Plot3D):

    def __init__(self, data_name: str, num_divisions=3, faces=None, projection=False, **kwargs):
        """
        Plotting (some faces of) polyhedra, demonstrating the subdivision of faces with points.

        Args:
            data_name:
            num_divisions: how many levels of faces subdivisions should be drawn
            faces: a set of indices indicating which faces to draw
            projection: if True display points projected on a sphere, not on faces
            **kwargs:
        """
        self.num_divisions = num_divisions
        self.faces = faces
        self.projection = projection
        plot_type = f"polytope_{num_divisions}"
        super().__init__(data_name, fig_path=PATH_OUTPUT_PLOTS, plot_type=plot_type, style_type=["empty"],
                         **kwargs)

    def _prepare_data(self) -> Polytope:
        if self.data_name == "ico":
            ico = IcosahedronPolytope()
        else:
            ico = Cube3DPolytope()
        for n in range(self.num_divisions):
            ico.divide_edges()
        return ico

    def _plot_data(self, **kwargs):
        if self.faces is None and self.data_name == "ico":
            self.faces = {12}
        elif self.faces is None:
            self.faces = {3}
        ico = self._prepare_data()
        ico.plot_points(self.ax, select_faces=self.faces, projection=self.projection)
        ico.plot_edges(self.ax, select_faces=self.faces)


def groupby_min_body_energy(df: pd.DataFrame, target_column: str, N_b: int) -> pd.DataFrame:
    """
    Take a dataframe with positions and energies and return only one row per COM position of the second molecule,
    namely the one with lowest energy.

    Args:
        df: dataframe resulting from a Pseudotrajectory with typical looping over:
            rotations of origin
                rotations of body
                    translations must be already filtered out by this point!
        target_column: name of the column in which energy values are found
        N_b: number of rotations around body for this PT.

    Returns:
        a DataFrame with a number of rows equal original_num_of_rows // N_b
    """
    # in case that translations have been removed, the index needs to be re-set
    df.reset_index(inplace=True, drop=True)
    start_len = len(df)
    new_df = df.loc[df.groupby(df.index // N_b)[target_column].idxmin()]
    assert len(new_df) == start_len // N_b
    return new_df


def get_only_one_translation_distance(df: pd.DataFrame, N_t: int, distance_index=0) -> pd.DataFrame:
    start_len = len(df)
    assert distance_index < N_t, f"Only {N_t} different distances available, " \
                                 f"cannot get the distance with index {distance_index}"
    new_df = df.iloc[range(0, len(df), N_t)]
    assert len(new_df) == start_len // N_t
    return new_df
