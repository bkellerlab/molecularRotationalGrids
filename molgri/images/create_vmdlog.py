"""
Vmdlogs are instruction files provided to VMD. Here we create vmdlogs that help display specific structures of the
(pseudo)trajectories, often the ones corresponding to eigenvector extremes, minimal energies etc.
"""
import numpy as np
from numpy.typing import NDArray, ArrayLike

VMD_COLOR_DICT = {"black": 16, "yellow": 4, "orange": 3, "green": 7, "blue": 0, "cyan": 10, "purple": 11,
              "gray": 2, "pink": 9, "red": 1, "magenta": 27, "silver": 6, "gold": 5, "lime": 12}

VMD_BOND_TYPE_DICT = {"CPK": "CPK 1.000000 0.300000 10.000000 10.000000",
                      "VDW": "VDW 1.000000 12.000000",
                      "Licorice": "Licorice 0.300000 10.000000 10.000000",
                      "DynamicBonds": "DynamicBonds 1.600000 0.300000 6.000000",
                      "Cartoon": "NewCartoon"}

class VMDCreator:
    """
    From pieces of strings build a long vmdlog file that can be used inside vmd to automatically execute commands
    like: create a CPK representation, rotate the molecule, render a picture, change the color, render again etc.
    """

    def __init__(self, index_first_molecule: str = "all", index_second_molecule: str = "all", is_protein: bool = False):
        """
        Because we specialize in two-molecule systems we need to know how to find the first and the second molecule,
        because we will typically represent each of them individually.

        Args:
            index_first_molecule (str): command that selects the first molecule, like 'index < 3'
            index_second_molecule (str): command that selects the second molecule, like 'index >= 3'
            is_protein (bool): use True to automatically select more protein-like representations (secondary
            structure rather than ball and stick)
        """
        self.index_first_molecule = index_first_molecule
        self.index_second_molecule = index_second_molecule
        self.translations_rotations_script = None
        self.is_protein = is_protein

        if is_protein:
            self.default_coloring_method = "Structure"
            self.default_drawing_method = "NewCartoon"
        else:
            self.default_coloring_method = "Type"
            self.default_drawing_method = "DynamicBonds 1.600000 0.300000 6.000000"

        self.num_representations = 0
        self._start_new_file()
    
    def _start_new_file(self):
        """
        Just a helper for starting a new string and adding default settings.
        """
        self.total_file_text = ""
        self._add_pretty_plot_settings()

    def write_text_to_file(self, output_file_path: str) -> None:
        """
        Because basically every method here writes to internal self.total_file_text my_property, in the end we just
        need to transfer it to a file.
        """
        with open(output_file_path, "w") as f:
            f.write(self.total_file_text)

        # in case we want to build another file after this one
        self._start_new_file()

    def _add_pbc_box(self):
        self.total_file_text += """pbc box\n"""

    def _add_pretty_plot_settings(self) -> None:
        """
        Delete the initial default representation. Don't add any representations yet.
        Additionally, some nice settings so that pictures look good.
        """

        self.total_file_text += f"""
history keep 0
mol delrep 0 0
color Display Background white
axes location Off
mol material Opaque
display shadows on
display ambientocclusion on
material add copy AOChalky
material change shininess Material22 0.000000
display depthcue off
display projection Orthographic
color Type C gray
color Name C gray
display nearclip 0.001
display farclip 1000.0
display resize 1800 1800
"""

    def add_box(self, length_x, length_y, length_z) -> None:
        self.total_file_text += f"""
set mol top
set x0 0.0
set y0 0.0
set z0 0.0
set x1 {str(float(length_x))}
set y1 {str(float(length_y))}
set z1 {str(float(length_z))}

# define corners
set c000 [list $x0 $y0 $z0]
set c100 [list $x1 $y0 $z0]
set c010 [list $x0 $y1 $z0]
set c110 [list $x1 $y1 $z0]
set c001 [list $x0 $y0 $z1]
set c101 [list $x1 $y0 $z1]
set c011 [list $x0 $y1 $z1]
set c111 [list $x1 $y1 $z1]

# draw edges
graphics $mol cylinder $c000 $c100 radius 0.1 resolution 20
graphics $mol cylinder $c000 $c010 radius 0.1 resolution 20
graphics $mol cylinder $c000 $c001 radius 0.1 resolution 20

graphics $mol cylinder $c100 $c110 radius 0.1 resolution 20
graphics $mol cylinder $c100 $c101 radius 0.1 resolution 20

graphics $mol cylinder $c010 $c110 radius 0.1 resolution 20
graphics $mol cylinder $c010 $c011 radius 0.1 resolution 20

graphics $mol cylinder $c001 $c101 radius 0.1 resolution 20
graphics $mol cylinder $c001 $c011 radius 0.1 resolution 20

graphics $mol cylinder $c110 $c111 radius 0.1 resolution 20
graphics $mol cylinder $c101 $c111 radius 0.1 resolution 20
graphics $mol cylinder $c011 $c111 radius 0.1 resolution 20
"""""

    def _zoom_on_box(self, length_x, length_y, zoom_level=1):
        self.total_file_text += f"""
set xmin 0
set ymin 0
set xmax {length_x}
set ymax  {length_y}

set cx [expr {{($xmin + $xmax)/2.0}}]
set cy [expr {{($ymin + $ymax)/2.0}}]

molinfo top set center [list $cx $cy 0.0]

set dx [expr {{$xmax - $xmin}}]
set dy [expr {{$ymax - $ymin}}]
set maxxy [expr {{max($dx, $dy)}}]

scale to [expr {{ {zoom_level} * 0.1 / $maxxy}}]
"""


    def _add_representation(self, first_molecule: bool = False, second_molecule: bool = True, coloring: str = None,
                            color: str = None, representation: str = None, trajectory_frames: ArrayLike = None,
                            periodic: str = None):
        """
        Use this to add a new representation of molecule 1, molecule 2 or both.

        Args:
            first_molecule (bool): True to show molecule 1 in this representation
            second_molecule (bool): True to show molecule 2 in this representation
            coloring (str): keyword understood by VMD how to group for coloring like 'Name', 'Type',
                'SecondaryStructure' or 'ColorId'
            color (str): only used if coloring='ColorId' is selected, color is translated to VMD's internal color ID
            representation (str): keyword understood by VMD how to show structure like 'CPK', 'VDW',
                'NewCartoon' ...
            trajectory_frames (ArrayLike): which frames to use, can be 'now', can be a single frame number (int) or a
                list-like object of multiple frame numbers
        """
        if coloring is None:
            coloring = self.default_coloring_method
        if coloring == "ColorID":
            if color is None:
                color = "black"
            # if the coloring type is ColorID, we need an additional parameter that specifies the color
            coloring = f"{coloring} {VMD_COLOR_DICT[color]}"
        if representation is None:
            representation = self.default_drawing_method

        if first_molecule and not second_molecule:
            molecular_index = self.index_first_molecule
        elif second_molecule and not first_molecule:
            molecular_index = self.index_second_molecule
        elif first_molecule and second_molecule:
            molecular_index = "all"
        else:
            raise ValueError("Trying to add a molecule but first_molecule=False and second_molecule=False.")

        # because trajectory frames may be an int, a string, or a list-like object we ned to pre-process it
        if isinstance(trajectory_frames, np.integer) or isinstance(trajectory_frames, int):
            trajectory_frames_as_str = str(trajectory_frames)
        elif isinstance(trajectory_frames, list):
            trajectory_frames_as_str = ', '.join(map(str, [int(x) for x in trajectory_frames]))
        elif isinstance(trajectory_frames, np.ndarray):
            trajectory_frames_as_str = ', '.join(map(str, trajectory_frames.flatten().astype(int)))
        elif isinstance(trajectory_frames, str):
            trajectory_frames_as_str = trajectory_frames
        else:
            raise ValueError(f"Trajectory frame indices of type {type(trajectory_frames)} cannot be read.")

        self.total_file_text += f"""
mol addrep 0
mol modstyle {self.num_representations} 0 {representation}
mol modselect {self.num_representations} 0 {molecular_index}
mol modcolor {self.num_representations} 0 {coloring}
mol drawframes 0 {self.num_representations} {{ {trajectory_frames_as_str} }}
        """

        if periodic is not None:
            self.total_file_text += f"""
mol showperiodic 0 {self.num_representations} {periodic}
mol numperiodic 0 {self.num_representations} 1
            """

        self.num_representations += 1

    def _add_dot(self, coordinate, radius=0.2):
        """
        Simply adds a small sphere at chosen position, e.g. for plotting grids.
        """
        self.total_file_text += f"""
graphics top sphere {{ {coordinate[0]} {coordinate[1]} {coordinate[2]} }} radius {radius} resolution 12
"""

    def add_grid(self, coordinates, radius=0.2):
        """
        Plotting position grid given an array of coordinates.
        """
        for coordinate in coordinates:
            self._add_dot(coordinate, radius)

    def _render_representations(self, list_representation_indices: ArrayLike, plot_path: str) -> None: 
        """
        Show representations that are in the list_representation_indices and hide all others. Save the rendered plot
        to plot_path.

        Args:
            list_representation_indices (ArrayLike): provide a list-like object of integers, each pointing to an (
            already added) representation we want to use for this plot
            plot_path (str): path to the plot that will be created
        """
        # show and hide as needed
        for repr_index in list_representation_indices:
            self._show_representation(repr_index)
        not_on_list = set(range(self.num_representations)) - set(list_representation_indices)
        for repr_index in not_on_list:
            self._hide_representation(repr_index)

        # render
        self.total_file_text += f"render TachyonInternal {plot_path}"


    def _add_rotations_translations(self):
        """
        If the script was loaded before, it will be used, else nothing happens.
        """
        if self.translations_rotations_script:
            with open(self.translations_rotations_script, "r") as f:
                contents = f.read()
            self.total_file_text += contents

    def _show_representation(self, representation_index: int) -> None:
        """
        Helper function, show a particular representation among already created ones. If not hidden does nothing.
        
        Args:
            representation_index (int): which representation to show.
        """
        self.total_file_text += f"\nmol showrep 0 {representation_index} 1\n"

    def _hide_representation(self, representation_index: int) -> None:
        """
        Helper function, hide a particular representation among already created ones. If already hidden does nothing.

        Args:
            representation_index (int): which representation to hide.
        """
        self.total_file_text += f"\nmol showrep 0 {representation_index} 0\n"

    def load_translation_rotation_script(self, path_translation_rotation_script: str = None) -> None:
        """
        Optionally, if there is a (manually created) sequence of translations/rotations/scaling in VMD format, you can 
        load it here. Useful to create multiple plots in exactly same orientation. If not provided, the default VMD 
        orientation is used.
        
        Args:
            path_translation_rotation_script (str): the path to the VMD script
        """
        self.translations_rotations_script = path_translation_rotation_script
        
    def prepare_frame_script(self, vmd_name: str, plot_name: str, num_frames: int, box_limits: list,
                             draw_m1: str, draw_m2: str, draw_rectangular_box: bool = True,
                             gridpoints: NDArray = None, zoom_level: int = 1, translation_rotation_script: str = None):
        """
        Plot one or multiple (overlapping) frames. Assumes the script will be run with additional "structure" first 
        frame which should be ignored (used for the reference so that the initial zoom level is always the same) and 
        one or more additional frames that should all be plotted.
        """
        self._start_new_file()

        if translation_rotation_script:
            self.load_translation_rotation_script(translation_rotation_script)
        if draw_m1 != "None":
            # plot only one non-zero frame since they are all the same
            self._add_representation(first_molecule=True, second_molecule=False, periodic="xyzXYZ",
                                       representation=VMD_BOND_TYPE_DICT[draw_m1], trajectory_frames=[1]) #
        if draw_m2 != "None":
            # plot all provided frames except 0
            self._add_representation(first_molecule=False, second_molecule=True, periodic="xyzXYZ",
                                       representation=VMD_BOND_TYPE_DICT[draw_m2], trajectory_frames=list(range(1,
                                                                                                                num_frames+1)))
        if draw_rectangular_box:
            self.add_box(*box_limits)
        if gridpoints is not None:
            self.add_grid(gridpoints)
        if translation_rotation_script:
            self._add_rotations_translations()
        if zoom_level > 0:
            self._zoom_on_box(box_limits[0], box_limits[1], zoom_level)
        # we only have two representations even if the second one potentially uses multiple frames
        self._render_representations([0, 1], plot_path=plot_name)
        self.write_text_to_file(vmd_name)

    def prepare_eigenvector_script(self, num_red: int, num_blue: int, vmd_name: str, plot_name: str, box_limits: list,
                             draw_m1: bool = True, draw_m2: bool = True, draw_rectangular_box: bool = True,
                             gridpoints: NDArray = None, zoom_level: int = 1, translation_rotation_script: str = None) -> None:
        """
        Plot multiple overlapping frames. Assumes the script will be run with:
        vmd structure_file frame_red1 ... frame_red_Nred frame_blue1 ... frame_blueNblue

        where the structure file should be ignored (used only as a reference so that the initial zoom level is always
        the same). The rest of the frames will be plotted - the first num_red in red and after that num_blue in blue
        """
        self._start_new_file()
        if translation_rotation_script:
            self.load_translation_rotation_script(translation_rotation_script)
        if draw_m1 != "None":
            # plot only one non-zero frame since they are all the same
            self._add_representation(first_molecule=True, second_molecule=False, periodic="xyzXYZ",
                                       representation=VMD_BOND_TYPE_DICT[draw_m1], trajectory_frames=[1])
        if draw_m2 != "None":
            # plot red frames
            if num_red > 0:
                self._add_representation(first_molecule=False, second_molecule=True, periodic=None,
                                         coloring= "ColorID", color="red",
                                        representation=VMD_BOND_TYPE_DICT[draw_m2], trajectory_frames=list(range(1,
                                                                                                                 num_red+1)))
            # plot blue frames
            if num_blue > 0:
                self._add_representation(first_molecule=False, second_molecule=True, periodic=None, color="blue",
                                         representation=VMD_BOND_TYPE_DICT[draw_m2],  coloring="ColorID",
                                         trajectory_frames=list(range(num_red+1, num_red + num_blue + 1)))
        if draw_rectangular_box:
            self.add_box(*box_limits)
        if gridpoints is not None:
            self.add_grid(gridpoints)
        if translation_rotation_script:
            self._add_rotations_translations()
        if zoom_level > 0:
            self._zoom_on_box(box_limits[0], box_limits[1], zoom_level)
        # we always have three representations (first molecule, second in red, second in blue
        if num_red > 0 and num_blue > 0:
            list_representation_indices = [0,1,2]
        # unless we only plot red or only plot blue, then we have two
        else:
            list_representation_indices = [0,1]
        self._render_representations(list_representation_indices, plot_path=plot_name)
        self.write_text_to_file(vmd_name)


    def prepare_clustering_script(self, indices_per_cluster: list, color_per_cluster: list, plot_name_all_together: str,
                                  plot_names_individual: list) -> None:
        """
        TODO: this is old, needs to be updated if used
        """
        assert len(color_per_cluster) == len(indices_per_cluster) == len(plot_names_individual), \
            f"{len(color_per_cluster)}!={len(indices_per_cluster)}!={len(plot_names_individual)}"

        # first molecule in a normal color
        self._add_representation(first_molecule=True, second_molecule=False, trajectory_frames=0)

        # now add second molecule separately for each cluster:
        for cluster_indices, cluster_color in zip(indices_per_cluster, color_per_cluster):
            self._add_representation(first_molecule=False, second_molecule=True, trajectory_frames=cluster_indices,
                                     coloring="ColorID", color=cluster_color)

        self._add_rotations_translations()

        # plot all together
        all_representations = list(range(self.num_representations))
        self._render_representations(all_representations, plot_name_all_together)

        # plot individually
        for cluster_i, plot_name in enumerate(plot_names_individual):
            self._render_representations([0, cluster_i+1], plot_name)

    
