"""
Vmdlogs are instruction files provided to VMD. Here we create vmdlogs that help display specific structures of the
(pseudo)trajectories, often the ones corresponding to eigenvector extremes.
"""
import numpy as np
from numpy.typing import NDArray, ArrayLike

from molgri.space.utils import k_argmax_in_array

VMD_COLOR_DICT = {"black": 16, "yellow": 4, "orange": 3, "green": 7, "blue": 0, "cyan": 10, "purple": 11,
              "gray": 2, "pink": 9, "red": 1, "magenta": 27}


def find_num_extremes(eigenvector_array: NDArray, explains_x_percent: float = 40, only_positive: bool = True) -> int:
    """
    In a 1D array that may contain positive and negative elements, we want to know how many of the largest positive
    values we need to sum to get at least X % of the total sum of positive elements (option only_positive=True) or
    how many of the most negative values we need to sum to get at least X % of the total sum of negative elements (
    option only_positive=False

    Args:
        eigenvector_array (NDArray):
        explains_x_percent (float): percentage we wanna explain, usually around 70-90
        only_positive (bool): if True we will only consider positive values in array, if False only negative values

    Returns:
        The number (int) of elements in eigenvector_array that we need to sum to get to X % (pos or neg). Note that
        is eigenvector_array is not sorted, these elements may not be the first elements.
    """
    if only_positive:
        allowed_elements = eigenvector_array[eigenvector_array > 0]
    else:
        allowed_elements = eigenvector_array[eigenvector_array < 0]
        allowed_elements = - allowed_elements

    # case where e.g. all elements positive but you are looking at the needed number of neg elements
    if len(allowed_elements) == 0:
        return 0

    total_sum = np.sum(allowed_elements)
    sorted_by_size = -np.sort(-allowed_elements)  # this is to sort from largest to smallest
    partial_sum = np.cumsum(sorted_by_size)
    percentage_explained = 100 * partial_sum / total_sum
    # first el that reaches x_percent
    larger_index = np.argmax(percentage_explained > explains_x_percent)
    # because of zero-indexing we use +1 -> e.g. if we reach 80% with the first element, the index will be 0 but we
    # need 1 element
    return larger_index + 1


def find_indices_of_largest_eigenvectors(eigenvector_array: NDArray, which: str, num_extremes: int,
                                         index_list: list = None, add_one: bool = True) -> NDArray:
    """
    Given a 1D eigenvector array, find indices with the largest values. This is a help function so that you can create
    VMD inputs that display most positive and most negative parts of an eigenvector with molecular stuctures.

    Args:
        eigenvector_array: array of shape (N_d,) of eigenvector values for a single eigenvector
            (here: N_d is the num of cells in the grid)
        which: "abs", "pos" or "neg" <- largest in absolute sense or most positive or most negative
        num_extremes: how many of the largest-value indices of eigenvector array to consider
        index_list: only used if indices of eigenvector array are not directly transferrable to assignment/Pt indices,
            perhaps because some pt frames with very high energies were filtered out or similar
        add_one: add +1 to indices, used because VMD has the structure file as the 0th frame
    """

    # determine the max values based on which parameter you sort by
    if num_extremes is None:
        if which == "abs" or which == "pos":
            num_extremes = find_num_extremes(np.abs(eigenvector_array), only_positive=True)
        else:
            num_extremes = find_num_extremes(eigenvector_array, only_positive=False)
    if which == "abs":
        most_populated = k_argmax_in_array(np.abs(eigenvector_array), num_extremes)
    elif which == "pos":
        most_populated = k_argmax_in_array(eigenvector_array, num_extremes)
    elif which == "neg":
        most_populated = k_argmax_in_array(-eigenvector_array, num_extremes)
    else:
        raise ValueError(f"The only available options for 'which' are 'abs', 'pos' and 'neg', not {which}.")
    print(f"Selected number of extremes is {num_extremes}")
    # now possibly perform re-indexing if index_list is involved
    original_index_populated = []
    if index_list is None:
        original_index_populated = most_populated
    else:
        for mp in most_populated:
            original_index_populated.append(index_list[mp])

    original_index_populated = np.array(original_index_populated)

    if add_one:
        original_index_populated += 1

    original_index_populated = original_index_populated.reshape((-1,))
    return original_index_populated


def find_assigned_trajectory_indices(assignments, indices_to_find, num_of_examples: int = 1,
                                     add_one: bool = True) -> NDArray:
    output = []
    for assigned_index in indices_to_find:
        try:
            output.extend(np.random.choice(np.where(assignments == assigned_index)[0], num_of_examples))
        except ValueError:
            # no occurences are that cell (might happen if you assigned whole trajectory but use a part for plotting)
            pass
    if add_one:
        output = [x + 1 for x in output]
    return np.array(output)


def from_eigenvector_array_to_dominant_eigenvector_indices(eigenvector_array: NDArray, index_list: list = None,
                                                           n_eigenvectors: int = 5, num_extremes: int = 10,
                                                           num_of_examples: int = 1, assignments: NDArray = None):
    if len(eigenvector_array) < n_eigenvectors:
        print(f"Warning! Don't have data for {n_eigenvectors}, will only plot {len(eigenvector_array)}")
        n_eigenvectors = len(eigenvector_array)

    if num_extremes is not None:
        double_extremes = 2 * num_extremes
    else:
        double_extremes = None

    # for msm case need to go over assignments
    if assignments is not None:
        zeroth_eigenvector = find_indices_of_largest_eigenvectors(eigenvector_array[0], which="abs",
                                                                  index_list=index_list,
                                                                  num_extremes=double_extremes,
                                                                  add_one=False)
        zeroth_eigenvector = find_assigned_trajectory_indices(assignments, zeroth_eigenvector, num_of_examples,
                                                              add_one=True)
    else:
        zeroth_eigenvector = find_indices_of_largest_eigenvectors(eigenvector_array[0], which="abs",
                                                                  index_list=index_list,
                                                                  num_extremes=double_extremes)
    all_pos_eigenvectors = []
    all_neg_eigenvectors = []
    for i in range(1, n_eigenvectors):
        if assignments is not None:
            pos_eigenvector = find_indices_of_largest_eigenvectors(eigenvector_array[i], which="pos", add_one=False,
                                                                   index_list=index_list, num_extremes=num_extremes)
            pos_eigenvector = find_assigned_trajectory_indices(self.assignments, pos_eigenvector, num_of_examples,
                                                               add_one=True)
            neg_eigenvector = find_indices_of_largest_eigenvectors(eigenvector_array[i], which="neg", add_one=False,
                                                                   index_list=index_list, num_extremes=num_extremes)
            neg_eigenvector = find_assigned_trajectory_indices(self.assignments, neg_eigenvector, num_of_examples,
                                                               add_one=True)
        else:
            pos_eigenvector = find_indices_of_largest_eigenvectors(eigenvector_array[i], which="pos",
                                                                   index_list=index_list, num_extremes=num_extremes)
            neg_eigenvector = find_indices_of_largest_eigenvectors(eigenvector_array[i], which="neg",
                                                                   index_list=index_list, num_extremes=num_extremes)
        all_pos_eigenvectors.append(pos_eigenvector)
        all_neg_eigenvectors.append(neg_eigenvector)

    zeroth_eigenvector = np.array(zeroth_eigenvector)
    all_pos_eigenvectors = np.array(all_pos_eigenvectors)
    all_neg_eigenvectors = np.array(all_neg_eigenvectors)
    return zeroth_eigenvector, all_pos_eigenvectors, all_neg_eigenvectors


class VMDCreator:
    """
    From pieces of strings build a long vmdlog file that can be used inside vmd to automatically execute commands
    like: create a CPK representation, rotate the molecule, render a picture, change the color, render again etc.
    """

    def __init__(self, index_first_molecule: str, index_second_molecule: str, is_protein: bool = False):
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
            self.default_drawing_method = "CPK 1.000000 0.300000 12.000000 10.000000"

        self.num_representations = 0
        self.total_file_text = ""
        self._add_pretty_plot_settings()

    def write_text_to_file(self, output_file_path: str) -> None:
        """
        Because basically every method here writes to internal self.total_file_text property, in the end we just
        need to transfer it to a file.
        """
        with open(output_file_path, "w") as f:
            f.write(self.total_file_text)

        # in case we want to build another file after this one
        self.total_file_text = ""

    def _add_pretty_plot_settings(self):
        """
        Delete the initial default representation.
        Additionally, some nice settings so that pictures look good.
        """

        self.total_file_text += f"""
mol delrep 0 0
color Display Background white
axes location Off
mol material Opaque
display projection Orthographic
display shadows on
display ambientocclusion on
material add copy AOChalky
material change shininess Material22 0.000000
display resize 1800 1200
"""

    def _add_representation(self, first_molecule: bool = False, second_molecule: bool = True, coloring: str = None,
                            color: str = None, representation: str = None, trajectory_frames: ArrayLike = None):
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
        elif type(trajectory_frames) == list:
            trajectory_frames_as_str = ', '.join(map(str, [int(x) for x in trajectory_frames]))
        elif type(trajectory_frames) == NDArray:
            trajectory_frames_as_str = ', '.join(map(str, trajectory_frames.flatten().astype(int)))
        elif type(trajectory_frames) == str:
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

        self.num_representations += 1

    def _render_representations(self, list_representation_indices: ArrayLike, plot_path: str):
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

    def _show_representation(self, representation_index: int):
        self.total_file_text += f"\nmol showrep 0 {representation_index} 1\n"

    def _hide_representation(self, representation_index: int):
        self.total_file_text += f"\nmol showrep 0 {representation_index} 0\n"

    def load_translation_rotation_script(self, path_translation_rotation_script: str):
        self.translations_rotations_script = path_translation_rotation_script

    def prepare_eigenvector_script(self, abs_eigenvector_frames: NDArray, pos_eigenvector_frames: NDArray,
                                   neg_eigenvector_frames: NDArray, plot_names: list):
        """
        Everything you need to plot eigenvectors:
        - make plotting pretty
        - translate, scale and rotate appropriately to have an optimal fig
        - add indices of zeroth eigenvector at max absolute values
        - add indices of higher eigenvectors at most positive/most negative values
        - render the plots

        Args:
            abs_eigenvector_frames (NDArray): a 1D array of integers for representative cells of the 0th eigenvector
            pos_eigenvector_frames (NDArray): a 2D array of integers, each row representing most positive cells of the
                ith eigenvector with i=1,2,3... Must have same length as neg_eigenvector_frames.
            neg_eigenvector_frames (NDArray): a 2D array of integers, each row representing most negative cells of the
                ith eigenvector with i=1,2,3... Must have same length as pos_eigenvector_frames.
            plot_names: file paths for all the renders. Must have the length of neg_eigenvector_frames + 1
        """
        assert len(pos_eigenvector_frames) == len(neg_eigenvector_frames)
        assert len(plot_names) == len(neg_eigenvector_frames) + 1

        # add first molecule without any special colors etc
        self._add_representation(first_molecule=True, second_molecule=False, trajectory_frames=0)

        # add zeroth eigenvector without any special colors
        self._add_representation(first_molecule=False, second_molecule=True, trajectory_frames=abs_eigenvector_frames[0])

        # for the rest add one red, one blue
        for pos_frames, neg_frames in zip(pos_eigenvector_frames, neg_eigenvector_frames):
            self._add_representation(first_molecule=False, second_molecule=True, coloring="ColorId", color="blue",
                                     trajectory_frames=pos_frames)
            self._add_representation(first_molecule=False, second_molecule=True, coloring="ColorId", color="red",
                                     trajectory_frames= neg_frames)

        self._add_rotations_translations()

        # render the zeroth eigenvector
        self._render_representations([0, 1], plot_names[0])

        # render the rest of eigenvectors
        last_used_representation = 1
        for plot_name in enumerate(plot_names):
            # each render contains first molecule in representation 0 and second molecule in representation 1, 2, 3 ...
            self._render_representations([0, last_used_representation+1, last_used_representation+2], plot_name)
            last_used_representation += 2



    def prepare_clustering_script(self, clusters: NDArray, color_per_cluster: list, plot_name: str, max_num_per_cluster: int = 20):
        # first molecule in a normal color
        self._add_representation(first_molecule=True, second_molecule=False, trajectory_frames=0)

        # for each unique label add a colored set of structures
        unique_labels = np.unique(labels)
        assert len(unique_labels)

        repr_index = 1
        for i, unique_label in enumerate(np.unique(labels)[:len(colors)]):
            cluster = np.where(labels == unique_label)[0]
            population = len(cluster)
            cluster_indices = np.array([x + 1 for x in cluster])
            if population > 5 and population < 1000:
                if population > max_num_per_cluster:
                    cluster_indices = np.random.choice(cluster_indices, max_num_per_cluster)
                total_string += self._add_colored_representation(repr_index, color_dict[colors[i]], cluster_indices)
                repr_index += 1

        total_string += self._add_rotations_translations()
        total_string+="\n"

        total_string += f"render TachyonInternal {plot_name}\n"

        n_colors = len(np.unique(labels)[:len(colors)])
        total_string += self._add_hide_all_representations(n_colors)
        for i in range(n_colors):
            total_string += self._show_representation_i(i+1)
            changed_plot_name = f"{plot_name[:-4]}_{i+1}{plot_name[-4:]}"
            total_string += f"render TachyonInternal {changed_plot_name}\n"
            total_string += self._hide_representation_i(i + 1)


        total_string += "quit"
        return total_string

    def prepare_path_script(self, my_path, plot_paths):
        """
        Use this method to represent a path with VMD pictures. Renders a figure for each grid point tn my_path.

        Args:
            my_path (list): a list of indices like [15, 7, 385, 22] describing a path on a grid, integers are grid
                indices
            plot_paths (list): a list of file names to which renders should be saved, should have the same length as
                my_path
        """

        self._add_representation(first_molecule=True, second_molecule=False, trajectory_frames=0)

        for path_index in my_path:
            self._add_representation(first_molecule=False, second_molecule=True, trajectory_frames=path_index+1)

        self._add_rotations_translations()

        for i, plot_path in enumerate(plot_paths):
            # each render contains first molecule in representation 0 and second molecule in representation 1, 2, 3 ...
            self._render_representations([0, i+1], plot_path)

    def _add_rotations_translations(self):
        if self.translations_rotations_script:
            with open(self.translations_rotations_script, "r") as f:
                contents = f.read()
            self.total_file_text += contents
