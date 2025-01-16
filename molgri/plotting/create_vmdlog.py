"""
Vmdlogs are instruction files provided to VMD. Here we create vmdlogs that help display specific structures of the
(pseudo)trajectories, often the ones corresponding to eigenvector extremes.
"""
import numpy as np
from numpy.typing import NDArray

from molgri.space.utils import k_argmax_in_array


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


class VMDCreator:
    """
    From pieces of strings build a long vmdlog file that displays particular frames in particular colors,
    has nice plotting (white background etc) and renders the figures to files.
    """

    def __init__(self, experiment_type: str, index_first_molecule: str, index_second_molecule: str,
                 assignments: NDArray = None, is_protein=False):
        self.experiment_type = experiment_type
        self.index_first_molecule = index_first_molecule
        self.index_second_molecule = index_second_molecule
        self.assignments = assignments
        self.is_protein = is_protein

        if is_protein:
            self.coloring = "Structure"
            self.representation = "NewCartoon 0.300000 10.000000 4.100000 0"
        else:
            self.coloring = "Type"
            self.representation = "CPK 1.000000 0.300000 12.000000 10.000000"

    def _search_and_replace(self, input_file: str, output_file: str, to_replace: list, new_strings: list):
        """
        Find a list of keywords in a file, replace them with other words and write out the result.

        Args:
            input_file: path to the start file
            output_file: path to the end file
            to_replace: list of words to find
            new_strings: list of words that should replace the found words (must have the same length as previous list)
        """
        assert len(to_replace) == len(new_strings), "Not equal number of words to find and to replace"
        with open(input_file, 'r') as file:
            file_contents = file.read()
            for search_word, replace_word in zip(to_replace, new_strings):
                file_contents = file_contents.replace(search_word, replace_word)
        with open(output_file, 'w') as file:
            file.write(file_contents)

    def _add_colored_representation(self, representation_index, color_id, frames):
        string_to_add = f"""
mol addrep 0
mol modselect {representation_index} 0 {self.index_second_molecule}
mol modcolor {representation_index} 0 ColorID {color_id}
mol drawframes 0 {representation_index} {{ {', '.join(map(str, frames.flatten().astype(int)))} }}
"""
        return string_to_add

    def _add_pos_neg_eigenvector(self, i1, frames_pos, frames_neg) -> str:
        i2 = i1 + 1

        string_pos_neg = f"""

mol addrep 0
mol modselect {i1} 0 {self.index_second_molecule}
mol modcolor {i1} 0 ColorID 0
mol drawframes 0 {i1} {{ {', '.join(map(str, frames_pos.flatten().astype(int)))} }}
mol addrep 0
mol modselect {i2} 0 {self.index_second_molecule}
mol modcolor {i2} 0 ColorID 1
mol drawframes 0 {i2} {{ {', '.join(map(str, frames_neg.flatten().astype(int)))} }}

"""
        return string_pos_neg

    def _add_first_molecule(self):

        string_first_molecule = f"""

mol addrep 0
mol modselect 0 0 {self.index_first_molecule}
mol modcolor 0 0 {self.coloring}
"""
        return string_first_molecule

    def _add_second_molecule(self, i1, frame_index):
        string_first_molecule = f"""

mol addrep 0
mol modselect {i1} 0 {self.index_second_molecule}
mol drawframes 0 {i1} {frame_index}
mol modcolor {i1} 0 {self.coloring}
"""
        return string_first_molecule

    def _add_pretty_plot_settings(self):

        string_pretty = f"""

mol modstyle 0 0 {self.representation}
color Display Background white
axes location Off
mol color Name
mol representation {self.representation}
mol selection all
mol material Opaque
mol modcolor 1 0 {self.coloring}
display projection Orthographic
display shadows on
display ambientocclusion on
material add copy AOChalky
material change shininess Material22 0.000000

"""
        return string_pretty

    def _add_zeroth_eigenvector(self, frames_abs):
        print(', '.join(map(str, frames_abs.flatten().astype(int))))

        string_zeroth_eigenvector = f"""

mol addrep 0
mol modselect 1 0 {self.index_second_molecule}
mol drawframes 0 1 {{ {', '.join(map(str, frames_abs.flatten().astype(int)))} }}
mol modcolor 1 0 {self.coloring}
"""

        return string_zeroth_eigenvector

    def prepare_eigenvector_script(self, eigenvector_array: NDArray, plot_names: list, index_list: list = None,
                                   n_eigenvectors: int = 5, num_extremes: int = 10, num_of_examples: int = 1):
        """
        Everything you need to plot eigenvectors:
        - make plotting pretty
        - translate, scale and rotate appropriately to have an optimal fig
        - add indices of zeroth eigenvector at max absolute values
        - add indices of higher eigenvectors at most positive/most negative values
        - render the plots

        Args:
            eigenvector_array: 2D eigenvector array, shape (num_evec, len_matrix_side)
        """
        if len(eigenvector_array) < n_eigenvectors:
            print(f"Warning! Don't have data for {n_eigenvectors}, will only plot {len(eigenvector_array)}")
            n_eigenvectors = len(eigenvector_array)

        total_string = ""
        total_string += self._add_pretty_plot_settings()

        total_string += self._add_first_molecule()

        if num_extremes is not None:
            double_extremes = 2*num_extremes
        else:
            double_extremes = None
        if "msm" in self.experiment_type:
            zeroth_eigenvector = find_indices_of_largest_eigenvectors(eigenvector_array[0], which="abs",
                                                                      index_list=index_list,
                                                                      num_extremes=double_extremes,
                                                                      add_one=False)
            zeroth_eigenvector = find_assigned_trajectory_indices(self.assignments, zeroth_eigenvector, num_of_examples,
                                                                  add_one=True)
        else:
            zeroth_eigenvector = find_indices_of_largest_eigenvectors(eigenvector_array[0], which="abs",
                                                                      index_list=index_list,
                                                                      num_extremes=double_extremes)
        total_string += self._add_zeroth_eigenvector(zeroth_eigenvector)
        print(zeroth_eigenvector)

        for i in range(1, n_eigenvectors):
            if "msm" in self.experiment_type:
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
            print(pos_eigenvector)
            print(neg_eigenvector)
            total_string += self._add_pos_neg_eigenvector(2 * i, pos_eigenvector, neg_eigenvector)
        total_string += self._add_rotations_translations()
        total_string += self._add_hide_all_representations(n_eigenvectors)

        total_string += self._add_render_a_plot(0, 1, plot_names[0])

        for i in range(1, n_eigenvectors):
            total_string += self._add_render_a_plot(2 * i, 2 * i + 1, plot_names[i])
        total_string += "quit"
        return total_string

    def prepare_clustering_script(self, labels: NDArray, colors: list, plot_name: str, max_num_per_cluster: int = 20):
        # translate named colors into VMD color ID
        color_dict = {"black": 16, "yellow": 4, "orange": 3, "green": 7, "blue": 0, "cyan": 10, "purple": 11,
                      "gray": 2, "pink": 9, "red": 1, "magenta": 27}

        total_string = ""
        total_string += self._add_pretty_plot_settings()

        total_string += self._add_first_molecule()

        # for each unique label add a colored set of structures

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

    def prepare_path_script(self, my_path):
        total_string = ""
        total_string += self._add_pretty_plot_settings()

        total_string += self._add_first_molecule()


        # increase each path index by one
        repr_index=1
        for path_index in my_path:
            total_string += self._add_second_molecule(repr_index, path_index)
            repr_index += 1

        total_string += self._add_rotations_translations()
        total_string+="\n"




        total_string += self._add_hide_all_representations(len(my_path)+1)
        for i in range(len(my_path)):
            total_string += self._show_representation_i(i+1)
            changed_plot_name = f"path_{i+1}.png"
            total_string += f"render TachyonInternal {changed_plot_name}\n"
            total_string += self._hide_representation_i(i + 1)


        total_string += "quit"
        return total_string
    
    def _hide_representation_i(self, i):
        return f"\nmol showrep 0 {i} 0\n"

    def _show_representation_i(self, i):
        return f"\nmol showrep 0 {i} 1\n"


    def _add_hide_all_representations(self, n_eigenvectors: int):
        total_substring = ""

        for i in range(1, 2 * n_eigenvectors + 3):
            total_substring += f"\nmol showrep 0 {i} 0\n"
        return total_substring

    def _add_render_a_plot(self, i, j, plot_path):

        string_rendering = f"""

mol showrep 0 0 1
mol showrep 0 {i} 1
mol showrep 0 {j} 1
render TachyonInternal {plot_path}
mol showrep 0 {i} 0
mol showrep 0 {j} 0

"""
        return string_rendering

    def _add_rotations_translations(self):
        if self.experiment_type == "sqra_water_in_vacuum" or self.experiment_type == "water_xyz":
            file = "molgri/scripts/vmd_position_sqra_water"
        elif self.experiment_type == "msm_water_in_vacuum":
            file = "molgri/scripts/vmd_position_msm_water_vacuum"
        elif self.experiment_type == "msm_water_in_helium":
            file = "molgri/scripts/vmd_position_msm_water_he"
        elif self.experiment_type == "sqra_fullerene":
            file = "molgri/scripts/vmd_position_sqra_fullerene"
        elif self.experiment_type == "sqra_bpti_trypsine":
            file = "molgri/scripts/vmd_position_sqra_bpti"
        elif self.experiment_type == "guanidinium_xyz":
            file = "molgri/scripts/vmd_position_guanidinium_xyz"
        else:
            return "\n"

        with open(file, "r") as f:
            contents = f.read()
        return contents
