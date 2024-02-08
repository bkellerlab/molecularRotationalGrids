"""
Write pseudo-trajectories to files.

This module contains PtWriter object to write Pseudotrajectories to trajectory/topology files and a PtIOManager that
is a high-level function combining: parsing from input files, creating grids and writing the outputs.
"""
import os

# noinspection PyPep8Naming
import MDAnalysis as mda
from MDAnalysis import Merge
import numpy as np

from molgri.constants import EXTENSION_TRAJECTORY, EXTENSION_TOPOLOGY, EXTENSION_LOGGING
from molgri.logfiles import PtLogger, paths_free_4_all
from molgri.space.fullgrid import FullGrid
from molgri.molecules.parsers import FileParser, ParsedMolecule
from molgri.paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT, PATH_OUTPUT_LOGGING
from molgri.molecules.pts import Pseudotrajectory
from molgri.wrappers import time_method


class PtIOManager:

    def __init__(self, name_central_molecule: str, name_rotating_molecule: str, o_grid_name: str, b_grid_name: str,
                 t_grid_name: str, output_name: str = None):
        """
        This class gets only strings as inputs - what a user is expected to provide. The first two strings
        determine where to find input molecular structures, last three how to construct a full grid.
        This object manages Parsers and Writers that provide a smooth input/output to a Pseudotrajectory.

        Args:
            name_central_molecule: name of the molecule that stays fixed (with or without extension,
                                   should be located in the input/ folder)
            name_rotating_molecule: name of the molecule that moves in a pseudotrajectory (with or without extension,
                                   should be located in the input/ folder)
            o_grid_name: name of the grid for rotations around the origin in form 'algorithm_num' (eg. 'ico_50) OR
                         in form 'num' (eg. '50') OR as string 'zero' or 'None' if no rotations needed
            b_grid_name: name of the grid for rotations around the body in form 'algorithm_num' (eg. 'ico_50) OR
                         in form 'num' (eg. '50') OR as string 'zero' or 'None' if no rotations needed
            t_grid_name: translation grid that will be forwarded to TranslationParser, can be a list of numbers,
                         a range or linspace function inside a string
            output_name: select the name under which all associated files will be saved (if None use names of molecules)
        """
        # parsing input files
        self.o_grid_name = o_grid_name
        self.b_grid_name = b_grid_name
        self.t_grid_name = t_grid_name
        central_file_path = f"{PATH_INPUT_BASEGRO}{name_central_molecule}"
        self.central_parser = FileParser(central_file_path)
        rotating_file_path = f"{PATH_INPUT_BASEGRO}{name_rotating_molecule}"
        self.rotating_parser = FileParser(rotating_file_path)
        self.rotating_molecule = self.rotating_parser.as_parsed_molecule()
        self.central_molecule = self.central_parser.as_parsed_molecule()
        # parsing grids
        self.full_grid = FullGrid(b_grid_name=b_grid_name, o_grid_name=o_grid_name, t_grid_name=t_grid_name)
        # initiating writer and pseudotrajectory objects
        self.writer = PtWriter(name_to_save=self.determine_pt_name(),
                               parsed_central_molecule=self.central_molecule)
        self.pt = Pseudotrajectory(self.rotating_molecule, self.full_grid)
        # if the user doesn't select a name, use names of the two molecules as the default
        if output_name is None:
            name_c_molecule = self.central_parser.get_topology_file_name()
            name_r_molecule = self.rotating_parser.get_topology_file_name()
            output_name = f"{name_c_molecule}_{name_r_molecule}"
        self.output_paths = None
        self.output_name = output_name


    def get_decorator_name(self) -> str:
        return f"Pt {self.get_name()}"

    def get_name(self):
        output_paths = self._get_all_output_paths()
        head, tail = os.path.split(output_paths[0])
        name, ext = os.path.splitext(tail)
        return name

    def determine_pt_name(self) -> str:
        """
        Determine the base name of pseudotrajectory file/directory without any paths or extensions.

        Returns:
            PT name, eg H2O_CL_o_ico_15_b_cube3D_45_t_123456
        """
        name_c_molecule = self.central_parser.get_topology_file_name()
        name_r_molecule = self.rotating_parser.get_topology_file_name()
        name_full_grid = self.full_grid.get_name()
        return f"{name_c_molecule}_{name_r_molecule}_{name_full_grid}"

    def _get_all_output_paths(self, extension_trajectory: str = EXTENSION_TRAJECTORY,
                              extension_structure: str = EXTENSION_TOPOLOGY) -> tuple:
        """
        Return paths to (trajectory_file, structure_file, log_file) with unique number ID.
        """
        if self.output_paths is None:
            # determine the first free file name
            paths = [PATH_OUTPUT_PT, PATH_OUTPUT_PT, PATH_OUTPUT_LOGGING]
            names = [self.output_name]*3
            endings = [extension_trajectory, extension_structure, EXTENSION_LOGGING]
            self.output_paths = paths_free_4_all(list_paths=paths, list_names=names, list_endings=endings)
        return self.output_paths

    def construct_pt(self, extension_trajectory: str = EXTENSION_TRAJECTORY,
                     extension_structure: str = EXTENSION_TOPOLOGY,
                     as_dir: bool = False, print_messages=False):
        """
        The highest-level method to be called in order to generate and save a pseudotrajectory.

        Args:
            extension_trajectory: what extension to provide to the trajectory file
            extension_structure: what extension to provide to the structure (topology) file
            as_dir: if True, don't save trajectory in one file but split it in frames
        """
        path_t, path_s, path_l = self._get_all_output_paths(extension_trajectory=extension_trajectory,
                                                            extension_structure=extension_structure)
        # log set-up before calculating PT in case any errors occur in-between
        if print_messages:
            print(f"Saved the log file to {path_l}")
        pt_logger = PtLogger(path_l)
        pt_logger.log_set_up(self)
        logger = pt_logger.logger
        logger.info(f"central molecule: {self.central_parser.get_topology_file_name()}")
        logger.info(f"rotating molecule: {self.rotating_parser.get_topology_file_name()}")
        logger.info(f"input grid parameters: {self.o_grid_name} {self.b_grid_name} {self.t_grid_name}")
        logger.info(f"full grid name: {self.pt.get_full_grid().get_name()}")
        logger.info(f"full grid coordinates:\n{self.pt.get_full_grid().get_full_grid_as_array()}")
        logger.info(f"translation grid [A]: {self.pt.get_full_grid().get_radii()}")
        logger.info(f"quaternions for rot around the body: {self.pt.get_full_grid().get_body_rotations().as_quat()}")
        logger.info(f"positions on a sphere for origin rot: {self.pt.get_full_grid().o_positions}")
        # generate a pt
        if as_dir:
            selected_function = self.writer.write_frames_in_directory
        else:
            selected_function = self.writer.write_full_pt
        selected_function(self.pt, path_t, path_s)
        if print_messages:
            print(f"Saved pseudo-trajectory to {path_t} and structure file to {path_s}")


    @time_method
    def construct_pt_and_time(self, **kwargs):
        """
        Same as construct_pt, but time the execution and write out a message about duration.

        Args:
            see construct_pt
        """
        self.construct_pt(**kwargs)


def _create_dir_or_empty_it(directory_name):
    """
    Helper function that determines the name of the directory in which single frames of the trajectory are
    saved. If the directory already exists, its previous contents are deleted.

    Returns:
        path to the directory
    """
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        # delete contents if folder already exist
        filelist = [f for f in os.listdir(directory_name)]
        for f in filelist:
            os.remove(os.path.join(directory_name, f))


class PtWriter:

    def __init__(self, name_to_save: str, parsed_central_molecule: ParsedMolecule):
        """
        This class writes a pseudotrajectory to a file. A PT consists of one molecule that is stationary at
        origin and one that moves with every time step. The fixed molecule is provided when the class is created
        and the mobile molecule as a generator when the method write_full_pt is called. Writing is done with
        MDAnalysis module, so all formats implemented there are supported.

        Args:
            name_to_save: base name of the PT file without paths or extensions
            parsed_central_molecule: a ParsedMolecule object describing the central molecule, will only be translated
                                     so that COM lies at (0, 0, 0) but not manipulated in any other way.
        """
        self.central_molecule = parsed_central_molecule
        self.central_molecule.translate_to_origin()
        self.box = self.central_molecule.get_box()
        self.file_name = name_to_save

    def _merge_and_write(self, writer: mda.Writer, pt: Pseudotrajectory):
        """
        Helper function to merge Atoms from central molecule with atoms of the moving molecule (at current positions)

        Args:
            writer: an already initiated object writing to file (eg a .gro or .xtc file)
            pt: a Pseudotrajectory object with method .get_molecule() that returns current ParsedMolecule
        """
        merged_universe = Merge(self.central_molecule.get_atoms(), pt.get_molecule().get_atoms())
        merged_universe.dimensions = self.box
        writer.write(merged_universe)

    def write_structure(self, pt: Pseudotrajectory, path_structure: str):
        """
        Write the one-frame topology file, eg in .gro format.

        Args:
            pt: a Pseudotrajectory object with method .get_molecule() that returns current ParsedMolecule
            path_structure: where topology should be saved
        """
        if not np.all(self.box == pt.get_molecule().get_box()):
            print(f"Warning! Simulation boxes of both molecules are different. Selecting the box of "
                  f"central molecule with dimensions {self.box}")
        with mda.Writer(path_structure) as structure_writer:
            self._merge_and_write(structure_writer, pt)

    def write_full_pt(self, pt: Pseudotrajectory, path_trajectory: str, path_structure: str):
        """
        Write the trajectory file as well as the structure file (only at the first time-step).

        Args:
            pt: a Pseudotrajectory object with method .generate_pseudotrajectory() that generates ParsedMolecule objects
            path_trajectory: where trajectory should be saved
            path_structure: where topology should be saved
        """
        n_atoms = len(self.central_molecule.atoms) + len(pt.molecule.atoms)
        trajectory_writer = mda.Writer(path_trajectory, n_atoms=n_atoms, multiframe=True)
        last_i = 0
        for i, _ in pt.generate_pseudotrajectory():
            # for the first frame write out topology
            if i == 0:
                self.write_structure(pt, path_structure)
            self._merge_and_write(trajectory_writer, pt)
            last_i = i
        product_of_grids = len(pt.get_full_grid().get_full_grid_as_array())
        assert last_i + 1 == product_of_grids, f"Length of PT not correct, {last_i}=/={product_of_grids}"
        trajectory_writer.close()

    def write_frames_in_directory(self, pt: Pseudotrajectory, path_trajectory: str, path_structure: str):
        """
        As an alternative to saving a full PT in a single trajectory file, you can create a directory with the same
        name and within it single-frame trajectories named with their frame index. Also save the structure file at
        first step.

            pt: a Pseudotrajectory object with method .generate_pseudotrajectory() that generates ParsedMolecule objects
            path_trajectory: where trajectory should be saved
            path_structure: where topology should be saved
        """
        directory_name, extension_trajectory = os.path.splitext(path_trajectory)
        _create_dir_or_empty_it(directory_name)
        for i, _ in pt.generate_pseudotrajectory():
            if i == 0:
                self.write_structure(pt, path_structure)
            f = f"{directory_name}/{i}.{extension_trajectory}"
            with mda.Writer(f) as structure_writer:
                self._merge_and_write(structure_writer, pt)
