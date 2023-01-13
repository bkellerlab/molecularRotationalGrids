"""
This module contains PtWriter object to write Pseudotrajectories to trajectory/topology files and a PtIOManager that
is a high-level function combining: parsing from input files, creating grids and writing the outputs.
"""
import os
import shutil

# noinspection PyPep8Naming
import MDAnalysis as mda
from MDAnalysis import Merge
import numpy as np

from molgri.constants import EXTENSION_TRAJECTORY, EXTENSION_TOPOLOGY
from molgri.space.fullgrid import FullGrid
from molgri.molecules.parsers import FileParser, ParsedMolecule
from molgri.paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT
from molgri.molecules.pts import Pseudotrajectory
from molgri.wrappers import time_method


class PtIOManager:

    def __init__(self, name_central_molecule: str, name_rotating_molecule: str, o_grid_name: str, b_grid_name: str,
                 t_grid_name: str):
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
        """
        # parsing input files
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
        self.decorator_label = f"Pseudotrajectory {self.determine_pt_name()}"  # needed for timing write-out

    def determine_pt_name(self) -> str:
        """
        Determine the base name of pseudotrajectory file/directory without any paths or extensions.

        Returns:
            PT name, eg H2O_CL_o_ico_15_b_cube3D_45_t_123456
        """
        name_c_molecule = self.central_parser.get_file_name()
        name_r_molecule = self.rotating_parser.get_file_name()
        name_full_grid = self.full_grid.get_full_grid_name()
        return f"{name_c_molecule}_{name_r_molecule}_{name_full_grid}"

    def construct_pt(self, extension_trajectory: str = EXTENSION_TRAJECTORY,
                     extension_structure: str = EXTENSION_TOPOLOGY,
                     as_dir: bool = False):
        """
        The highest-level method to be called in order to generate and save a pseudotrajectory.

        Args:
            extension_trajectory: what extension to provide to the trajectory file
            extension_structure: what extension to provide to the structure (topology) file
            as_dir: if True, don't save trajectory in one file but split it in frames
        """
        if as_dir:
            selected_function = self.writer.write_frames_in_directory
        else:
            selected_function = self.writer.write_full_pt
        selected_function(self.pt, extension_trajectory=extension_trajectory, extension_structure=extension_structure)

    @time_method
    def construct_pt_and_time(self, **kwargs):
        """
        Same as construct_pt, but time the execution and write out a message about duration.

        Args:
            see construct_pt
        """
        self.construct_pt(**kwargs)


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

    def write_structure(self, pt: Pseudotrajectory, extension_structure: str = "gro"):
        """
        Write the one-frame topology file, eg in .gro format.

        Args:
            pt: a Pseudotrajectory object with method .get_molecule() that returns current ParsedMolecule
            extension_structure: determines type of file to which topology should be saved
        """
        structure_path = f"{PATH_OUTPUT_PT}{self.file_name}.{extension_structure}"
        if not np.all(self.box == pt.get_molecule().get_box()):
            print(f"Warning! Simulation boxes of both molecules are different. Selecting the box of "
                  f"central molecule with dimensions {self.box}")
        with mda.Writer(structure_path) as structure_writer:
            self._merge_and_write(structure_writer, pt)

    def write_full_pt(self, pt: Pseudotrajectory, extension_trajectory: str = "xtc", extension_structure: str = "gro"):
        """
        Write the trajectory file as well as the structure file (only at the first time-step).

        Args:
            pt: a Pseudotrajectory object with method .generate_pseudotrajectory() that generates ParsedMolecule objects
            extension_trajectory: determines type of file to which trajectory should be saved
            extension_structure: determines type of file to which topology should be saved
        """
        output_path = f"{PATH_OUTPUT_PT}{self.file_name}.{extension_trajectory}"
        trajectory_writer = mda.Writer(output_path, multiframe=True)
        last_i = 0
        for i, _ in pt.generate_pseudotrajectory():
            # for the first frame write out topology
            if i == 0:
                self.write_structure(pt, extension_structure)
            self._merge_and_write(trajectory_writer, pt)
            last_i = i
        product_of_grids = pt.position_grid.shape[0] * len(pt.rot_grid_body) * pt.position_grid.shape[1]
        assert last_i + 1 == product_of_grids, f"Length of PT not correct, {last_i}=/={product_of_grids}"
        trajectory_writer.close()

    def _create_dir_or_empty_it(self) -> str:
        """
        Helper function that determines the name of the directory in which single frames of the trajectory are
        saved. If the directory already exists, its previous contents are deleted.

        Returns:
            path to the directory
        """
        directory = f"{PATH_OUTPUT_PT}{self.file_name}"
        try:
            os.mkdir(directory)
        except FileExistsError:
            # delete contents if folder already exist
            filelist = [f for f in os.listdir(directory)]
            for f in filelist:
                os.remove(os.path.join(directory, f))
        return directory

    def write_frames_in_directory(self, pt: Pseudotrajectory, extension_trajectory: str = "xtc",
                                  extension_structure: str = "gro"):
        """
        As an alternative to saving a full PT in a single trajectory file, you can create a directory with the same
        name and within it single-frame trajectories named with their frame index. Also save the structure file at
        first step.

            pt: a Pseudotrajectory object with method .generate_pseudotrajectory() that generates ParsedMolecule objects
            extension_trajectory: determines type of file to which trajectory should be saved
            extension_structure: determines type of file to which topology should be saved
        """
        directory = self._create_dir_or_empty_it()
        for i, _ in pt.generate_pseudotrajectory():
            if i == 0:
                self.write_structure(pt, extension_structure)
            f = f"{directory}/{i}.{extension_trajectory}"
            with mda.Writer(f) as structure_writer:
                self._merge_and_write(structure_writer, pt)


def converter_gro_dir_gro_file_names(pt_file_path: str = None, pt_directory_path: str = None,
                                     extension: str = None) -> tuple:
    """
    Converter that helps separate a PT path into base path, directory/file name and extension. Provide one of
    the arguments, pt_file_path or pt_directory_path+extension; if you provide both, only pt_file_path will be used.

    Args:
        pt_file_path: full path with extension pointing to the PT file
        pt_directory_path: full path with extension pointing to the PT directory
        extension: extension of PT

    Returns:
        (base file path, name without extension, full file path with extension, full directory path)
    """
    if pt_file_path:
        without_ext, file_extension = os.path.splitext(pt_file_path)
        file_path, file_name = os.path.split(without_ext)
        pt_directory_path = os.path.join(file_path, file_name+"/")
    elif pt_directory_path and extension:
        file_path, file_name = os.path.split(pt_directory_path)
        file_with_ext = file_name + f".{extension}"
        pt_file_path = os.path.join(file_path, file_with_ext)
    else:
        raise ValueError("pt_file_path nor pt_directory_path + extension provided.")
    return file_path + "/", file_name, pt_file_path, pt_directory_path


def directory2full_pt(directory_path: str, trajectory_endings: str = "xtc"):
    """
    Convert a directory full of single-frame PTs in a single long PT.

    Args:
        directory_path: full path with extension pointing to the PT directory
        trajectory_endings: extension of PT files in the directory
    """
    path_to_dir, dir_name = os.path.split(directory_path)
    filelist = [f for f in os.listdir(directory_path) if f.endswith(f".{trajectory_endings}")]
    filelist.sort(key=lambda x: int(x.split(".")[0]))
    with open(f"{path_to_dir}/{dir_name}.{trajectory_endings}", 'wb') as wfd:
        for f in filelist:
            with open(f"{directory_path}/{f}", 'rb') as fd:
                shutil.copyfileobj(fd, wfd)


def full_pt2directory(full_pt_path: str, structure_path: str):
    """
    Convert a long PT into a directory of single-frame PTs.

    Args:
        full_pt_path: full path with extension pointing to the PT file
        structure_path: full path with extension pointing to the structure/topology file

    Returns:

    """
    with open(structure_path, "r") as f_read:
        lines = f_read.readlines()
    num_atoms = int(lines[1].strip("\n").strip())
    num_frame_lines = num_atoms + 3
    directory = full_pt_path.split(".")[0]
    with open(full_pt_path, "rb") as f_read:
        lines = f_read.readlines()
    try:
        os.mkdir(directory)
    except FileExistsError:
        # delete contents if folder already exist
        filelist = [f for f in os.listdir(directory) if f.endswith(".gro")]
        for f in filelist:
            os.remove(os.path.join(directory, f))
    for i in range(len(lines) // num_frame_lines):
        with open(f"{directory}/{i}.gro", "wb") as f_write:
            f_write.writelines(lines[num_frame_lines*i:num_frame_lines*(i+1)])
