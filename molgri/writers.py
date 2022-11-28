import os
import shutil

import MDAnalysis as mda
from MDAnalysis import Merge
import numpy as np

from molgri.grids import FullGrid
from molgri.parsers import FileParser, ParsedMolecule
from molgri.paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT
from molgri.pts import Pseudotrajectory
from molgri.wrappers import time_method


class PtIOManager:

    def __init__(self, name_central_molecule: str, name_rotating_molecule: str, o_grid_name: str, b_grid_name: str,
                 t_grid_name: str):
        """
        This class gets only strings as inputs, combines them to correct paths, manages Parsers and Writers that
        provide a smooth input/output to a Pseudotrajectory. In the end, only this class needs to be called

        Args:
            name_central_molecule: name of the molecule that stays fixed (with or without extension)
            name_rotating_molecule: name of the molecule that moves in a pseudotrajectory
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
        self.writer = PtWriter(name_to_save=self.determine_pt_name(),
                               parsed_central_molecule=self.central_molecule)
        self.pt = Pseudotrajectory(self.rotating_molecule, self.full_grid)
        self.decorator_label = f"Pseudotrajectory {self.determine_pt_name()}"

    def determine_pt_name(self):
        name_c_molecule = self.central_parser.get_file_name()
        name_r_molecule = self.rotating_parser.get_file_name()
        name_full_grid = self.full_grid.get_full_grid_name()
        return f"{name_c_molecule}_{name_r_molecule}_{name_full_grid}"

    def construct_pt(self, ending_trajectory: str = "xtc", ending_structure: str = "gro",
                     as_dir: bool = False):
        if as_dir:
            selected_function = self.writer.write_frames_in_directory
        else:
            selected_function = self.writer.write_full_pt
        selected_function(self.pt, ending_trajectory=ending_trajectory, ending_structure=ending_structure)

    @time_method
    def construct_pt_and_time(self, **kwargs):
        self.construct_pt(**kwargs)


class PtWriter:

    def __init__(self, name_to_save: str, parsed_central_molecule: ParsedMolecule):
        """
        We read in two base gro files, each containing one molecule. Capable of writing a new gro file that
        contains one or more time steps in which the second molecule moves around. First molecule is only read
        and the lines copied at every step; second molecule is read and represented with Atom objects which can rotate
        and translate.
        """
        self.central_molecule = parsed_central_molecule
        self.central_molecule.translate_to_origin()
        self.box = self.central_molecule.get_box()
        self.file_name = name_to_save

    def _merge_and_write(self, writer: mda.Writer, pt: Pseudotrajectory):
        merged_universe = Merge(self.central_molecule.get_atoms(), pt.get_molecule().get_atoms())
        merged_universe.dimensions = self.box
        writer.write(merged_universe)

    def write_structure(self, pt: Pseudotrajectory, ending_structure: str = "gro"):
        structure_path = f"{PATH_OUTPUT_PT}{self.file_name}.{ending_structure}"
        if not np.all(self.box == pt.get_molecule().get_box()):
            print(f"Warning! Simulation boxes of both molecules are different. Selecting the box of"
                  f"central molecule with dimensions {self.box}")
        with mda.Writer(structure_path) as structure_writer:
            self._merge_and_write(structure_writer, pt)

    def write_full_pt(self, pt: Pseudotrajectory, ending_trajectory: str = "xtc", ending_structure: str = "gro"):
        self.write_structure(pt, ending_structure)
        output_path = f"{PATH_OUTPUT_PT}{self.file_name}.{ending_trajectory}"
        trajectory_writer = mda.Writer(output_path, multiframe=True)
        for _ in pt.generate_pseudotrajectory():
            self._merge_and_write(trajectory_writer, pt)
        trajectory_writer.close()

    def _create_dir_or_empty_it(self) -> str:
        directory = f"{PATH_OUTPUT_PT}{self.file_name}"
        try:
            os.mkdir(directory)
        except FileExistsError:
            # delete contents if folder already exist
            filelist = [f for f in os.listdir(directory)]
            for f in filelist:
                os.remove(os.path.join(directory, f))
        return directory

    def write_frames_in_directory(self, pt: Pseudotrajectory, ending_trajectory: str = "xtc",
                                  ending_structure: str = "gro"):
        self.write_structure(pt, ending_structure)
        directory = self._create_dir_or_empty_it()
        for i, second_molecule in pt.generate_pseudotrajectory():
            f = f"{directory}/{i}.{ending_trajectory}"
            with mda.Writer(f) as structure_writer:
                self._merge_and_write(structure_writer, pt)


def converter_gro_dir_gro_file_names(pt_file_path=None, pt_directory_path=None) -> tuple:
    if pt_file_path:
        without_ext, file_extension = os.path.splitext(pt_file_path)
        file_path, file_name = os.path.split(without_ext)
        pt_directory_path = os.path.join(file_path, file_name+"/")
    elif pt_directory_path:
        file_path, file_name = os.path.split(pt_directory_path)
        file_with_ext = file_name + ".gro"
        pt_file_path(file_path, file_with_ext)
    else:
        raise ValueError("pt_file_path nor pt_directory_path provided.")
    return file_path + "/", file_name, pt_file_path, pt_directory_path


def directory2full_pt(directory_path: str, trajectory_endings: str = "xtc"):
    path_to_dir, dir_name = os.path.split(directory_path)
    filelist = [f for f in os.listdir(directory_path) if f.endswith(f".{trajectory_endings}")]
    filelist.sort(key=lambda x: int(x.split(".")[0]))
    with open(f"{path_to_dir}{dir_name}.{trajectory_endings}", 'wb') as wfd:
        for f in filelist:
            with open(f"{directory_path}/{f}", 'rb') as fd:
                shutil.copyfileobj(fd, wfd)


def full_pt2directory(full_pt_path: str, structure_path: str):
    with open(structure_path, "r") as f_read:
        lines = f_read.readlines()
    num_atoms = int(lines[1].strip("\n").strip())
    num_frame_lines = num_atoms + 3
    directory = full_pt_path.split(".")[0]
    with open(full_pt_path, "r") as f_read:
        lines = f_read.readlines()
    try:
        os.mkdir(directory)
    except FileExistsError:
        # delete contents if folder already exist
        filelist = [f for f in os.listdir(directory) if f.endswith(".gro")]
        for f in filelist:
            os.remove(os.path.join(directory, f))
    for i in range(len(lines) // num_frame_lines):
        with open(f"{directory}/{i}.gro", "w") as f_write:
            f_write.writelines(lines[num_frame_lines*i:num_frame_lines*(i+1)])
