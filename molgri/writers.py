from typing import Tuple
import os
import shutil

import MDAnalysis as mda
from MDAnalysis import Merge
import numpy as np

from molgri.grids import ZeroGrid, FullGrid
from molgri.parsers import TranslationParser, TrajectoryParser, ParsedMolecule
from molgri.paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT
from molgri.pts import Pseudotrajectory

class PtWriter:

    def __init__(self, name_central_gro: str, name_rotating_gro: str, full_grid: FullGrid):
        """
        We read in two base gro files, each containing one molecule. Capable of writing a new gro file that
        contains one or more time steps in which the second molecule moves around. First molecule is only read
        and the lines copied at every step; second molecule is read and represented with Atom objects which can rotate
        and translate.

        Args:
            name_central_gro: name of the molecule that stays fixed
            name_rotating_gro: name of the molecule that moves in a pseudotrajectory
            full_grid: consists of unter-grids that span state space
        """

        central_file_path = f"{PATH_INPUT_BASEGRO}{name_central_gro}.gro"
        self.central_parser = TrajectoryParser(central_file_path)
        self.central_atoms = self.central_parser.get_atoms()
        rotating_file_path = f"{PATH_INPUT_BASEGRO}{name_rotating_gro}.gro"
        self.rotating_parser = TrajectoryParser(rotating_file_path)
        if not np.all(self.central_parser.box == self.rotating_parser.box):
            print(f"Warning! Simulation boxes of both molecules are different. Selecting the box of"
                  f"{self.central_parser.molecule_name} with dimensions {self.central_parser.get_box()}")
        self.box = self.central_parser.get_box()
        self.full_grid = full_grid
        self.pt = Pseudotrajectory(self.rotating_parser.as_parsed_molecule(), full_grid)
        self.file_name = self.get_output_name()

    def get_output_name(self):
        mol_name1 = self.central_parser.molecule_name
        mol_name2 = self.rotating_parser.molecule_name
        result_file_path = f"{mol_name1}_{mol_name2}_{self.full_grid.get_full_grid_name()}"
        return result_file_path

    def write_full_pt(self, ending_trajectory: str = "xtc", ending_structure: str = "gro", measure_time: bool = False):
        output_path = f"{PATH_OUTPUT_PT}{self.file_name}.{ending_trajectory}"
        structure_path = f"{PATH_OUTPUT_PT}{self.file_name}.{ending_structure}"
        with mda.Writer(structure_path) as structure_writer:
            merged_universe = Merge(self.central_atoms, self.rotating_parser.get_atoms())
            merged_universe.dimensions = self.box
            structure_writer.write(merged_universe)
        trajectory_writer = mda.Writer(output_path, multiframe=True)
        if measure_time:
            generating_func = self.pt.generate_pt_and_time
        else:
            generating_func = self.pt.generate_pseudotrajectory
        for i, second_molecule in generating_func():
            merged_universe = Merge(self.central_atoms, second_molecule.get_atoms())
            merged_universe.dimensions = self.box
            trajectory_writer.write(merged_universe)
        trajectory_writer.close()

    def write_frames_in_directory(self):
        self.f.close()
        directory = f"{PATH_OUTPUT_PT}{self.get_output_name()}"

        try:
            os.mkdir(directory)
        except FileExistsError:
            # delete contents if folder already exist
            filelist = [f for f in os.listdir(directory) if f.endswith(".gro")]
            for f in filelist:
                os.remove(os.path.join(directory, f))
        for i, second_molecule in self.pt.generate_pseudotrajectory():
            self.f = open(f"{directory}/{i}.gro", "w")
            self.write_frame(i, second_molecule)
            self.f.close()


class TwoMoleculeGroWriter(PtWriter):

    def __init__(self, name_central_gro: str, name_rotating_gro: str, translation_nm: float):
        """
        A class to create a 'PT' that contains only one frame, namely of the two molecules separated by the distance
        translation_nm in z-direction.

        Args:
            name_central_gro: name of the molecule that stays fixed
            name_rotating_gro: name of the molecule that moves in a pseudotrajectory
            translation_nm: how far away molecules should be in the end
        """
        trans_grid = TranslationParser(f"[{translation_nm}]")
        full_grid = FullGrid(b_grid=ZeroGrid(), o_grid=ZeroGrid(), t_grid=trans_grid)
        super().__init__(name_central_gro, name_rotating_gro, full_grid)


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


def directory2full_pt(directory_path: str):
    path_to_dir, dir_name = os.path.split(directory_path)
    filelist = [f for f in os.listdir(directory_path) if f.endswith(".gro")]
    filelist.sort(key=lambda x: int(x.split(".")[0]))
    with open(f"{path_to_dir}{dir_name}.gro", 'wb') as wfd:
        for f in filelist:
            with open(f"{directory_path}/{f}", 'rb') as fd:
                shutil.copyfileobj(fd, wfd)


def full_pt2directory(full_pt_path: str):
    with open(full_pt_path, "r") as f_read:
        lines = f_read.readlines()
    num_atoms = int(lines[1].strip("\n").strip())
    num_frame_lines = num_atoms + 3
    directory = full_pt_path.split(".")[0]
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


if __name__ == '__main__':
    from molgri.grids import IcoGrid
    grid = FullGrid(b_grid=ZeroGrid(), o_grid=IcoGrid(15), t_grid=TranslationParser("[1, 2, 3]"))
    ptwriter = PtWriter("H2O", "CL", grid)
    ptwriter.write_full_pt(measure_time=True)