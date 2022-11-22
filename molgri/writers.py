from typing import Tuple
import os
import shutil

from molgri.bodies import Molecule
from molgri.grids import ZeroGrid, FullGrid
from molgri.parsers import BaseGroParser, TranslationParser
from molgri.paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT
from molgri.pts import Pseudotrajectory


class GroWriter:

    def __init__(self, file_name: str):
        """
        This simple class determines only the format of how data is written to the .gro file, all information
        is directly provided as arguments

        Args:
            file_name: entire name of the file where the gro file should be saved including path and ending
        """
        self.file_name = file_name
        self.f = open(self.file_name, "w")

    def write_comment_num(self,  num_atoms: int, comment: str = ""):
        """
        Args:
            num_atoms: required, number of atoms in this frame
            comment: string of a comment without new line symbol
        """
        # write comment
        self.f.write(f"{comment}\n")
        # write total number of atoms
        self.f.write(f"{num_atoms:5}\n")

    def write_atom_line(self, residue_num: int, residue_name: str, atom_name: str, atom_num: str,
                        pos_nm_x: float, pos_nm_y: float, pos_nm_z: float,
                        vel_x: float = 0, vel_y: float = 0, vel_z: float = 0):
        self.f.write(f"{residue_num:5}{residue_name:5}{atom_name:>5}{atom_num:5}{pos_nm_x:8.3f}{pos_nm_y:8.3f}"
                     f"{pos_nm_z:8.3f}{vel_x:8.4f}{vel_y:8.4f}{vel_z:8.4f}\n")

    def write_box(self, box: Tuple[float]):
        assert len(box) == 3, "simulation box must have three dimensions"
        for box_el in box:
            self.f.write(f"\t{box_el}")
        self.f.write("\n")


class PtWriter(GroWriter):

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
        self.central_parser = BaseGroParser(central_file_path, parse_atoms=False)
        rotating_file_path = f"{PATH_INPUT_BASEGRO}{name_rotating_gro}.gro"
        self.rotating_parser = BaseGroParser(rotating_file_path, parse_atoms=True)
        self.full_grid = full_grid
        self.pt = Pseudotrajectory(self.rotating_parser.molecule_set, full_grid)
        super().__init__(f"{PATH_OUTPUT_PT}{self.get_output_name()}.gro")
        self.c_num = self.central_parser.num_atoms
        self.r_num = self.rotating_parser.num_atoms

    def get_output_name(self):
        mol_name1 = self.central_parser.molecule_name
        mol_name2 = self.rotating_parser.molecule_name
        result_file_path = f"{mol_name1}_{mol_name2}_{self.full_grid.get_full_grid_name()}"
        return result_file_path

    def write_frame(self, frame_num: int, second_molecule: Molecule):
        comment = f"c_num={self.c_num}, r_num={self.r_num}, t={frame_num}"
        total_num = self.c_num + self.r_num
        self.write_comment_num(comment=comment, num_atoms=total_num)
        self._write_first_molecule()
        self._write_current_second_molecule(second_molecule=second_molecule)
        self.write_box(box=self.central_parser.box)

    def _write_first_molecule(self):
        self.f.writelines(self.central_parser.atom_lines_nm)

    def _write_current_second_molecule(self, second_molecule: Molecule):
        num_atom = self.c_num + 1
        num_molecule = 2
        for atom in second_molecule.atoms:
            pos_nm = atom.position
            name = atom.gro_label
            self.write_atom_line(residue_num=num_molecule, residue_name=second_molecule.residue_name,
                                 atom_name=name, atom_num=num_atom, pos_nm_x=pos_nm[0], pos_nm_y=pos_nm[1],
                                 pos_nm_z=pos_nm[2])
            num_atom += 1

    def write_full_pt_gro(self, measure_time: bool = False):
        if measure_time:
            generating_func = self.pt.generate_pt_and_time
        else:
            generating_func = self.pt.generate_pseudotrajectory
        for i, second_molecule in generating_func():
            self.write_frame(i, second_molecule)
        self.f.close()

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
