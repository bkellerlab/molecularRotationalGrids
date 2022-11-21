from typing import Tuple

from molgri.bodies import Molecule
from molgri.parsers import BaseGroParser
from molgri.paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT
from molgri.pts import Pseudotrajectory

class GroWriter:

    def __init__(self, file_name : str):
        """
        This simple class determines only the format of how data is written to the .gro file, all information
        is directly provided as arguments

        Args:
            file_name: entire name of the file where the gro file should be saved including path and ending
        """
        self.f = open(file_name, "w")

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

    def __init__(self, name_central_gro: str, name_rotating_gro: str, result_name_gro=None):
        """
        We read in two base gro files, each containing one molecule. Capable of writing a new gro file that
        contains one or more time steps in which the second molecule moves around. First molecule is only read
        and the lines copied at every step; second molecule is read and represented with Atom objects which can rotate
        and translate.

        Args:
            name_central_gro: name of the molecule that stays fixed
            name_rotating_gro: name of the molecule that moves in a pseudotrajectory
        """
        central_file_path = f"{PATH_INPUT_BASEGRO}{name_central_gro}.gro"
        rotating_file_path = f"{PATH_INPUT_BASEGRO}{name_rotating_gro}.gro"
        if result_name_gro is None:
            result_file_path = f"{PATH_OUTPUT_PT}{name_central_gro}_{name_rotating_gro}_run.gro"
        else:
            result_file_path = f"{PATH_OUTPUT_PT}{result_name_gro}.gro"
        super().__init__(result_file_path)
        self.central_parser = BaseGroParser(central_file_path, parse_atoms=False)
        self.c_num = self.central_parser.num_atoms
        # parse rotating file as Atoms
        self.rotating_parser = BaseGroParser(rotating_file_path, parse_atoms=True)
        self.r_num = self.rotating_parser.num_atoms

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

    def write_full_pt_gro(self, pt_generator: ):

        self.f.close()
        pass

    def generate_two_molecule_gro(self, translation_nm=0.3):
        # move second molecule for initial dist
        self.rotating_parser.molecule_set.translate_objects_radially(translation_nm)
        self.write_frame(frame_num=0, second_molecule=self.rotating_parser.molecule_set.all_objects[0])
        self.f.close()


#class GroDirectoryWriter:
