"""
Use two base_gro files (each containing only one molecule) to get a two-molecule base file.
"""
from molgri.my_constants import *
from molgri.paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT
from molgri.parsers.base_gro_parser import BaseGroParser
import os

class TwoMoleculeGro:

    def __init__(self, name_central_gro, name_rotating_gro, result_name_gro=None):
        """
        We read in two base gro files, each containing one molecule. Capable of writing a new gro file that
        contains one or more time steps in which the second molecule moves around. First molecule is only read in as lines
        Args:
            name_central_gro:
            name_rotating_gro:
        """
        central_file_path = f"{PATH_INPUT_BASEGRO}{name_central_gro}.gro"
        rotating_file_path = f"{PATH_INPUT_BASEGRO}{name_rotating_gro}.gro"
        if result_name_gro is None:
            result_file_path = f"{PATH_OUTPUT_PT}{name_central_gro}_{name_rotating_gro}_run.gro"
        else:
            result_file_path = f"{PATH_OUTPUT_PT}{result_name_gro}.gro"
        self.f = open(result_file_path, "w")
        self.cental_parser = BaseGroParser(central_file_path, parse_atoms=False)
        # parse rotating file as Atoms
        self.rotating_parser = BaseGroParser(rotating_file_path, parse_atoms=True, gro_write=self.f)

    def _write_comment_num(self, frame_num=0):
        num_atoms_cen = self.cental_parser.num_atoms
        num_atoms_rotating = self.rotating_parser.num_atoms
        # write comment
        self.f.write(f"c_num={num_atoms_cen}, r_num={num_atoms_rotating}, t={frame_num}\n")
        # write total number of atoms
        self.f.write(f"{num_atoms_cen + num_atoms_rotating:5}\n")

    def _write_first_molecule(self):
        self.f.writelines(self.cental_parser.atom_lines_nm)

    def _write_current_second_molecule(self, residue="SOL"):
        # translate the atoms of the second file and write them
        num_atoms_cen = self.cental_parser.num_atoms
        num_atom = num_atoms_cen + 1
        num_molecule = 2
        hydrogen_counter = 1
        for atom in self.rotating_parser.molecule_set.atoms:
            pos_nm = atom.position
            if atom.element.symbol == "O":
                name = "OW"
            elif atom.element.symbol == "H":
                name = "HW" + str(hydrogen_counter)
                hydrogen_counter += 1
            else:
                name = atom.element.symbol.upper()
            self.f.write(f"{num_molecule:5}{residue:5}{name:>5}{num_atom:5}{pos_nm[0]:8.3f}{pos_nm[1]:8.3f}"
                                f"{pos_nm[2]:8.3f}{0:8.4f}{0:8.4f}{0:8.4f}\n")
            num_atom += 1

    def _write_box(self):
        for box_el in self.cental_parser.box:
            self.f.write(f"\t{box_el}")
        self.f.write("\n")

    def generate_two_molecule_gro(self, translation_nm=0.3):
        # move second molecule for initial dist
        self.rotating_parser.molecule_set.translate([0, 0, translation_nm])
        self._write_current_frame(frame_num=0)
        self.f.close()

    def _write_current_frame(self, frame_num=0, pseudo_database=False):
        #if pseudo_database:
        #    self._add_pseudo_line()
        self._write_comment_num(frame_num=frame_num)
        self._write_first_molecule()
        self._write_current_second_molecule()
        self._write_box()

    def _add_pseudo_line(self):
        pass


if __name__ == "__main__":
    TwoMoleculeGro("protein0", "NA").generate_two_molecule_gro(translation_nm=3)
    # visualize
    # double_file = BaseGroParser(f"{PATH_BASE_GRO_FILES}H2O_H2O.gro", parse_atoms=True)
    # import matplotlib.pyplot as plt
    # from plotting.plotting_helper_functions import set_axes_equal
    # plt.style.use('dark_background')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # double_file.molecule_set.draw(ax)
    # set_axes_equal(ax)
    # plt.show()
