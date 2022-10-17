import numpy as np

from .grids import Grid
from .constants import DEFAULT_DISTANCES
from .parsers import BaseGroParser, TranslationParser
from .paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT
from .wrappers import time_method


class TwoMoleculeGro:

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
        self.f = open(result_file_path, "w")
        self.central_parser = BaseGroParser(central_file_path, parse_atoms=False)
        # parse rotating file as Atoms
        self.rotating_parser = BaseGroParser(rotating_file_path, parse_atoms=True)

    def _write_comment_num(self, frame_num=0):
        num_atoms_cen = self.central_parser.num_atoms
        num_atoms_rotating = self.rotating_parser.num_atoms
        # write comment
        self.f.write(f"c_num={num_atoms_cen}, r_num={num_atoms_rotating}, t={frame_num}\n")
        # write total number of atoms
        self.f.write(f"{num_atoms_cen + num_atoms_rotating:5}\n")

    def _write_first_molecule(self):
        self.f.writelines(self.central_parser.atom_lines_nm)

    def _write_current_second_molecule(self, residue="SOL"):
        # translate the atoms of the second file and write them
        num_atoms_cen = self.central_parser.num_atoms
        num_atom = num_atoms_cen + 1
        num_molecule = 2
        hydrogen_counter = 1
        for atom in self.rotating_parser.molecule_set.atoms:
            pos_nm = atom.position
            name = atom.gro_label
            self.f.write(f"{num_molecule:5}{residue:5}{name:>5}{num_atom:5}{pos_nm[0]:8.3f}{pos_nm[1]:8.3f}"
                                f"{pos_nm[2]:8.3f}{0:8.4f}{0:8.4f}{0:8.4f}\n")
            num_atom += 1

    def _write_box(self):
        for box_el in self.central_parser.box:
            self.f.write(f"\t{box_el}")
        self.f.write("\n")

    def generate_two_molecule_gro(self, translation_nm=0.3):
        # move second molecule for initial dist
        self.rotating_parser.molecule_set.translate([0, 0, translation_nm])
        self._write_current_frame(frame_num=0)
        self.f.close()

    def _write_current_frame(self, frame_num=0):
        self._write_comment_num(frame_num=frame_num)
        self._write_first_molecule()
        self._write_current_second_molecule()
        self._write_box()

    def _add_pseudo_line(self):
        pass


class Pseudotrajectory(TwoMoleculeGro):

    def __init__(self, name_central_gro: str, name_rotating_gro: str, rot_grid: Grid, trans_grid: TranslationParser,
                 traj_type="full"):
        grid_name = rot_grid.standard_name  # for example ico_500
        pseudo_name = f"{name_central_gro}_{name_rotating_gro}_{grid_name}_{traj_type}"
        self.pt_name = pseudo_name
        super().__init__(name_central_gro, name_rotating_gro, result_name_gro=pseudo_name)
        self.quaternions = rot_grid.as_quaternion()
        self.trans_grid = trans_grid
        self.traj_type = traj_type
        self.name_rotating = name_rotating_gro
        self.decorator_label = f"pseudotrajectory {pseudo_name}"

    def _gen_trajectory(self, frame_index=0) -> int:
        """
        This does not deal with any radii yet, only with rotations.
        Args:
            frame_index: index of the last frame written

        Returns:
            the new frame index after all rotations completed
        """
        frame_index = frame_index
        for one_rotation in self.quaternions:
            initial_atom_set = self.rotating_parser.molecule_set
            initial_atom_set.rotate_about_origin(one_rotation, method="quaternion")
            self._write_current_frame(frame_num=frame_index)
            frame_index += 1
            if self.traj_type == "full":
                for body_rotation in self.quaternions:
                    # rotate there
                    initial_atom_set.rotate_about_body(body_rotation, method="quaternion")
                    self._write_current_frame(frame_num=frame_index)
                    # rotate back
                    initial_atom_set.rotate_about_body(body_rotation, method="quaternion", inverse=True)
                    frame_index += 1
            initial_atom_set.rotate_about_origin(one_rotation, method="quaternion", inverse=True)
        return frame_index

    def generate_pseudotrajectory(self) -> int:
        index = 0
        trans_increments = self.trans_grid.get_increments()
        # initial set-up of molecules
        self.rotating_parser.molecule_set.translate([0, 0, trans_increments[0]])
        self._write_current_frame(index)
        index += 1
        # go over different radii
        for shell_d in trans_increments[1:]:
            index = self._gen_trajectory(frame_index=index)
            self.rotating_parser.molecule_set.translate([0, 0, shell_d])
        self.f.close()
        return index

    @time_method
    def generate_pt_and_time(self) -> int:
        return self.generate_pseudotrajectory()
