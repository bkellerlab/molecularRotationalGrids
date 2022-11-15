import numpy as np

from .grids import Grid, ZeroGrid
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
        for atom in self.rotating_parser.molecule_set.all_objects[0].atoms:
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

    def __init__(self, name_central_gro: str, name_rotating_gro: str, rot_grid_origin: Grid,
                 trans_grid: TranslationParser, rot_grid_body: Grid or None):
        """

        Args:
            name_central_gro: name of the .gro file with the central (fixed) molecule that can be found at location
                              specified by PATH_INPUT_BASEGRO (in paths.py)
            name_rotating_gro: name of the .gro file with the second (movable) molecule that can be found at location
                              specified by PATH_INPUT_BASEGRO (in paths.py)
            rot_grid_origin: name of the grid for rotation around origin, eg. ico_15
            trans_grid: a parser for translations
            rot_grid_body: name of the grid for rotation around origin, eg. ico_20, or None if no body rotations
                           are included (e.g. for spherically symmetrical rotating particles
        """
        self.trans_grid = trans_grid
        self.rot_grid_origin = rot_grid_origin
        if not rot_grid_body:
            zg = ZeroGrid()
            self.rot_grid_body = zg
        else:
            self.rot_grid_body = rot_grid_body
        self.name_rotating = name_rotating_gro
        origin_grid_name = self.rot_grid_origin.standard_name  # for example ico_500
        body_grid_name = self.rot_grid_body.standard_name
        pseudo_name = f"{name_central_gro}_{name_rotating_gro}_o_{origin_grid_name}_b_{body_grid_name}_t_{trans_grid.grid_hash}"
        self.pt_name = pseudo_name
        self.decorator_label = f"pseudotrajectory {pseudo_name}"
        super().__init__(name_central_gro, name_rotating_gro, result_name_gro=pseudo_name)
        # convert to quaternions
        self.rot_grid_body = self.rot_grid_body.as_quaternion()
        self.rot_grid_origin = self.rot_grid_origin.as_quaternion()

    def generate_pseudotrajectory(self) -> int:
        # center second molecule if not centered yet
        dist_origin = np.linalg.norm(self.rotating_parser.molecule_set.position)
        self.rotating_parser.molecule_set.translate_objects_radially(-dist_origin)
        index = 0
        trans_increments = self.trans_grid.get_increments()
        increment_sum = self.trans_grid.sum_increments_from_first_radius()
        self.rotating_parser.molecule_set.translate_objects_radially(trans_increments[0])
        for origin_rotation in self.rot_grid_origin:
            initial_mol_set = self.rotating_parser.molecule_set
            initial_mol_set.rotate_objects_about_origin(origin_rotation, method="quaternion")
            for body_rotation in self.rot_grid_body:
                initial_mol_set.rotate_objects_about_body(body_rotation, method="quaternion")
                # write the frame at initial distance
                self._write_current_frame(frame_num=index)
                index += 1
                for translation in trans_increments[1:]:
                    self.rotating_parser.molecule_set.translate_objects_radially(translation)
                    self._write_current_frame(frame_num=index)
                    index += 1
                self.rotating_parser.molecule_set.translate_objects_radially(-increment_sum)
                initial_mol_set.rotate_objects_about_body(body_rotation, method="quaternion", inverse=True)
            initial_mol_set.rotate_objects_about_origin(origin_rotation, method="quaternion", inverse=True)
        self.f.close()
        return index

    @time_method
    def generate_pt_and_time(self) -> int:
        return self.generate_pseudotrajectory()
