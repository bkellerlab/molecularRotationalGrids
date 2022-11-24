"""
A Pseudotrajectory takes a Molecule and a FullGrid and returns a generator that provides this Molecule in all
combinations of positions/orientations defined by the grid. This class does not deal with any file input/output.
For this purpose, Writers in molgri.writers module are provided.
"""

from typing import Tuple, Generator

from molgri.parsers import ParsedMolecule
from .constants import NM2ANGSTROM
from .grids import FullGrid
from .wrappers import time_method


class Pseudotrajectory:

    def __init__(self, molecule: ParsedMolecule, full_grid: FullGrid):
        """
        Args:
            molecule: a molecule that will be moved/rotated into all combinations of stated defined by full_grid
            full_grid: an object combining all relevant grids
        """
        self.molecule = molecule
        self.trans_grid = full_grid.t_grid
        self.rot_grid_origin = full_grid.o_grid
        self.rot_grid_body = full_grid.b_grid
        # all the self name stuff
        # TODO: this will all be simplified when i have a better idea how to use FullGrid object
        self.pt_o_name = self.rot_grid_origin.standard_name
        self.pt_b_name = self.rot_grid_body.standard_name
        self.pt_t_name = self.trans_grid.grid_hash
        self.decorator_label = f"Pseudotrajectory {full_grid.get_full_grid_name()}"
        # convert grids to quaternions
        self.rot_grid_body = self.rot_grid_body.as_rot_matrix()
        self.rot_grid_origin = self.rot_grid_origin.as_rot_matrix()
        self.current_frame = 0

    def generate_pseudotrajectory(self) -> Generator[Tuple[int, ParsedMolecule], None, None]:
        # center second molecule if not centered yet
        self.molecule.translate(-self.molecule.get_center_of_mass())
        trans_increments = self.trans_grid.get_increments()*NM2ANGSTROM
        increment_sum = self.trans_grid.sum_increments_from_first_radius()*NM2ANGSTROM
        # move in z-direction for first increment
        self.molecule.translate_radially(trans_increments[0])
        # first step: rotation around origin
        for origin_rotation in self.rot_grid_origin:
            self.molecule.rotate_about_origin(origin_rotation)
            # second step: rotation around body
            for body_rotation in self.rot_grid_body:
                self.molecule.rotate_about_body(body_rotation)
                # write the frame at initial distance
                yield self.current_frame, self.molecule
                self.current_frame += 1
                # final step: translation for the rest of increments
                for translation in trans_increments[1:]:
                    self.molecule.translate_radially(translation)
                    yield self.current_frame, self.molecule
                    self.current_frame += 1
                self.molecule.translate_radially(-increment_sum)
                self.molecule.rotate_about_body(body_rotation, inverse=True)
            self.molecule.rotate_about_origin(origin_rotation, inverse=True)

    @time_method
    def generate_pt_and_time(self) -> Generator[Tuple[int, ParsedMolecule], None, None]:
        return self.generate_pseudotrajectory()
