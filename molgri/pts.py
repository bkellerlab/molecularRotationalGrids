"""
A Pseudotrajectory takes a ParsedMolecule and a FullGrid and returns a generator that provides this ParsedMolecule in
all combinations of positions/orientations defined by the grid. This class does not deal with any file input/output.
For this purpose, Writers in molgri.writers module are provided.
"""

from typing import Tuple, Generator

from molgri.parsers import ParsedMolecule
from molgri.grids import FullGrid


class Pseudotrajectory:

    def __init__(self, molecule: ParsedMolecule, full_grid: FullGrid):
        """
        A Pseudotrajectory (PT) is a generator of frames in which a molecule assumes new positions in accordance
        with a grid. Initiate with molecule in any position, the method .generate_pseudotrajectory will make sure
        to first center and then correctly position/orient the molecule

        Args:
            molecule: a molecule that will be moved/rotated into all combinations of stated defined by full_grid
            full_grid: an object combining all relevant grids
        """
        self.molecule = molecule
        self.trans_grid = full_grid.t_grid
        self.rot_grid_origin = full_grid.o_grid.as_rotation_object()
        self.rot_grid_body = full_grid.b_grid.as_rotation_object()
        self.current_frame = 0

    def get_molecule(self):
        return self.molecule

    def generate_pseudotrajectory(self) -> Generator[Tuple[int, ParsedMolecule], None, None]:
        """
        A generator of ParsedMolecule elements, for each frame one. Only deals with the molecule that moves.

        Yields:
            frame index, molecule with current position attribute
        """
        # center second molecule if not centered yet
        self.molecule.translate_to_origin()
        trans_increments = self.trans_grid.get_increments()
        increment_sum = self.trans_grid.sum_increments_from_first_radius()
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
