"""
A Pseudotrajectory takes a ParsedMolecule and a FullGrid and returns a generator that provides this ParsedMolecule in
all combinations of positions/orientations defined by the grid. This class does not deal with any file input/output.
For this purpose, Writers in molgri.writers module are provided.
"""

from typing import Tuple, Generator

from molgri.molecules.parsers import ParsedMolecule
from molgri.space.fullgrid import FullGrid


class Pseudotrajectory:

    def __init__(self, molecule: ParsedMolecule, full_grid: FullGrid):
        """
        A Pseudotrajectory (PT) is a generator of frames in which a molecule assumes new positions in accordance
        with a grid. Initiate with molecule in any position, the method .generate_pseudotrajectory will make sure
        to first center and then correctly position/orient the molecule

        The first origin rotation is & first rotation about body are performed. All translational distances are covered.
        Then, another body rotation, again at all translational distances tested. When all body rotations are
        exhausted, move on to the next position. As a consequence:

            trajectory[0:N_t*N_b] COM will always be on the same vector from origin
            trajectory[::N_t] will always be at the smallest radius

        Args:
            molecule: a molecule that will be moved/rotated into all combinations of stated defined by full_grid
            full_grid: an object combining all relevant grids
        """
        self.molecule = molecule
        #self.trans_grid = full_grid.t_grid
        #self.rot_grid_origin = full_grid.o_rotations.grid_z # TODO: this could in future be a different (averaged?) grid
        self.position_grid = full_grid.get_position_grid()
        self.rot_grid_body = full_grid.b_rotations.rotations
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
        #trans_increments = self.trans_grid.get_increments()
        #increment_sum = self.trans_grid.sum_increments_from_first_radius()
        # move in z-direction for first increment
        #self.molecule.translate_radially(trans_increments[0])
        # first step: rotation around origin
        #com_before = self.molecule.get_center_of_mass()
        for origin_rotations in self.position_grid:
            #self.molecule.rotate_to(origin_rotation)
            # second step: rotation around body
            for body_rotation in self.rot_grid_body:
                self.molecule.rotate_about_body(body_rotation)
                # write the frame at initial distance
                # yield self.current_frame, self.molecule
                # self.current_frame += 1

                #for radial_translation in
                # final step: translation for the rest of increments
                for origin_rotation in origin_rotations:
                    self.molecule.rotate_to(origin_rotation)
                # for translation in trans_increments[1:]:
                #     self.molecule.translate_radially(translation)
                    yield self.current_frame, self.molecule
                    self.current_frame += 1
                #self.molecule.translate_radially(-increment_sum)
                self.molecule.rotate_about_body(body_rotation, inverse=True)
            #self.molecule.rotate_about_origin(origin_rotation, inverse=True)
