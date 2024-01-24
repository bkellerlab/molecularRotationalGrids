"""
Apply a FullGrid to a ParsedMolecule in a specific sequence.

A Pseudotrajectory takes a ParsedMolecule and a FullGrid and returns a generator that provides this ParsedMolecule in
all combinations of positions/orientations defined by the grid. This class does not deal with any file input/output.
For this purpose, Writers in molgri.writers module are provided.
"""
from typing import Tuple, Generator

import numpy as np
from scipy.spatial.transform import Rotation

from molgri.molecules.parsers import ParsedMolecule
from molgri.space.fullgrid import FullGrid
from molgri.space.utils import normalise_vectors


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
        self.full_grid = full_grid
        self.position_grid = full_grid.get_position_grid()
        self.rot_grid_body = full_grid.get_body_rotations()
        self.current_frame = 0

    def get_full_grid(self):
        return self.full_grid

    def get_molecule(self):
        return self.molecule

    def generate_pseudotrajectory(self) -> Generator[Tuple[int, ParsedMolecule], None, None]:
        """
        A generator of ParsedMolecule elements, for each frame one. Only deals with the molecule that moves. The
        order of generated structures is the order of 7D coordinates in SE(3) space given by
        self.full_grid.get_full_grid_as_array().

        Yields:
            frame index, molecule with current position attribute
        """
        fg = self.full_grid.get_full_grid_as_array()
        # center second molecule if not centered yet
        self.molecule.translate_to_origin()

        for se3_coo in fg:
            position = se3_coo[:3]
            orientation = se3_coo[3:]
            rotation_body = Rotation.from_quat(orientation)

            # position and orient as desired
            self.molecule.rotate_about_body(rotation_body)
            self.molecule.translate_radially(np.linalg.norm(position))
            self.molecule.double_rotate(normalise_vectors(position))

            yield self.current_frame, self.molecule
            self.current_frame += 1
            # rotate back
            self.molecule.double_rotate(normalise_vectors(position), inverse=True)
            self.molecule.translate_radially(-np.linalg.norm(position))
            self.molecule.rotate_about_body(rotation_body, inverse=True)

