"""
Apply a FullGrid to a ParsedMolecule in a specific sequence.

A Pseudotrajectory takes a ParsedMolecule and a FullGrid and returns a generator that provides this ParsedMolecule in
all combinations of positions/orientations defined by the grid. This class does not deal with any file input/output.
For this purpose, Writers in molgri.writers module are provided.
"""
from typing import Tuple, Generator

from MDAnalysis import Merge, Universe

from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class Pseudotrajectory:

    def __init__(self, molecule1: Universe, molecule2: Universe, full_grid: NDArray):
        """
        A Pseudotrajectory (PT) is a generator of frames in which a molecule assumes new positions in accordance
        with a grid. Initiate with molecule in any position, the method .generate_pseudotrajectory will make sure
        to first center and then correctly position/orient the molecule

        Args:
            molecule: a molecule that will be moved/rotated into all combinations of stated defined by full_grid
            full_grid: an object combining all relevant grids
        """
        self.static_molecule = molecule1.copy()  # Important! Individual instances that get individually manipulated
        self.moving_molecule = molecule2.copy()  # Important! Individual instances that get individually manipulated
        self.full_grid = full_grid
        self.current_frame = 0

    def get_full_grid(self):
        return self.full_grid

    def generate_pseudotrajectory(self) -> Generator[Tuple[int, Universe], None, None]:
        """
        A generator of ParsedMolecule elements, for each frame one. Only deals with the molecule that moves. The
        order of generated structures is the order of 7D coordinates in SE(3) space given by
        self.full_grid.get_full_grid_as_array().

        Yields:
            frame index, molecule with current position attribute
        """
        fg = self.full_grid
        starting_positions = self.moving_molecule.atoms.positions
        for se3_coo in fg:
            self.moving_molecule.atoms.positions = starting_positions
            position = se3_coo[:3]
            orientation = se3_coo[3:]
            rotation_body = Rotation.from_quat(orientation)
            self.moving_molecule.atoms.rotate(rotation_body.as_matrix(), point=self.moving_molecule.atoms.center_of_mass())
            self.moving_molecule.atoms.translate(position)
            merged_universe = Merge(self.static_molecule.atoms, self.moving_molecule.atoms)
            yield self.current_frame, merged_universe
            self.current_frame += 1