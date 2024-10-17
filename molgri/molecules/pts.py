"""
Apply a FullGrid to a ParsedMolecule in a specific sequence.

A Pseudotrajectory takes a ParsedMolecule and a FullGrid and returns a generator that provides this ParsedMolecule in
all combinations of positions/orientations defined by the grid. This class does not deal with any file input/output.
For this purpose, Writers in molgri.writers module are provided.
"""
from typing import Tuple, Generator

import numpy as np
from numpy.typing import NDArray
from MDAnalysis import Merge, Universe
from MDAnalysis.coordinates.memory import MemoryReader
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
        self.pt = None

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

    def get_pt_as_universe(self) -> Universe:
        # only use the generator once
        if self.pt is None:
            # Step 1: Collect the single frames from each universe
            universes = [u for i, u in self.generate_pseudotrajectory()]
            frames = np.array([u.atoms.positions.copy() for u in universes])
            # Step 2: Create a combined universe with a memory-based trajectory
            # Get topology from one of the universes (assuming all have the same topology)

            combined_universe = Universe(universes[0]._topology, frames, format=MemoryReader)
            self.pt = combined_universe
        return self.pt

    def _determine_which_molecule(self, return_mol2=True) -> str:
        num_atoms_m1 = len(self.static_molecule.atoms)
        num_atoms_m2 = len(self.moving_molecule.atoms)
        num_atoms_total = num_atoms_m1+num_atoms_m2
        # indexing in MDAnalysis is 1-based, indices are inclusive
        if return_mol2:
            return f"bynum  {num_atoms_m1+1}:{num_atoms_total+1}"
        else:
            return f"bynum 1:{num_atoms_m1}"

    def get_one_molecule_pt_as_universe(self, return_mol2=True) -> Universe:
        pt_universe = self.get_pt_as_universe()
        selected_atoms = pt_universe.select_atoms(self._determine_which_molecule(return_mol2=return_mol2))

        # Step 2: Create a new universe with only the protein selection and the trajectory

        selected_frames = []

        # Loop over the trajectory to store positions of selected atoms at each time step
        for _ in pt_universe.trajectory:
            selected_frames.append(selected_atoms.positions.copy())

        selected_frames = np.array(selected_frames)

        if return_mol2:
            selected_universe = Universe(self.moving_molecule._topology, selected_frames, format=MemoryReader)
        else:
            selected_universe = Universe(self.static_molecule._topology, selected_frames, format=MemoryReader)
        return selected_universe
