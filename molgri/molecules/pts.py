"""
Apply a FullGrid to a ParsedMolecule in a specific sequence.

A Pseudotrajectory takes a ParsedMolecule and a FullGrid and returns a generator that provides this ParsedMolecule in
all combinations of positions/orientations defined by the grid. This class does not deal with any file input/output.
For this purpose, Writers in molgri.writers module are provided.
"""
import os
from typing import Tuple, Generator

import MDAnalysis as mda
import numpy as np
from MDAnalysis import Merge

from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from molgri.molecules.parsers import ParsedMolecule


class Pseudotrajectory:

    def __init__(self, molecule: ParsedMolecule, full_grid: NDArray):
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
        fg = self.full_grid
        # center second molecule if not centered yet
        self.molecule.translate_to_origin()
        starting_positions = self.molecule.atoms.positions
        for se3_coo in fg:
            self.molecule.atoms.positions = starting_positions
            position = se3_coo[:3]
            orientation = se3_coo[3:]
            rotation_body = Rotation.from_quat(orientation)
            self.molecule.atoms.rotate(rotation_body.as_matrix(), point=self.molecule.atoms.center_of_mass())
            self.molecule.atoms.translate(position)
            yield self.current_frame, self.molecule
            self.current_frame += 1


def _create_dir_or_empty_it(directory_name):
    """
    Helper function that determines the name of the directory in which single frames of the trajectory are
    saved. If the directory already exists, its previous contents are deleted.

    Returns:
        path to the directory
    """
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        # delete contents if folder already exist
        filelist = [f for f in os.listdir(directory_name)]
        for f in filelist:
            os.remove(os.path.join(directory_name, f))


class PtWriter:

    def __init__(self, name_to_save: str, parsed_central_molecule: ParsedMolecule):
        """
        This class writes a pseudotrajectory to a file. A PT consists of one molecule that is stationary at
        origin and one that moves with every time step. The fixed molecule is provided when the class is created
        and the mobile molecule as a generator when the method write_full_pt is called. Writing is done with
        MDAnalysis module, so all formats implemented there are supported.

        Args:
            name_to_save: base name of the PT file without paths or extensions
            parsed_central_molecule: a ParsedMolecule object describing the central molecule, will only be translated
                                     so that COM lies at (0, 0, 0) but not manipulated in any other way.
        """
        self.central_molecule = parsed_central_molecule
        self.central_molecule.translate_to_origin()
        self.box = self.central_molecule.get_box()
        self.file_name = name_to_save

    def _merge_and_write(self, writer: mda.Writer, pt: Pseudotrajectory):
        """
        Helper function to merge Atoms from central molecule with atoms of the moving molecule (at current positions)

        Args:
            writer: an already initiated object writing to file (eg a .gro or .xtc file)
            pt: a Pseudotrajectory object with method .get_molecule() that returns current ParsedMolecule
        """
        merged_universe = Merge(self.central_molecule.get_atoms(), pt.get_molecule().get_atoms())
        merged_universe.dimensions = self.box
        writer.write(merged_universe)

    def write_structure(self, pt: Pseudotrajectory, path_structure: str):
        """
        Write the one-frame topology file, eg in .gro format.

        Args:
            pt: a Pseudotrajectory object with method .get_molecule() that returns current ParsedMolecule
            path_structure: where topology should be saved
        """
        if not np.all(self.box == pt.get_molecule().get_box()):
            print(f"Warning! Simulation boxes of both molecules are different. Selecting the box of "
                  f"central molecule with dimensions {self.box}")
        with mda.Writer(path_structure) as structure_writer:
            self._merge_and_write(structure_writer, pt)

    def write_full_pt(self, pt: Pseudotrajectory, path_trajectory: str, path_structure: str):
        """
        Write the trajectory file as well as the structure file (only at the first time-step).

        Args:
            pt: a Pseudotrajectory object with method .generate_pseudotrajectory() that generates ParsedMolecule objects
            path_trajectory: where trajectory should be saved
            path_structure: where topology should be saved
        """
        n_atoms = len(self.central_molecule.atoms) + len(pt.molecule.atoms)
        trajectory_writer = mda.Writer(path_trajectory, n_atoms=n_atoms, multiframe=True)
        last_i = 0
        for i, _ in pt.generate_pseudotrajectory():
            # for the first frame write out topology
            if i == 0:
                self.write_structure(pt, path_structure)
            self._merge_and_write(trajectory_writer, pt)
            last_i = i
        product_of_grids = len(pt.get_full_grid())
        assert last_i + 1 == product_of_grids, f"Length of PT not correct, {last_i}=/={product_of_grids}"
        trajectory_writer.close()

    def write_frames_in_directory(self, pt: Pseudotrajectory, path_trajectory: str, path_structure: str):
        """
        As an alternative to saving a full PT in a single trajectory file, you can create a directory with the same
        name and within it single-frame trajectories named with their frame index. Also save the structure file at
        first step.

            pt: a Pseudotrajectory object with method .generate_pseudotrajectory() that generates ParsedMolecule objects
            path_trajectory: where trajectory should be saved
            path_structure: where topology should be saved
        """
        directory_name, extension_trajectory = os.path.splitext(path_trajectory)
        _create_dir_or_empty_it(directory_name)
        for i, _ in pt.generate_pseudotrajectory():
            if i == 0:
                self.write_structure(pt, path_structure)
            f = f"{directory_name}/{i}{extension_trajectory}"
            with mda.Writer(f) as structure_writer:
                self._merge_and_write(structure_writer, pt)