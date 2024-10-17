"""
Write pseudo-trajectories to files.

This module contains PtWriter object to write Pseudotrajectories to trajectory/topology files and a PtIOManager that
is a high-level function combining: parsing from input files, creating grids and writing the outputs.
"""
import os

# noinspection PyPep8Naming
import MDAnalysis as mda
from MDAnalysis import Merge
import numpy as np
import pandas as pd

from molgri.constants import EXTENSION_TRAJECTORY, EXTENSION_TOPOLOGY, EXTENSION_LOGGING
from molgri.logfiles import PtLogger, paths_free_4_all
from molgri.space.fullgrid import FullGrid
from molgri.molecules.parsers import FileParser, ParsedMolecule
from molgri.paths import PATH_INPUT_BASEGRO, PATH_OUTPUT_PT, PATH_OUTPUT_LOGGING
from molgri.molecules.pts import Pseudotrajectory
from molgri.wrappers import time_method



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

    def __init__(self, name_to_save: str, parsed_central_molecule: ParsedMolecule, box):
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
        self.box = box
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
