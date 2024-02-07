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
        self.molecule.atoms.align_principal_axis(0, [1, 0, 0])
        #self.molecule.atoms.align_principal_axis(1, [0, 1, 0])
        self.molecule.atoms.align_principal_axis(2, [0, 1, 0])
        print("start", np.round(self.molecule.atoms.principal_axes().T, 3))

        for se3_coo in fg:
            position = se3_coo[:3]
            orientation = se3_coo[3:]
            rotation_body = Rotation.from_quat(orientation)

            QR = self.molecule.atoms.principal_axes().T
            R = rotation_body.as_matrix()
            #print("QR", np.round(QR, 3))
            print("R", np.round(R, 3))

            self.molecule.atoms.rotate(rotation_body.as_matrix(), point=self.molecule.atoms.center_of_mass())
            self.molecule.atoms.translate(position)

            QC = self.molecule.atoms.principal_axes().T
            produkt = R@QR
            #print("QC", np.round(QC, 3))
            #print("produkt", np.round(produkt, 3))

            mom_inertia = self.molecule.atoms.moment_of_inertia()
            eigenval, eigenvec = np.linalg.eig(mom_inertia)
            #print(eigenval)
            print(np.round(eigenvec, 3))

            from MDAnalysis.analysis import align

            #print(np.round(align.rotation_matrix(QR, rotation_body.apply(QC))[0], 3))

            #self.molecule.rotate_about_body(rotation_body)
            #print("(Rc@inv(QR))", np.round((QC @ np.linalg.inv(QR)), 3))
            yield self.current_frame, self.molecule
            self.current_frame += 1
            # rotate back
            rotation_body_i = rotation_body.inv()

            self.molecule.atoms.translate(-position)
            self.molecule.atoms.rotate(rotation_body_i.as_matrix(), point=self.molecule.atoms.center_of_mass())
            #self.molecule.rotate_about_body(rotation_body, inverse=True)



