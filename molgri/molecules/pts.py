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
from molgri.space.utils import normalise_vectors, q_in_upper_sphere


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

    def determine_positive_directions(self):
        pas = self.molecule.atoms.principal_axes()
        com = self.molecule.atoms.center_of_mass()
        directions = [0, 0, 0]
        for atom_pos in self.molecule.atoms.positions:
            for i, pa in enumerate(pas):
                cosalpha = pa.dot(atom_pos-com)/np.linalg.norm(pa-com)/np.linalg.norm(atom_pos-com)
                #print(cosalpha)
                directions[i] = np.sign(cosalpha)
            if not np.any(np.isclose(directions,0)):
                break
        return directions


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


        #self.molecule.atoms.align_principal_axis(0, [1, 0, 0])
        #self.molecule.atoms.align_principal_axis(1, [0, 1, 0])
        #self.molecule.atoms.align_principal_axis(2, [0, 0, 1])

        starting_positions = self.molecule.atoms.positions
        initial_direction = self.determine_positive_directions()
        print(initial_direction)
        # TODO: force starting principal axes to be np.eye?
        print("start", np.round(self.molecule.atoms.principal_axes().T, 3))
        for se3_coo in fg:
            self.molecule.atoms.positions = starting_positions
            position = se3_coo[:3]
            orientation = se3_coo[3:]
            rotation_body = Rotation.from_quat(orientation)
            R = rotation_body.as_matrix()

            QR = self.molecule.atoms.principal_axes().T
            print("quat", np.round(orientation, 3))
            #print("QR", np.round(QR, 3))
            #print("R", np.round(R, 3))

            self.molecule.atoms.rotate(rotation_body.as_matrix(), point=self.molecule.atoms.center_of_mass())
            self.molecule.atoms.translate(position)

            QC = self.molecule.atoms.principal_axes().T

            direction = self.determine_positive_directions()
            #direction = np.tile(direction, (3, 1))
            #print("direction", np.array(direction), np.array(initial_direction))
            #print("R@QR", direction[0]*np.round(R@QR, 3)[:, 0])
            #print("QC", np.round(QC, 3)[:, 0])
            #print(np.allclose(direction[2]*np.round(R@QR, 3)[:, 2], np.round(QC, 3)[:, 2]))
            #print("missing factor", np.round(QC.T @ np.linalg.inv(R@QR).T, 3))
            #print("R", rotation_body.as_matrix())
            produkt = np.multiply(QC, np.tile(direction, (3, 1))) @ np.linalg.inv(QR)
            calc_quat = np.round(Rotation.from_matrix(produkt).as_quat(), 3)
            if not q_in_upper_sphere(calc_quat):
                calc_quat = -calc_quat
            print("end", calc_quat)
            print(np.allclose(orientation, calc_quat, atol=1e-3, rtol=1e-3))
            yield self.current_frame, self.molecule
            self.current_frame += 1



