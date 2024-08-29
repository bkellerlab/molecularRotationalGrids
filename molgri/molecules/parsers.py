"""
Parse files and energies - wrap MDAnalysis functions.

The parsers module deals with input. This can be a trajectory or single-frame topology file (FileParser),
specifically a Pseudotrajectory file (PtParser) or energy write-out (XVGParser).

The result of file parsing is immediately transformed into a ParsedMolecule or a sequence of ParsedMolecule
objects -> ParsedTrajectory. Similarly, energies are stored in the ParsedEnergy object that can be connected to
ParsedTrajectory. Those are the objects that other modules should access.
"""

import os
from copy import copy
from pathlib import Path
from typing import Generator, Tuple, List, Callable

import numpy as np

import pandas as pd
from MDAnalysis.auxiliary.XVG import XVGReader
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import MDAnalysis as mda
from MDAnalysis.core import AtomGroup

from molgri.constants import EXTENSIONS_STR, EXTENSIONS_TRJ
from molgri.paths import PATH_OUTPUT_ENERGIES
from molgri.space.utils import two_vectors2rot


class ParsedMolecule:

    def __init__(self, atoms: AtomGroup, box=None):
        """
        Wraps the behaviour of AtomGroup and Box implemented by MDAnalysis and provides the necessary
        rotation and translation functions.

        Args:
            atoms: an AtomGroup that makes up the molecule
            box: the box in which the molecule is simulated
        """
        self.atoms = atoms
        self.box = box
        self._update_attributes()

    def _update_attributes(self):
        self.num_atoms = len(self.atoms)
        self.atom_labels = self.atoms.names
        self.atom_types = self.atoms.types

    def get_atoms(self) -> AtomGroup:
        return self.atoms

    def get_center_of_mass(self) -> NDArray:
        return self.atoms.center_of_mass()

    def get_positions(self) -> NDArray:
        return self.atoms.positions

    def get_box(self):
        return self.box

    def __str__(self):
        return f"<ParsedMolecule with atoms {self.atom_types} at {self.get_center_of_mass()}>"

    def _rotate(self, about_what: str, rotational_obj: Rotation, inverse: bool = False):
        if about_what == "origin":
            point = (0, 0, 0)
        elif about_what == "body":
            point = self.get_center_of_mass()
        else:
            raise ValueError(f"Rotation about {about_what} unknown, try 'origin' or 'body'.")
        if inverse:
            rotational_obj = rotational_obj.inv()
        self.atoms.rotate(rotational_obj.as_matrix(), point=point)

    def rotate_about_origin(self, rotational_obj: Rotation, inverse: bool = False):
        self._rotate(about_what="origin", rotational_obj=rotational_obj, inverse=inverse)

    def rotate_about_body(self, rotational_obj: Rotation, inverse: bool = False):
        self._rotate(about_what="body", rotational_obj=rotational_obj, inverse=inverse)

    def translate(self, vector: np.ndarray):
        self.atoms.translate(vector)



    def translate_to_origin(self):
        """
        Translates the object so that the COM is at origin. Does not change the orientation.
        """
        current_position = self.get_center_of_mass()
        self.translate(-current_position)

    def translate_radially(self, distance_change: float):
        """
        Moves the object away from the origin in radial direction for the amount specified by distance_change (or
        towards the origin if a negative distance_change is given). If the object is at origin, translate in z-direction
        of the internal coordinate system.

        Args:
            distance_change: the change in length of the vector origin-object
        """
        # need to work with rounding because gromacs files only have 3-point precision
        initial_vector = np.round(self.get_center_of_mass(), 3)
        if np.allclose(initial_vector, [0, 0, 0], atol=1e-3):
            initial_vector = np.array([0, 0, 1])
        len_initial = np.linalg.norm(initial_vector)
        rescaled_vector = distance_change*initial_vector/len_initial
        self.atoms.translate(rescaled_vector)



class ParsedEnergy:

    def __init__(self, energies: NDArray, labels, unit):
        self.energies = energies
        self.labels = labels
        self.unit = unit

    def get_energies(self, energy_type: str):
        if energy_type not in self.labels:
            raise ValueError(f"Energy type {energy_type} not available, choose from: {self.labels}")
        i = self.labels.index(energy_type)
        return self.energies[:, i]


class XVGParser(object):

    def __init__(self, path_xvg: str):
        # this is done in order to function with .xvg ending or with no ending
        if not path_xvg.endswith(".xvg"):
            self.path_name = f"{path_xvg}.xvg"
        else:
            self.path_name = path_xvg
        reader = XVGReader(self.path_name)
        self.all_values = reader._auxdata_values
        reader.close()

    def get_parsed_energy(self):
        return ParsedEnergy(self.get_all_columns(), self.get_all_labels(), self.get_y_unit())

    def get_all_columns(self) -> NDArray:
        return self.all_values

    def get_all_labels(self) -> tuple:
        result = ["None"]
        with open(self.path_name, "r") as f:
            for line in f:
                # parse column number
                for i in range(0, 10):
                    if line.startswith(f"@ s{i} legend"):
                        split_line = line.split('"')
                        result.append(split_line[-2])
                if not line.startswith("@") and not line.startswith("#"):
                    break
        return tuple(result)

    def get_y_unit(self) -> str:
        y_unit = None
        with open(self.path_name, "r") as f:
            for line in f:
                # parse property unit
                if line.startswith("@    yaxis  label"):
                    split_line = line.split('"')
                    y_unit = split_line[1]
                    break
        if y_unit is None:
            print("Warning: energy units could not be detected in the xvg file.")
            y_unit = "[?]"
        return y_unit

    def get_column_index_by_name(self, column_label) -> Tuple[str, int]:
        correct_column = None
        with open(self.path_name, "r") as f:
            for line in f:
                # parse column number
                if f'"{column_label}"' in line:
                    split_line = line.split(" ")
                    correct_column = int(split_line[1][1:]) + 1
                if not line.startswith("@") and not line.startswith("#"):
                    break
        if correct_column is None:
            print(f"Warning: a column with label {column_label} not found in the XVG file. Using the first y-axis "
                  f"column instead.")
            column_label = "XVG column 1"
            correct_column = 1
        return column_label, correct_column


class ParsedTrajectory:

    def __init__(self, name: str, molecule_generator: Callable, energies: ParsedEnergy, is_pt=False,
                 default_atom_selection=None, dt=1):
        self.name = name
        self.is_pt = is_pt
        self.molecule_generator = molecule_generator
        self.energies = energies
        self.c_num = None
        self.r_num = None
        self.default_atom_selection = default_atom_selection
        self.dt = dt

    def get_num_unique_com(self, energy_type="Potential", atom_selection=None):
        if atom_selection is None:
            atom_selection = self.default_atom_selection
        return len(self.get_unique_com(energy_type=energy_type, atom_selection=atom_selection)[0])

    def set_c_r(self, c_num: int, r_num: int):
        self.c_num = c_num
        self.r_num = r_num

    def get_all_energies(self, energy_type: str):
        return self.energies.get_energies(energy_type)

    def get_all_COM(self, atom_selection=None) -> NDArray:
        if atom_selection is None:
            atom_selection = self.default_atom_selection
        all_com = []
        for mol in self.molecule_generator(atom_selection):
            all_com.append(mol.get_center_of_mass())
        all_com = np.array(all_com)
        return all_com

    def get_name(self):
        return self.name

    def get_atom_selection_r(self):
        if self.c_num is None or self.r_num is None:
            return None
        return f"id {self.c_num + 1}:{self.c_num + self.r_num}"

    def get_atom_selection_c(self):
        if self.c_num is None or self.r_num is None:
            return None
        return f"id 1:{self.c_num}"

    def get_unique_com(self, energy_type: str = "Potential", atom_selection=None):
        """
        Get only the subset of COMs that have unique positions. Among those, select the ones with lowest energy (if
        energy info is provided)
        """
        if atom_selection is None:
            atom_selection = self.default_atom_selection
        round_to = 3  # number of decimal places
        my_energies = self.energies
        my_coms = self.get_all_COM(atom_selection)

        if my_energies is None or energy_type is None:
            _, indices = np.unique(my_coms.round(round_to), axis=0, return_index=True)
            unique_coms = np.take(my_coms, indices, axis=0)
            return unique_coms, None
        else:
            my_energies = self.energies.get_energies(energy_type)

        # if there are energies, among same COMs, select the one with lowest energy
        coms_tuples = [tuple(row.round(round_to)) for row in my_coms]
        df = pd.DataFrame()
        df["coms_tuples"] = coms_tuples
        df["energy"] = my_energies
        new_df = df.loc[df.groupby(df["coms_tuples"], sort=False)["energy"].idxmin()]
        unique_coms = np.take(my_coms, new_df.index, axis=0)
        unique_energies = np.take(my_energies, new_df.index, axis=0)
        return unique_coms, unique_energies

    def get_only_lowest_highest(self, coms: NDArray, energies: NDArray, lowest_k: int = None, highest_j: int = None):
        """
        After any step that produces a set of COMs and corresponding energies, you may filter out only the (COM, energy)
        with k lowest and/or j highest energies. This is useful for plotting so that only relevant points are visible.
        """
        if lowest_k or highest_j:
            # if not set, just set it to zero
            if lowest_k is None:
                lowest_k = 0

            order = energies.argsort()
            coms = coms[order]
            energies = energies[order]
            # use lowest values
            selected_coms = coms[:lowest_k]
            selected_energies = energies[:lowest_k]
            if highest_j is not None:
                selected_coms = np.concatenate((selected_coms, coms[-highest_j:]))
                selected_energies = np.concatenate((selected_energies, energies[-highest_j:]))
            return selected_coms, selected_energies
        return coms, energies

    def get_unique_com_till_N(self, N: int, energy_type: str = "Potential", atom_selection=None):
        if atom_selection is None:
            atom_selection = self.default_atom_selection
        coms, ens = self.get_unique_com(energy_type=energy_type, atom_selection=atom_selection)
        if ens is None:
            return coms[:N], None
        return coms[:N], ens[:N]


if __name__ == "__main__":
    from molgri.paths import PATH_INPUT_BASEGRO
    from scipy.spatial.transform import Rotation
    my_molecule = FileParser(f"{PATH_INPUT_BASEGRO}H2O.gro").as_parsed_molecule()
    rr = Rotation.random()
    print(rr.as_matrix())
    print(my_molecule.atoms.masses)
    my_molecule.rotate_about_body(rr)
    print(np.round(my_molecule.get_positions()/10, 3))

