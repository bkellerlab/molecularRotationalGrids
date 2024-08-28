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

    def double_rotate(self, position:NDArray, inverse=False):
        """
        The starting position is at (0, 0, 1). We perform two rotations: first around the y-axis till the wished z-coo
        is reached, then around z-axis till the wished x, y, z coordinates are reached.
        Args:
            position ():

        Returns:

        """
        x = position[0]
        y = position[1]
        z = position[2]
        b = np.sqrt(1 - z ** 2)
        first_matrix = Rotation.from_matrix(np.array([[z, 0, b], [0, 1, 0], [-b, 0, z]]))
        # special cases z == +-1 -> b==0, x==0, y==0
        if np.isclose(np.abs(z), 1):
            assert np.allclose([x, y], 0)
            second_matrix = Rotation.from_matrix(np.eye(3))
        else:
            second_matrix = Rotation.from_matrix(np.array([[x/b, -y/b, 0], [y/b, x/b, 0], [0, 0, 1]]))
        if inverse:
            self.rotate_about_origin(second_matrix, inverse=True)
            self.rotate_about_origin(first_matrix, inverse=True)
        else:
            # TODO: wtf is happening here
            #assert np.allclose(self.get_center_of_mass()[:2], [0, 0], atol=2e-5), f"{self.get_center_of_mass()[:2]}"
            self.rotate_about_origin(first_matrix)
            self.rotate_about_origin(second_matrix)

    def rotate_to(self, position: NDArray):
        """
        1) rotate around origin to get to a rotational position described by position
        2) rescale radially to original length of position vector

        Args:
            position: 3D coordinates of a point on a sphere, end position of COM
        """
        assert len(position) == 3, "Position must be a 3D location in space"
        com_radius = np.linalg.norm(self.get_center_of_mass())
        position_radius = np.linalg.norm(position)
        self.translate_radially(position_radius - com_radius)
        rot_matrix = two_vectors2rot(self.get_center_of_mass(), position)
        self.rotate_about_origin(Rotation.from_matrix(rot_matrix))


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


class FileParser:

    def __init__(self, path_topology: str, path_trajectory: str = None, path_energy: str = None):
        """
        This module serves to load topology or trajectory information and output it in a standard format of
        a ParsedMolecule or a generator of ParsedMolecules (one per frame). No properties should be accessed directly,
        all work in other modules should be based on attributes and methods of ParsedMolecule.

        Args:
            path_topology: a full path to the structure file (.gro, .xtc, .xyz ....) that should be read
            path_trajectory: a full path to the trajectory file (.xtc, .xyz ....) that should be read
        """
        self.path_topology = path_topology
        self.path_trajectory = path_trajectory
        self.path_energy = self._try_find_energy_file(path_energy)
        self.path_topology, self.path_trajectory = self._try_to_add_extension()
        if self.path_trajectory is not None and self.path_topology is not None:
            self.universe = mda.Universe(self.path_topology, self.path_trajectory)
        elif self.path_topology is not None:
            self.universe = mda.Universe(self.path_topology)
        elif self.path_trajectory is not None:
            self.universe = mda.Universe(self.path_trajectory)
        else:
            raise ValueError("FileParser cannot produce any output if path_topology and path_trajectory are None.")

    def _try_find_energy_file(self, path_energy):
        # file exists and is provided
        if path_energy is not None and os.path.isfile(path_energy):
            return path_energy
        else:
            # try based on other names
            name_getters = [self.get_topology_file_name(), self.get_trajectory_file_name()]
            for proposed_name in name_getters:
                if proposed_name is not None:
                    proposed_path = f"{PATH_OUTPUT_ENERGIES}{proposed_name}.xvg"
                    if os.path.isfile(proposed_path):
                        return proposed_path
        return None

    def _try_to_add_extension(self) -> List[str]:
        """
        If no extension is added, the program will try to add standard extensions. If no or multiple files with
        standard extensions are added, it will raise an error. It is always possible to explicitly state an extension.
        """
        possible_exts_structure = EXTENSIONS_STR
        possible_exts_traj = EXTENSIONS_TRJ
        possible_exts = (possible_exts_structure, possible_exts_traj)
        to_return = [self.path_topology, self.path_trajectory]
        for i, path in enumerate(to_return):
            if path is None:
                continue
            root, ext = os.path.splitext(path)
            if not ext:
                options = []
                for possible_ext in possible_exts[i]:
                    test_file = f"{path}.{possible_ext.lower()}"
                    if os.path.isfile(test_file):
                        options.append(test_file)
                # if no files found, complain
                if len(options) == 0:
                    raise FileNotFoundError(f"No file with name {path} could be found and adding standard "
                                            f"extensions did not help.")
                # if exactly one file found, use it
                elif len(options) == 1:
                    to_return[i] = options[0]
                # if several files with same name but different extensions found, complain
                else:
                    raise AttributeError(f"More than one file corresponds to the name {path}. "
                                         f"Possible choices: {options}. "
                                         f"Please specify the extension you want to use.")
        return to_return

    def get_dt(self):
        "Time in ps between trajectory frames (assuming constant time step)"
        return self.universe.trajectory[1].time - self.universe.trajectory[0].time

    def get_file_path_trajectory(self):
        return self.path_trajectory

    def get_file_path_topology(self):
        return self.path_topology

    def get_topology_file_name(self):
        if self.path_topology is None:
            return None
        return Path(self.path_topology).stem

    def get_trajectory_file_name(self):
        if self.path_trajectory is None:
            return None
        return Path(self.path_trajectory).stem

    def get_parsed_trajectory(self, default_atom_selection=None):
        if not self.path_energy:
            parsed_energies = None
        else:
            xvg_parser = XVGParser(self.path_energy)
            parsed_energies = xvg_parser.get_parsed_energy()

        name = self.get_topology_file_name()
        molecule_generator = self.generate_frame_as_molecule
        return ParsedTrajectory(name, molecule_generator, parsed_energies, is_pt=isinstance(self, PtParser),
                                default_atom_selection=default_atom_selection, dt=self.get_dt())

    def as_parsed_molecule(self) -> ParsedMolecule:
        """
        Method to use if you want to parse only a single molecule, eg. if only a topology file was provided.

        Returns:
            a parsed molecule from current or only frame
        """
        return ParsedMolecule(self.universe.atoms, box=self.universe.dimensions)

    def generate_frame_as_molecule(self, atom_selection = None) -> Generator[ParsedMolecule, None, None]:
        """
        Method to use if you want to parse an entire trajectory and generate a ParsedMolecule for each frame.

        Returns:
            a generator yielding each frame individually
        """
        if atom_selection is None:
            atom_selection = "all"
        for _ in self.universe.trajectory:
            yield ParsedMolecule(self.universe.select_atoms(atom_selection, sorted=False), box=self.universe.dimensions)


class PtParser(FileParser):

    def __init__(self, m1_path: str, m2_path: str, path_topology: str, path_trajectory: str = None,
                 path_energy: str = None):
        """
        A parser specifically for Pseudotrajectories written with the molgri.writers module. Useful to test the
        behaviour of writers or to analyse the resulting pseudo-trajectories.

        Args:
            m1_path: full path to the topology of the central molecule
            m2_path: full path to the topology of the rotating molecule
            path_topology: full path to the topology of the combined system
            path_trajectory: full path to the trajectory of the combined system
        """
        super().__init__(path_topology=path_topology, path_trajectory=path_trajectory, path_energy=path_energy)
        self.c_num = FileParser(m1_path).as_parsed_molecule().num_atoms
        self.r_num = FileParser(m2_path).as_parsed_molecule().num_atoms
        assert self.c_num + self.r_num == self.as_parsed_molecule().num_atoms

    def generate_frame_as_double_molecule(self) -> Generator[Tuple[ParsedMolecule, ParsedMolecule], None, None]:
        """
        Differently to FileParser, this generator always yields a pair of molecules: central, rotating (for each frame).

        Yields:
            (central molecule, rotating molecule) at each frame of PT

        """
        for _ in self.universe.trajectory:
            c_molecule = ParsedMolecule(self.universe.atoms[:self.c_num], box=self.universe.dimensions)
            r_molecule = ParsedMolecule(self.universe.atoms[self.c_num:], box=self.universe.dimensions)
            yield c_molecule, r_molecule

    def get_parsed_trajectory(self):
        pt = super().get_parsed_trajectory()
        pt.set_c_r(c_num=self.c_num, r_num=self.r_num)
        return pt

    def generate_r_molecule(self) -> Generator[ParsedMolecule, None, None]:
        return self.generate_frame_as_molecule(atom_selection=self.get_atom_selection_r())

    def generate_c_molecule(self) -> Generator[ParsedMolecule, None, None]:
        return self.generate_frame_as_molecule(atom_selection=self.get_atom_selection_c())


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

