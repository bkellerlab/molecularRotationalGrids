from __future__ import annotations

import numbers
import pickle
import os

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from MDAnalysis.topology.guessers import guess_masses
from numpy.typing import NDArray
from scipy.sparse import save_npz, sparray, load_npz, spmatrix
import MDAnalysis as md

from molgri.utils.arrays import iter_elements_nested, nested_numpy_types_to_python_types


def write_object(my_object, filename) -> None:
    file_extension = os.path.splitext(filename)[1]
    if isinstance(my_object,np.ndarray) and file_extension != ".txt":
        function = _write_array
    elif isinstance(my_object, sparray) or isinstance(my_object, spmatrix):
        function = _write_sparse_array
    elif isinstance(my_object, nx.Graph):
        function = _write_network
    elif file_extension == ".xtc":
        function = _write_trajectory
    elif file_extension == ".gro":
        function = _write_structure
    elif file_extension == ".xyz":
        function = _write_xyz
    elif file_extension == ".csv":
        function = _write_csv
    elif file_extension == ".txt":
        function = _write_txt
    elif file_extension == ".yaml":
        function = _write_yaml
    else:
        raise TypeError(f"Cannot write object of type {type(my_object)} to a {file_extension} file.")

    function(my_object, filename)

def read_object(filename, **kwargs):
    file_extension = os.path.splitext(filename)[1]
    if file_extension == ".npy":
        function = _read_array
    elif file_extension == ".npz":
        function = _read_sparse_array
    elif file_extension == ".pkl":
        function = _read_network
    elif file_extension == ".gro" or file_extension == ".xyz":
        function = _read_molecular_structure
    elif file_extension == ".xvg":
        function = _read_energy
    elif file_extension == ".csv":
        function = _read_csv
    elif file_extension == ".txt":
        function = _read_txt
    elif file_extension == ".yaml":
        function = _read_yaml
    else:
        raise TypeError(f"Cannot read object from file with extension {file_extension}")
    return function(filename, **kwargs)

def _write_network(network, filename: str) -> nx.Graph:
    with open(filename, "wb") as f:
        pickle.dump(network, f)

def _read_network(filename: str, *args, **kwargs):
    with open(filename, "rb") as f:
        my_network = pickle.load(f)
    return my_network

def _read_energy(filename: str, *args, **kwargs):
    def _get_column_names(filename) -> list:
        result = ["Time [ps]"]
        with open(filename, "r") as f:
            for line in f:
                # parse column number
                for i in range(0, 10):
                    if line.startswith(f"@ s{i} legend"):
                        split_line = line.split('"')
                        result.append(split_line[-2])
                if not line.startswith("@") and not line.startswith("#"):
                    break
        return result

    column_names = _get_column_names(filename)
    # skip 13 rows commented with # and then also a variable amount of rows commented with @
    table = pd.read_csv(filename, sep=r'\s+', comment='@', skiprows=13, header=None, names=column_names)
    return table

def _write_txt(some_array: NDArray, filename: str):
    if np.issubdtype(some_array.dtype, np.integer):
        fmt="%d"
    else:
        fmt="%.12f"

    np.savetxt(filename, some_array, fmt=fmt)

def _read_txt(filename: str, *args, **kwargs) -> NDArray:
    array_or_num = np.loadtxt(filename)
    if np.issubdtype(array_or_num.dtype, np.integer) or np.issubdtype(array_or_num.dtype, float):
        array_or_num = np.array([array_or_num])
    return array_or_num.reshape(-1)

def _write_csv(df, filename: str):
    df.to_csv(filename)

def _read_csv(filename: str, *args, **kwargs) -> pd.DataFrame:
    print(kwargs)
    return pd.read_csv(filename, index_col=0, **kwargs)

def _write_array(array, filename: str):
    np.save(filename, array)

def _read_array(filename: str, *args, **kwargs) -> NDArray:
    return np.load(filename)

def _write_sparse_array(sparse_array, filename: str) -> None:
    save_npz(filename, sparse_array)

def _read_sparse_array(filename: str, *args, **kwargs) -> sparray:
    return load_npz(filename)

def _write_structure(universe, filename: str) -> None:
    universe.atoms.write(filename)

def _write_trajectory(universe, filename: str) -> None:
    with md.coordinates.XTC.XTCWriter(filename, n_atoms=universe.atoms.n_atoms) as W:
        for ts in universe.trajectory:
            W.write(universe.atoms)

def _write_xyz(universe, filename: str) -> None:
    with md.coordinates.XYZ.XYZWriter(filename, n_atoms=universe.atoms.n_atoms) as W:
        for ts in universe.trajectory:
            W.write(universe.atoms)

def _read_molecular_structure(filename: str, *args, **kwargs) -> md.Universe:
    u = md.Universe(filename)

    # guess masses
    u.add_TopologyAttr('elements')
    dict_labels = {"1": "H", "2": "C", "3": "O", "4": "O", "5": "O", "6": "Zr", "OW": "O", "HW1": "H", "HW2": "H",
                   "7": "Ar", "CD1": "C", "CD2": "C", "CE1": "C", "CE2": "C", "CG": "C", "CZ": "C", "HD1": "H",
                   "HD2": "H", "HE1": "H", "HE2": "H", "HG": "H", "HZ": "H"}
    initial_labels = u.atoms.names
    u.atoms.elements = [dict_labels[name] if name in dict_labels.keys() else name for name in u.atoms.names]

    u.add_TopologyAttr('masses')
    u.atoms.masses = guess_masses(u.atoms.elements)

    u.atoms.elements = initial_labels
    return u

def _write_yaml(dict_like_file, filename: str) -> None:
    FlowSeqDumper.add_representer(list, represent_flow_sequence)
    with open(filename, "w") as f:
        yaml.dump(dict_like_file, f, Dumper=FlowSeqDumper, sort_keys=False)

def _read_yaml(filename: str, *args, **kwargs) -> dict:
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data

def get_num_atoms(structure_file:str) -> int:
    file = read_object(structure_file)
    return int(file.atoms.n_atoms)

def get_atomgoup_m1(universe_both: md.Universe, path_str1: str):
    n1 = get_num_atoms(path_str1)

    m1_atoms = universe_both.select_atoms(f"all")
    m1_atoms = m1_atoms[m1_atoms.indices < n1]
    return m1_atoms

def get_atomgoup_m2(universe_both: md.Universe, path_str1: str):
    n1 = get_num_atoms(path_str1)

    m2_atoms = universe_both.select_atoms(f"all")
    m2_atoms = m2_atoms[m2_atoms.indices >= n1]
    return m2_atoms


def read_from_mdrun(path_to_file, param_to_find):
    with open(path_to_file, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith(param_to_find):
            my_line = line
            if ";" in line:
                my_line = line.split(";")[0]
            return my_line.split("=")[1].strip()

class FlowSeqDumper(yaml.SafeDumper):
    pass


def represent_flow_sequence(dumper, seq):
    """
    This is a quick helper function that forces the yaml to write lists in square brackets on the same line, not as a
    super complicated nested list.
    """
    if isinstance(seq, (list, tuple, np.ndarray)):
        for el in iter_elements_nested(seq):
            if isinstance(el, numbers.Number):
                seq = nested_numpy_types_to_python_types(seq)
                break

    return dumper.represent_sequence(
        'tag:yaml.org,2002:seq',
        seq,
        flow_style=True
    )


def from_xvg_to_csv_energy(xvg_file, csv_file, energy_types: list):
    my_energy = read_object(xvg_file)
    total_energy = np.zeros(len(my_energy))
    for energy_type in energy_types:
        total_energy += my_energy[energy_type].to_numpy()

    df = pd.DataFrame(total_energy.T,
        columns=["Energy [kJ/mol]"])
    df.index.name = "Total index"
    write_object(df,csv_file)
