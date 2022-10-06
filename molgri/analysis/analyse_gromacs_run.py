"""
This file creates pickle summaries of simulations and reads from them to create a quick summary of methods
(number of grid points, lowest E and the corresponding H length).
"""
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.auxiliary.core import auxreader

from examples.wrappers import time_function
from my_constants import *
from MDAnalysis.analysis.distances import dist
import pandas as pd
from tqdm import tqdm

from parsers.name_parser import NameParser


class TrajectoryData(AnalysisBase):  # subclass AnalysisBase

    def __init__(self, c_num, r_num, u, aux=None, verbose=True):
        """
        Set up the initial analysis parameters.
        """
        # must first run AnalysisBase.__init__ and pass the trajectory
        self.u = u
        trajectory = u.trajectory
        super(TrajectoryData, self).__init__(trajectory, verbose=verbose)
        # set atomgroup as a property for access in other methods
        # we can calculate masses now because they do not depend
        # on the trajectory frame.
        self.box = u.dimensions
        self.molecule_c = u.select_atoms(f"bynum 1:{c_num}")
        self.molecule_r = u.select_atoms(f"bynum {c_num+1}:{c_num+r_num}")
        self.num_atoms_c = c_num
        self.num_atoms_r = r_num
        self.hbonds = HydrogenBondAnalysis(universe=self.u, donors_sel="name O OW", hydrogens_sel="name H HW1 HW2",
            acceptors_sel="name O OW",
            update_selections=True
        )
        self.aux = aux
        self.num_extra_columns = 8

    def _prepare(self):
        """
        Create array of zeroes as a placeholder for results.
        This is run before we begin looping over the trajectory.
        """
        self.results = np.zeros((self.n_frames, self.num_extra_columns + 3*self.num_atoms_r+3*self.num_atoms_c))

    def _single_frame(self):
        """
        This function is called for every frame that we choose
        in run().
        """
        # # the current timestep of the trajectory is self._ts
        self.results[self._frame_index, 0] = self._ts.frame
        # # the actual trajectory is at self._trajectory
        self.results[self._frame_index, 1] = self._trajectory.time
        # H-bonds in 2 and 3
        self.results[self._frame_index, 2] = None
        self.results[self._frame_index, 3] = None
        # energies in 4-9 ["LJ Energy", "Dis.corr.", "Coulomb", "Coulomb recip.", "Potential"]
        self.results[self._frame_index, 4:self.num_extra_columns] = np.NaN
        # in case only potential energy is written
        if self.aux is not None and len(self.aux[self._frame_index].data) == 2:
            self.results[self._frame_index, 9] = self.aux[self._frame_index].data[1]
        # in case contributions to energy are also known
        elif self.aux is not None and len(self.aux[self._frame_index].data) == 6:
            for i in range(1, 6):
                self.results[self._frame_index, 3+i] = self.aux[self._frame_index].data[i]
        elif self.aux is not None and len(self.aux[self._frame_index].data) == 5:
            for i in range(1, 5):
                self.results[self._frame_index, 3 + i] = self.aux[self._frame_index].data[i]
        num_coordinates = 3*self.num_atoms_r+3*self.num_atoms_c
        pos_in_nm = self.u.atoms.positions.reshape((num_coordinates,)) * ANGSTROM2NM
        self.results[self._frame_index, self.num_extra_columns:] = pos_in_nm

    def _conclude(self):
        """
        Finish up by calculating an average and transforming our
        results into a DataFrame.
        """
        # by now self.result is fully populated
        run_hbonds = self.hbonds.run().results.hbonds
        all_hbonds = np.empty((self.n_frames, 2))
        all_hbonds[:] = np.NaN
        all_hbonds[:, 0].put(indices=run_hbonds[:, 0].astype(int), values=run_hbonds[:, 4] * ANGSTROM2NM)
        all_hbonds[:, 1].put(indices=run_hbonds[:, 0].astype(int), values=run_hbonds[:, 5])
        # TODO: select shortest Hbond
        self.results[:, 2:4] = all_hbonds
        self.df = pd.DataFrame(self.results, columns=self._get_column_names())

    def _get_column_names(self) -> list:
        first_variables = ["Frame", "Time [ps]", "H length [nm]", "H angle [deg]"]
        energies = ["LJ Energy", "Dis.corr.", "Coulomb", "Potential"]
        energies = [an + " [kJ/mol]" for an in energies]
        atom_names = []
        for i, atom in enumerate(self.molecule_c.atoms):
            for coor in ("x", "y", "z"):
                atom_names.append(f"m_1_{atom.type}_{i}_{coor}")
        for i, atom in enumerate(self.molecule_r.atoms):
            for coor in ("x", "y", "z"):
                atom_names.append(f"m_2_{atom.type}_{i}_{coor}")
        atom_names = [an + " [nm]" for an in atom_names]
        first_variables.extend(energies)
        first_variables.extend(atom_names)
        return first_variables


def get_cnum_rnum(name):
    if "run" in name:
        path = f"{PATH_GENERATED_GRO_FILES}{'_'.join(name.split('_')[:-1])}.gro"
    else:
        path = f"{PATH_GENERATED_GRO_FILES}{name}.gro"
    with open(path) as f:
        first_line = f.readline()
        first_line = first_line.split(",")
        c_num = int(first_line[0].split("=")[1])
        r_num = int(first_line[1].split("=")[1])
    return c_num, r_num


def generate_gro_results(name: str):
    """
    Create a pickle file in location PATH_GRO_RESULTS with given name where the results of a GROMACS simulation
    are saved.

    Includes data:
        frame, timestep, H bond length, energy, positions of all atoms

    Args:
        name: ID of the simulation
    """
    path = f"{PATH_GROMACS}{name}/"
    c_num, r_num = get_cnum_rnum(name)
    if "run" in name:
        gro_path = f"{PATH_GROMACS}{name}/{'_'.join(name.split('_')[:-1])}.gro"
    else:
        gro_path = f"{PATH_GENERATED_GRO_FILES}{name}.gro"
    u = mda.Universe(gro_path, path + "test.trr")
    aux = auxreader(path + "full_energy.xvg")
    read_data = TrajectoryData(c_num, r_num, u, aux=aux, verbose=True).run()
    df = read_data.df
    file_path_results = PATH_GRO_RESULTS + name + ".pkl"
    value = "Potential [kJ/mol]"
    print(len(df))
    df.sort_values(value, inplace=True)
    print("######### HEAD ##########")
    for i in range(100):
        print(int(df["Frame"].iloc[i]), end=", ")
    print()
    print(df[value].iloc[0], df[value].iloc[10], df[value].iloc[30])
    print(df[["Frame", value]].head(10))
    print("######### TAIL ##########")
    print(df[["Frame", value]].tail(5))
    df.to_pickle(file_path_results)


@time_function
def generate_all_gro_results(set_type: str = "circular", set_size="normal"):
    options_names = [f"{name}_{num}_{set_type}" for name, num in zip(SIX_METHOD_NAMES, SIZE2NUMBERS[set_size])]
    options_names.append(FULL_RUN_NAME)
    for name in tqdm(options_names):
        generate_gro_results(name)


def summary_gro_results(mol1, mol2, set_type: str = "circular", set_size=500, use_omm=True) -> pd.DataFrame:
    """
    Returns a dataframe including number of grid points, min energy and H bond length at that energy for all
    tested methods.

    Args:
        set_type: 'circular' or 'full'
        set_size: 'normal' or 'small'

    Returns:
        dataframe with number of grid points, min energy and H bond length
    """
    print(f"############ RESULTS: {mol1} {mol2} {set_type} {set_size} #################")
    options_names = [NameParser(f"{mol1}_{mol2}_{name}_{set_type}_{set_size}").get_standard_name()
                     for name in SIX_METHOD_NAMES]
    options_names.append(f"{mol1}_{mol2}_{FULL_RUN_NAME}")
    data = np.zeros((len(options_names), 5))
    for i, name in tqdm(enumerate(options_names)):
        if use_omm:
            file_path_results = PATH_OMM_RESULTS + name + "_openMM.pkl"
        else:
            file_path_results = PATH_GRO_RESULTS + name + ".pkl"
        df = pd.read_pickle(file_path_results)
        df.sort_values("Potential [kJ/mol]", inplace=True)
        data[i][0] = len(df)
        # ["LJ Energy [kJ/mol]", "Dis.corr. [kJ/mol]", "Coulomb [kJ/mol]", "Coulomb recip. [kJ/mol]",
        #                 "Potential [kJ/mol]"]
        data[i][1] = df.iloc[0]["Potential [kJ/mol]"]
        data[i][2] = df.iloc[0]["LJ Energy [kJ/mol]"]
        data[i][3] = df.iloc[0]["Coulomb [kJ/mol]"]
        data[i][4] = df.iloc[0]["H length [nm]"]
    names = [x for x in PRETTY_METHOD_NAMES]
    names.append(FULL_RUN_NAME)
    df = pd.DataFrame(data, index=names, columns=["N points", "E min [kJ/mol]", "LJ", "Coulomb", "H length [nm]"])
    return df.sort_values("E min [kJ/mol]")


if __name__ == "__main__":
    generate_gro_results("protein0_CL_ico_1000_full")
    #generate_gro_results("H2O_H2O_run_150")
    # molecule1 = "H2O"
    # molecule2 = "H2O"
    # N=500
    # for name in SIX_METHOD_NAMES:
    #     for traj_type in ("circular",):
    #         nap = NameParser(f"{molecule1}_{molecule2}_{traj_type}_{name}_{N}")
    #         generate_gro_results(nap.get_standard_name())
    # name="H2O_H2O_cube3D_50_circular"
    # path = f"{PATH_GROMACS}{name}/"
    # u = mda.Universe(path + f"{name}.gro", path + "test.trr")
    # aux = auxreader(path + "full_energy.xvg")
    # td = TrajectoryData(3, 3, u, aux).run()
    # print(td.df[:5])
    # generate_all_gro_results("circular", "normal")
    # generate_all_gro_results("full", "normal")
    #print(summary_gro_results("circular", 500))
    #print(summary_gro_results("full", 500))
