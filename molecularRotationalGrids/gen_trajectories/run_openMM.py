from MDAnalysis.auxiliary import auxreader
from openmm.app import *
from openmm import *
# noinspection PyUnresolvedReferences
from openmm.unit import nanometer, kelvin, picosecond, picoseconds, kilojoule, mole

import numpy as np
from parsers.name_parser import NameParser

from analysis.analyse_gromacs_run import TrajectoryData, get_cnum_rnum
import MDAnalysis as mda
from examples.wrappers import time_function
from my_constants import *
from tqdm import tqdm
import pandas as pd


@time_function
def evaluate_energy_pseudotrajectory(name: str, disable_tqdm: bool = False) -> np.ndarray:
    """
    Use openMM and a pseudotrajectory written to a .gro file to evaluate the energy at each frame of the
    pseudotrajectory (a faster version of gromacs rerun).

    Args:
        name: like 'randomE_500_circular'
        disable_tqdm: forward to tqdm in order to disable the progress bar

    Returns:
        an array of energies (in kJ/mol) for each step in gro file associated with name
    """
    gro = GromacsGroFile(f'{PATH_GENERATED_GRO_FILES}{name}.gro')
    frame_num = gro.getNumFrames()
    # /home/janjoswig/local/gromacs-2022/share/gromacs/top
    nap = NameParser(name)
    top = GromacsTopFile(f'{PATH_TOPOL}{nap.central_molecule}_{nap.rotating_molecule}.top',
                         periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir=f"{PATH_FF}")
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.004 * picoseconds)
    simulation = Simulation(top.topology, system, integrator)
    potential_energies = np.zeros(frame_num)
    kj_mol = kilojoule / mole
    for frame in tqdm(range(frame_num), disable=disable_tqdm):
        simulation.context.setPositions(gro.getPositions(frame=frame))
        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        potential_energies[frame] = energy.value_in_unit(kj_mol)
    return potential_energies


# def evaluate_all_energy_pseudotrajectory(set_type="full", set_size="normal", disable_tqdm=False):
#     """
#     Run evaluate_energy_pseudotrajectory for each grid of certain type and size.
#     This function is mostly used to print how long all openMM calculations take. For other purposes, use
#     all_openMM_to_pickle that actually saves the results.
#
#     Args:
#         set_type: 'circular' or 'full'
#         set_size: 'normal' or 'small'
#         disable_tqdm: forward to tqdm in order to disable the progress bar
#     """
#     options_names = [f"{name}_{num}_{set_type}" for name, num in zip(SIX_METHOD_NAMES, SIZE2NUMBERS[set_size])]
#     for name in options_names:
#         evaluate_energy_pseudotrajectory(name, disable_tqdm=disable_tqdm)


def open_MM_to_pickle(name):
    """
    Evaluate the energy of a pseudotrajectory with openMM, then save all information about the trajectory to a pickle
    file in PATH_OMM_RESULTS.

    Args:
        name: like 'randomE_500_circular'
    """
    path = f"{PATH_GROMACS}{name}/"
    c_num, r_num = get_cnum_rnum(name)
    u = mda.Universe(path + f"{name}.gro", path + "test.trr")
    read_data = TrajectoryData(3, 3, u, aux=None, verbose=True).run()
    df = read_data.df
    df["Potential [kJ/mol]"] = evaluate_energy_pseudotrajectory(name)
    file_path_results = PATH_OMM_RESULTS + name + "_openMM.pkl"
    df.to_pickle(file_path_results)


def all_openMM_to_pickle(molecule1, molecule2, set_type="full", set_size=500):
    """
    Run open_MM_to_pickle for each grid of certain type and size.

    Args:
        set_type: 'circular' or 'full'
        set_size: 'normal' or 'small'
    """
    for method_name in SIX_METHOD_NAMES:
        nap = NameParser({"central_molecule": molecule1, "rotating_molecule": molecule2, "traj_type": set_type,
                          "num_grid_points": set_size, "grid_type": method_name})
        open_MM_to_pickle(nap.get_standard_name())


def max_error_openMM_to_gromacs(name: str) -> tuple:
    """
    Load the pickle files created with openMM and with gromacs for a simulation with given name. Compare the
    energies of both simulations and return the largest difference in energy over the course of the trajectory.

    Args:
        name: like 'randomE_500_circular'

    Returns:
        max difference of energies over all steps of the simulation
    """
    open_MM_path = PATH_OMM_RESULTS + name + "_openMM.pkl"
    openMM_result = pd.read_pickle(open_MM_path)
    gromacs_path = PATH_GRO_RESULTS + name + ".pkl"
    gromacs_result = pd.read_pickle(gromacs_path)
    average_openmm = np.average(openMM_result["Potential [kJ/mol]"])
    average_gromacs = np.average(gromacs_result["Potential [kJ/mol]"])
    difference = openMM_result["Potential [kJ/mol]"] - gromacs_result["Potential [kJ/mol]"]
    rel_difference = difference/gromacs_result["Potential [kJ/mol]"]
    return average_openmm, average_gromacs, np.max(np.abs(difference)), np.max(np.abs(rel_difference))*100


def all_max_error_openMM_to_gromacs(molecule1, molecule2, set_type: str = "full", set_size: int = 500):
    """
    Run max_error_openMM_to_gromacs for each grid of certain type and size. Order the max differeces between gromacs
    and openMM in a dataframe (for each method) and return it.

    Args:
            set_type: 'circular' or 'full'
            set_size: 'normal' or 'small'

    Returns:
        a dataframe where rows are different methods and the only column is the max distance in energy between
        the gromacs and the openMM calculation
    """
    print(f"############ COMPARE openMM and GROMACS: {molecule1} {molecule2} {set_type} {set_size} ############")
    all_max_errors = np.zeros((len(SIX_METHOD_NAMES), 4))
    for i, method_name in enumerate(SIX_METHOD_NAMES):
        nap = NameParser({"central_molecule": molecule1, "rotating_molecule": molecule2, "traj_type": set_type,
                          "num_grid_points": set_size, "grid_type": method_name})
        all_max_errors[i] = max_error_openMM_to_gromacs(nap.get_standard_name())
    columns = ["Avg openMM [kJ/mol]", "Avg GROMACS [kJ/mol]", "Max abs error [kJ/mol]", "Max rel error [%]"]
    df = pd.DataFrame(all_max_errors, index=PRETTY_METHOD_NAMES, columns=columns)
    print(df.sort_values("Max abs error [kJ/mol]"))


if __name__ == "__main__":
    name = "H2O_H2O_randomQ_500_circular"
    #open_MM_to_pickle(name)
    all_openMM_to_pickle("H2O", "H2O", set_type="circular", set_size=500)

    all_max_error_openMM_to_gromacs("H2O", "H2O", set_type="circular", set_size=50)
