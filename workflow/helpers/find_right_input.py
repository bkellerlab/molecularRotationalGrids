"""
We have a problem that we often want to perform similar operations (eg. finding minimal energy frames and plotting
them) on pseudotrajectory, trajectory or wrapped trajectory. Here are some helper functions to find the correct files.
"""

def find_the_right_structure(what: str) -> str:
    """
    We want to use the structure in the right directory. Additionally, if we are only plotting the center of mass for
    the second trajectory we want a structure with the correct number of atoms (m1_COM_m2 structure).

    Args:
        wc (): wildcards from Snakemake workflow

    Returns:
        a string giving a path to stucture file
    """
    return f"<pseudosimulation>{what}.gro"

def find_the_right_frames(where: str, what: str, indices: list, grid_len) -> tuple:
    """
    Since we create VMD plots from multiple .gro files, each one containing just one frame, we often need to select
    the right set of frames. Here we select from the correct directory given that we know the indices of desired frames.

    Args:
        wc (): wildcards from Snakemake rule
        indices (list): a list of one or more indices we want to select

    Returns:
        a tuple where all elements are paths to .gro files we want
    """
    result = []
    for frame_index in indices:
        if frame_index < grid_len:
            result.append(f"{where}trajectory_slices/{what}{frame_index}.gro")
        elif frame_index == grid_len:
            result.append(f"<pseudosimulation>bulk_structure.gro")
        else:
            raise ValueError(f"Cannot find index {frame_index} in PT of length {grid_len}!")
    return tuple(result)

def where_to_look(keyword: str) -> str:
    """
    We are often looking for molecular structures, trajectories ... in one of three folders:
    - <pseudosimulation> if we want the generated structures
    - <simulation> if we want the original simulation data
    - <outputs_assignment> if we are looking for wrapped trajectories
    This function helps select the right one based on the keyword.

    Args:
        keyword (str): a string containing the information what we want

    Returns:
        one of the three paths as described above
    """
    if "sqra" in keyword:
        return "<pseudosimulation>"


    if "wrapped" in keyword:
        return "<outputs_assignment>"
    # IMPORTANT - we must first search for pseudosimulation keyword because simulation keyword obviously contains it!
    elif "pseudosimulation" in keyword:
        return "<pseudosimulation>"
    elif "simulation" in keyword:
        return "<simulation>"
    elif int(keyword):
        return "<outputs_assignment>"
    else:
        raise ValueError(f"Keyword must contain 'wrapped', 'simulation' or 'pseudosimulation', cannot understand {keyword}.")

def what_to_provide(keyword: str, for_a_structure: bool = False) -> str:
    """
    Based on some keyword we can decide to provide either the full structure of molecule 2 or only its center of mass.

    Args:
        keyword (str): some string that might contain the word COM or not

    Returns:
        prefix of .gro files we want
    """
    if for_a_structure:
        if "COM" in keyword:
            return "structure_COM"
        else:
            return "structure"
    else:
        if "COM" in keyword:
            return "m1_COM_m2_frame_"
        else:
            return "frame_"