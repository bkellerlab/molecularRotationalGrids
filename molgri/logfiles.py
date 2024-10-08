"""
Logging. Naming of files with first free index.
"""
import inspect
import os
from abc import ABC, abstractmethod
import logging

import numpy as np

from molgri.paths import PATH_OUTPUT_LOGGING


class GeneralLogger(ABC):

    def __init__(self, investigated_object: object = None, file_path = None):
        class_name = investigated_object.__class__.__name__
        if file_path is None:
            logging.basicConfig(filename=f"{PATH_OUTPUT_LOGGING}{class_name}", level="INFO")
        else:
            logging.basicConfig(filename=file_path)
        self.logger = logging.getLogger(class_name)
        self.logger.info(f"SET UP OF: {investigated_object.__class__.__name__}")

    def log_set_up(self, *args, **kwargs):
        pass

    def log_ran_method(self, my_method, time, *args, **kwargs):
        """
        Every time a method has been run you want to record the arguments that were used.

        Args:
            my_method ():
            *args ():
            *kwargs ():

        Returns:

        """
        names = list(inspect.getfullargspec(my_method).args[1:])
        names.extend(kwargs.keys())
        values = list(args[1:])
        defaults = inspect.getfullargspec(my_method).defaults
        if np.any(defaults):
            values.extend(defaults)
        values.extend(kwargs.values())
        my_text = ""
        for n, v in zip(names, values):
            my_text += f"{n}={v}, "
        self.logger.info(f"Ran the method {my_method.__name__} with arguments: {my_text}")
        self.logger.info(f"Hash of this method run is {hash(tuple(values))}.")
        self.logger.info(f"Runtime of the method {my_method.__name__} with arguments: {my_text}")


class PtLogger(GeneralLogger):

    def log_set_up(self, investigated_object, ):
        super(PtLogger, self).log_set_up(investigated_object)


def first_index_free_4_all(list_names, list_endings, list_paths, index_places: int = 4) -> int:
    """Similar to find_first_free_index, but you have a list of files (eg. trajectory, topology, log) -> all must
    be still free to use this specific index."""
    assert len(list_paths) == len(list_names) == len(list_endings)
    i = 0
    # the loop only makes sense till you use up all numbers that could be part of the name
    while i < 10**index_places:
        for name, ending, path in zip(list_names, list_endings, list_paths):
            # if any one of them exist, break
            if os.path.exists(format_name(file_path=path, file_name=name, num=i, places_num=index_places,
                                          suffix=ending)):
                i += 1
                break
        # if you did not break the loop, all of the indices are free
        else:
            return i
    raise FileNotFoundError(f"All file names with unique numbers {format(0, f'0{index_places}d')}-{10**index_places-1} "
                            f"are already used up!")


def paths_free_4_all(list_names, list_endings, list_paths, index_places: int = 4) -> tuple:
    num = first_index_free_4_all(list_names, list_endings, list_paths, index_places)
    result_paths = []
    for name, ending, path in zip(list_names, list_endings, list_paths):
        result = format_name(file_name=name, file_path=path, num=num, places_num=index_places, suffix=ending)
        result_paths.append(result)
    return tuple(result_paths)


def find_first_free_index(name: str, ending: str = None, index_places: int = 4, path: str = "") -> int:
    """
    Problem: you want to save a file with a certain name, eg. pics/PrettyFigure.png, but not overwrite any existing
    files.

    Solution: this function checks if the given file already exists and increases the index until a unique one is
    found. So for our example it may return the string pics/PrettyFigure_007.png

    If no such file exists yet, the return string will include as many zeros as defined by index_places

    Args:
        path: path to appropriate directory (if not current), ending with /, in the example 'pics/'
        name: name of the file, in the example 'PrettyFigure'
        ending: in the example, select 'png' (NO DOT)
        index_places: how many places are used for the number

    Returns:
        number - first free index (can be forwarded to format_name)
    """
    i = 0
    while os.path.exists(format_name(file_path=path, file_name=name, num=i, places_num=index_places, suffix=ending)):
        i += 1
    return i


def format_name(file_name: str, num: int = None, places_num: int = 4, suffix: str = None, file_path: str = ""):
    """

    Args:
        file_path: eg. output/
        file_name: eg my_file
        num: eg 17
        places_num: eg 4 -> num will be formatted as 0017
        suffix: ending of the file

    Returns:
        full path and file name in correct format
    """
    till_num = os.path.join(file_path, file_name)
    if num is None:
        till_ending = till_num
    else:
        till_ending = f"{till_num}_{format(num, f'0{places_num}d')}"
    if suffix is None:
        return till_ending
    return f"{till_ending}.{suffix}"