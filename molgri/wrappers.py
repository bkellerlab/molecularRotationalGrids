"""
All project-applicable wrappers.

Specifically, functionality for:
 - timing the function and writing out the result
 - using saved data if it exists
 - marking deprecated functions
"""

import os
import warnings
from functools import wraps
from time import time
from datetime import timedelta
import numpy as np
import pandas as pd
import pickle
import inspect
from typing import Optional

from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from molgri.paths import PATH_OUTPUT_AUTOSAVE
from mpl_toolkits.mplot3d import Axes3D

from molgri.logfiles import GeneralLogger


def _inspect_method_print(my_method, *args, **kwargs):
    """
    Inspect the arguments of a method and nicely print them out

    Args:
        my_method (): a method of some class

    Returns:

    """
    names = list(inspect.getfullargspec(my_method).args[1:])
    names.extend(kwargs.keys())
    values = list(args[1:])
    values.extend(inspect.getfullargspec(my_method).defaults)
    values.extend(kwargs.values())
    my_text = ""
    for n, v in zip(names, values):
        my_text += f"{n}={v}, "
    print(my_text[:-2])


def time_method(my_method):
    """
    This wrapper times the execution of a method and prints the duration of the execution.

    Args:
        my_method: method with first argument self, the class it belongs to must implement a method
                   self.get_decorator_name() label with value like 'grid ico_15' or
                   'pseudotrajectory H2O_HCl_ico_15_full'

    Returns:
        what my_method returns
    """
    @wraps(my_method)
    def decorated(*args, **kwargs):
        self_arg = args[0]

        t1 = time()
        func_value = my_method(*args, **kwargs)
        t2 = time()
        print(f"Timing the execution of {my_method.__name__} of {self_arg.get_decorator_name()} ", end="")
        print(f"with arguments ", end="")
        _inspect_method_print(my_method, *args, **kwargs)
        print(f": {timedelta(seconds=t2-t1)} hours:minutes:seconds")
        return func_value
    return decorated


def save_or_use_saved(my_method):
    """
    This can be used on any method that provides data for plotting (important if data production takes a long time).
    Able to save any python objects: if numpy array will save as .npy, if pandas DataFrame as .csv, everything else
    as pickle.

    If you are saving, you automatically also log if the class has a self.logger attribute.

    Requirements:
        the class must have an attribute self.use_saved (bool)
        the class must have a method get_name() which provides a name that is suitable for saving (no whitespace etc)
        the method should not have any parameters that are commonly changed or their values should be part of the name
        (otherwise you risk reading and using data with old parameters)

    Args:
        my_method: around which method the wrapper is applied

    Returns:
        whatever the original method would return, either freshly created or read from a file
    """
    @wraps(my_method)
    def decorated(self, *args, **kwargs):
        method_name = my_method.__name__
        name_without_ext = f"{PATH_OUTPUT_AUTOSAVE}{method_name}_{self.get_name()}"


        # try to find a suitable saved file
        if self.use_saved:
            if os.path.isfile(f"{name_without_ext}.npy"):
                return np.load(f"{name_without_ext}.npy")
            elif os.path.isfile(f"{name_without_ext}.csv"):
                return pd.read_csv(f"{name_without_ext}.csv", index_col=0)
            elif os.path.isfile(name_without_ext):
                with open(name_without_ext, 'rb') as f:
                    loaded_data = pickle.load(f)
                return loaded_data
            # else will simply continue
        # don't use else - the rest should be run if 1) not self.use_saved OR 2) file doesn't exist
        t1 = time()
        method_output = my_method(self, *args, **kwargs)
        t2 = time()
        # logging
        if "logger" in self.__dict__.keys():
            self.logger.log_ran_method(my_method, t2-t1, *args, **kwargs)
        if isinstance(method_output, pd.DataFrame):
            method_output.to_csv(f"{name_without_ext}.csv", index=True)
        elif isinstance(method_output, np.ndarray):
            np.save(f"{name_without_ext}.npy", method_output)
        else:
            with open(name_without_ext, 'wb') as f:
                pickle.dump(method_output, f)
        return method_output
    return decorated


def deprecated(my_method):
    """
    Mark functions or methods that may be removed in the future. Raises a warning.
    """
    warnings.warn(f"The method {my_method.__name__} is deprecated and may be removed in the future.",
                  DeprecationWarning, stacklevel=2)

    @wraps(my_method)
    def decorated(*args, **kwargs):
        return my_method(*args, **kwargs)

    return decorated


def plot_method(my_method):
    """
    Mark functions or methods that make a complete plot with 2D axes. The wrapper:
    1) makes sure the name of the function begins with make_
    2) processes input parameters: fig=None, ax=None, save=True
    3) adds saving if requested
    """
    @wraps(my_method)
    def decorated(*args, **kwargs) -> None:
        split_name = my_method.__name__.split("_")
        assert split_name[0] == "plot", "Name of the method not starting with plot_, maybe not a plotting method?"
        self = args[0]
        fig: Figure = kwargs.pop("fig", None)
        ax: Axes = kwargs.pop("ax", None)
        save: bool = kwargs.pop("save", True)
        # for 2D plots, ignore animate_rot and projection arguments if they occur
        kwargs.pop("animate_rot", None)
        kwargs.pop("projection", None)

        self._create_fig_ax(fig=fig, ax=ax)
        my_method(*args, **kwargs)
        if save:
            self._save_plot_type(my_method.__name__)
    return decorated


def plot3D_method(my_method):
    """
    Mark functions or methods that make a complete plot with 3D axes. The wrapper:
    1) makes sure the name of the function begins with make_
    2) processes input parameters: fig=None, ax=None, save=True, animate_rot=False
    3) adds saving and animation generation if requested
    """
    @wraps(my_method)
    def decorated(*args, **kwargs) -> Optional[FuncAnimation]:
        split_name = my_method.__name__.split("_")
        assert split_name[0] == "plot", "Name of the method not starting with plot_, maybe not a plotting method?"
        self = args[0]
        fig: Figure = kwargs.pop("fig", None)
        ax: Axes3D = kwargs.pop("ax", None)
        save: bool = kwargs.pop("save", True)
        animate_rot: bool = kwargs.pop("animate_rot", False)
        projection: str = kwargs.pop("projection", "3d")

        self._create_fig_ax(fig=fig, ax=ax, projection=projection)
        my_method(*args, **kwargs)

        if save:
            self._save_plot_type(my_method.__name__)
        if animate_rot:
            return self._animate_figure_view(self.fig, self.ax)
    return decorated


def make_all_method(my_method):
    """
    This is a method for making all subplots in a MultiRepresentationPlot object.
    """
    @wraps(my_method)
    def decorated(*args, **kwargs) -> Optional[FuncAnimation]:
        split_name = my_method.__name__.split("_")
        assert split_name[0] == "make" and split_name[1] == "all", "Name of the method not starting with make_all, " \
                                                                 "maybe not a plotting method?"
        self = args[0]
        fig: Figure = kwargs.pop("fig", None)

    return decorated
