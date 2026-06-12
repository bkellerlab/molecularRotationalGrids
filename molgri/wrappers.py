"""
All project-applicable wrappers.

Specifically, functionality for:
 - timing the function and writing out the result
 - using saved data if it exists
 - marking deprecated functions
"""

from functools import wraps
from time import time
from datetime import timedelta
from typing import Optional

from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


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
        #print(f"with arguments ", end="")
        #_inspect_method_print(my_method, *args, **kwargs)
        print(f": {timedelta(seconds=t2-t1)} hours:minutes:seconds")
        return func_value
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
