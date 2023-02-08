import functools
import os
import datetime
import timeit
from typing import Callable
import logging
from abc import ABC, abstractmethod


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from molgri.paths import PATH_OUTPUT_TIMING, PATH_OUTPUT_LOGGING
from molgri.molecules.writers import PtIOManager
from molgri.space.utils import find_first_free_index, format_name





class PerformanceManager:

    def __init__(self, measured_function: Callable, varied_parameter: str, function_kwargs: dict = None,
                 ):
        """
        All arguments (except the varied one) should be provided as kwargs
        Args:
            measured_function:
            varied_parameter:
            *args:
            **kwargs:
        """
        if function_kwargs is None:
            function_kwargs = dict()
        self.measured_function = measured_function
        self.varied_parameter = varied_parameter
        self.parameter_range = None
        self.kwargs = function_kwargs
        self.df: pd.DataFrame = None

    def set_parameter_range(self, values):
        self.parameter_range = values

    def timeit(self, repeat=5):
        df_list = []
        for value in self.parameter_range:
            # set up the function anew
            self.kwargs[self.varied_parameter] = value
            times = timeit.repeat(functools.partial(self.measured_function, **self.kwargs),
                                  repeat=repeat, number=1)
            df_list.extend([[value, t] for t in times])
        self.df = pd.DataFrame(np.array(df_list), columns=[self.varied_parameter, "Time [s]"])
        self.df["Time [s]"] = pd.to_numeric(self.df["Time [s]"])

    def save_to_file(self, name_to_save: str):
        # determine index to use in a file
        free_index = find_first_free_index(path=PATH_OUTPUT_TIMING, name=name_to_save, ending="csv")
        path_csv = format_name(file_path=PATH_OUTPUT_TIMING, file_name=name_to_save, num=free_index, suffix="csv")
        if self.df is None:
            self.timeit()
        self.df.to_csv(path_csv)



def plot_performance(df_data: pd.DataFrame, ax: plt.Axes, x: str, y: str = "Time [s]"):
    sns.lineplot(data=df_data, ax=ax, x=x, y=y)


def time_pt(m1: str, m2: str, b_name: str = "50", o_name: str = "50", t_name: str = "(1, 2, 3)"):
    # standard example of H2O-H2O system
    manager = PtIOManager(name_central_molecule=m1, name_rotating_molecule=m2,
                          b_grid_name=b_name, o_grid_name=o_name, t_grid_name=t_name)
    manager.construct_pt(as_dir=False)
    # logging
    path_logger = format_name(file_path=PATH_OUTPUT_LOGGING, file_name=name_to_save, num=free_index)
    PtLogger(path_logger).log_set_up()


def test_funct(one_arg, second_arg):
    return one_arg**2 + np.sqrt(second_arg)


if __name__ == "__main__":
    #timeit.timeit('time_with_timeit()', setup="from __main__ import time_with_timeit", number=10)
    pm = PerformanceManager(time_pt, "b_name", function_kwargs={"m1": "H2O", "m2": "H2O", "o_name": "cube3D_10"})
    number_options = np.linspace(5, 50, num=5, dtype=int)
    number_options = np.unique(number_options)
    number_options = [f"ico_{num}" for num in number_options]
    pm.set_parameter_range(number_options)
    pm.timeit()
    pm.save_to_file("test_b_name")
    fig, ax = plt.subplots(1, 1)
    plot_performance(pm.df, ax, x="b_name")
    plt.show()

    # pm = PerformanceManager(test_funct, "second_arg", function_kwargs={"one_arg": 3})
    # number_options = np.linspace(100, 10000, num=10, dtype=int)
    # number_options = np.unique(number_options)
    # #number_options = [f"ico_{num}" for num in number_options]
    # pm.set_parameter_range(number_options)
    # pm.timeit()
    # fig, ax = plt.subplots(1, 1)
    # plot_performance(pm.df, ax, x="second_arg")
    # plt.show()