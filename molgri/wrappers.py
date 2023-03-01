import os
from functools import wraps
from time import time
from datetime import timedelta
import numpy as np
import pandas as pd
import pickle

from molgri.paths import OUTPUT_PLOTTING_DATA


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
        print(f"Timing the generation of {self_arg.get_decorator_name()}: ", end="")
        print(f"{timedelta(seconds=t2-t1)} hours:minutes:seconds")
        return func_value
    return decorated


def save_or_use_saved(my_method):
    """
    This can be used on any method that provides data for plotting (important if data production takes a long time).
    Able to save any python objects: if numpy array will save as .npy, if pandas DataFrame as .csv, everything else
    as pickle.

    Requirements:
        the class must have an attribute self.use_saved (bool)
        the class must have a method get_name() which provides a name that is suitable for saving (no whitespace etc)
        the method should not have any parameters that are commonly changed

    Args:
        my_method: around which method the wrapper is applied

    Returns:
        whatever the original method would return, either freshly created or read from a file
    """
    @wraps(my_method)
    def decorated(self, *args, **kwargs):
        method_name = my_method.__name__
        name_without_ext = f"{OUTPUT_PLOTTING_DATA}{method_name}_{self.get_name()}"
        # try to find a suitable saved file
        if self.use_saved:
            if os.path.isfile(f"{name_without_ext}.npy"):
                #print("using saved npy")
                return np.load(f"{name_without_ext}.npy")
            elif os.path.isfile(f"{name_without_ext}.csv"):
                #print("using saved csv")
                return pd.read_csv(f"{name_without_ext}.csv", index_col=0)
            elif os.path.isfile(name_without_ext):
                with open(name_without_ext, 'rb') as f:
                    #print("using saved pickle")
                    loaded_data = pickle.load(f)
                return loaded_data
            # else will simply continue
        # don't use else - the rest should be run if 1) not self.use_saved OR 2) file doesn't exist
        method_output = my_method(self, *args, **kwargs)
        if isinstance(method_output, pd.DataFrame):
            method_output.to_csv(f"{name_without_ext}.csv", index=True)
        elif isinstance(method_output, np.ndarray):
            #print("saving")
            np.save(f"{name_without_ext}.npy", method_output)
        else:
            #print("pickling", name_without_ext)
            with open(name_without_ext, 'wb') as f:
                pickle.dump(method_output, f)
        return method_output
    return decorated
