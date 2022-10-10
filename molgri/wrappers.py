from functools import wraps
from time import time
from datetime import timedelta


def time_method(my_method):
    """
    This wrapper times the execution of a method and prints the duration of the execution.

    Args:
        my_method: method with first argument self, the class it belongs to must implement a property
                   self.decorator label with value like 'grid ico_15' or 'pseudotrajectory H2O_HCl_ico_15_full'

    Returns:
        what my_method returns
    """
    @wraps(my_method)
    def decorated(*args, **kwargs):
        self_arg = args[0]

        t1 = time()
        func_value = my_method(*args, **kwargs)
        t2 = time()
        print(f"Timing the generation of {self_arg.decorator_label}: ", end="")
        print(f"{timedelta(seconds=t2-t1)} hours:minutes:seconds")
        return func_value
    return decorated
