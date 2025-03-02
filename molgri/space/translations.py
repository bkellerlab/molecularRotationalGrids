"""
Parse linear discretisation in units provided by users.

Able to process strings like '[1, 2, 3]', 'linspace(1, 5, 50)' or 'range(0.5, 3, 0.4)' and translate them to appropriate
arrays of distances.
"""

import hashlib
import numbers
from ast import literal_eval

import numpy as np
from numpy.typing import NDArray

from molgri.constants import NM2ANGSTROM


class TranslationParser(object):

    """
    User input is expected in nanometers (nm)!

        Parse all ways in which the user may provide a linear translation grid. Currently supported formats:
            - a list of numbers, eg '[1, 2, 3]'
            - a linearly spaced list with optionally provided number of elements eg. 'linspace(1, 5, 50)'
            - a range with optionally provided step, eg 'range(0.5, 3, 0.4)'
    """

    def __init__(self, user_input: str):
        """
        Args:
            user_input: a string in one of allowed formats
        """
        self.user_input = user_input
        if "linspace" in self.user_input:
            bracket_input = self._read_within_brackets()
            self.trans_grid = np.linspace(*bracket_input, dtype=float)
        elif "range" in self.user_input:
            bracket_input = self._read_within_brackets()
            self.trans_grid = np.arange(*bracket_input, dtype=float)
        else:
            self.trans_grid = literal_eval(self.user_input)
            self.trans_grid = np.array(self.trans_grid, dtype=float)
            self.trans_grid = np.sort(self.trans_grid, axis=None)
        # all values must be non-negative
        assert np.all(self.trans_grid >= 0), "Distance from origin cannot be negative."
        # convert to angstrom
        self.trans_grid = self.trans_grid * NM2ANGSTROM
        # we use a (shortened) hash value to uniquely identify the grid used, no matter how it's generated
        self.grid_hash = int(hashlib.md5(self.trans_grid).hexdigest()[:8], 16)

    def get_name(self):
        return f"{self.grid_hash}"

    def get_trans_grid(self) -> NDArray:
        """Getter to access all distances from origin in angstorms."""
        return self.trans_grid

    def get_N_trans(self) -> int:
        """Get the number of translations in this grid."""
        return len(self.trans_grid)

    def sum_increments_from_first_radius(self) -> float:
        """
        Get final distance - first non-zero distance == sum(increments except the first one).

        Useful because often the first radius is large and then only small increments are made.
        """
        return float(np.sum(self.get_increments()[1:]))

    def get_increments(self) -> NDArray:
        """
        Get an array where each element represents an increment needed to get to the next radius.

        Example:
            self.trans_grid = np.array([10, 10.5, 11.2])
            self.get_increments() -> np.array([10, 0.5, 0.7])
        """
        return get_increments(self.get_trans_grid())

    def _read_within_brackets(self) -> tuple:
        """
        Helper function to aid reading linspace(start, stop, num) and arange(start, stop, step) formats.
        """
        str_in_brackets = self.user_input.split('(', 1)[1].split(')')[0]
        str_in_brackets = literal_eval(str_in_brackets)
        if isinstance(str_in_brackets, numbers.Number):
            str_in_brackets = tuple((str_in_brackets,))
        return str_in_brackets


def get_increments(my_array: NDArray) -> NDArray:
    """
    Get an array where each element represents an increment needed to get to the next radius.

    Example:
        my_grid = np.array([10, 10.5, 11.2])
        get_increments(my_grid) -> np.array([10, 0.5, 0.7])
    """
    increment_grid = [my_array[0]]
    for start, stop in zip(my_array, my_array[1:]):
        increment_grid.append(stop-start)
    increment_grid = np.array(increment_grid)
    assert np.all(increment_grid > 0), "Negative or zero increments in translation grid make no sense!"
    return increment_grid


def get_between_radii(my_array: NDArray, include_zero=False) -> NDArray:
    """
    Get the radii at which Voronoi cells of the position grid should be positioned. This should be right in-between
    two orientation point layers (except the first layer that is fully encapsulated by the first voronoi layer
    and the last one that is above the last one so that the last layer of points is right in-between the two last
    Voronoi cells

    Returns:
        an array of distances, same length as the self.get_radii array but with all distances larger than the
        corresponding point radii
    """
    # get increments to each radius, remove first one and add an extra one at the end with same distance as
    # second-to-last one
    increments = list(get_increments(my_array))
    if len(increments) > 1:
        increments.pop(0)
        increments.append(increments[-1])
        increments = np.array(increments)
        increments = increments / 2
    else:
        increments = np.array(increments)

    between_radii = my_array + increments

    if include_zero:
        between_radii = np.concatenate([[0,], between_radii])
    return between_radii
