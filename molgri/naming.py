"""
Naming conventions are defined here.
"""

from molgri.constants import ALL_GRID_ALGORITHMS, DEFAULT_ALGORITHM_O, DEFAULT_ALGORITHM_B


class NameParser:

    def __init__(self, name_string: str):
        self.name_string = name_string
        self.N = self._find_a_number()
        self.algo = self._find_algorithm()
        self.dim = self._find_dimensions()

    def _find_a_number(self) -> int or None:
        """
        Try to find an integer representing number of grid points anywhere in the name.

        Returns:
            the number of points as an integer, if it can be found, else None

        Raises:
            ValueError if more than one integer present in the string (e.g. 'ico_12_17')
        """
        split_string = self.name_string.split("_")
        candidates = []
        for fragment in split_string:
            if fragment.isnumeric():
                candidates.append(int(fragment))
        # >= 2 numbers found in the string
        if len(candidates) > 1:
            raise ValueError(f"Found two or more numbers in grid name {self.name_string},"
                             f" can't determine num of points.")
        # exactly one number in the string -> return it
        elif len(candidates) == 1:
            return candidates[0]
        # no number in the string -> return None
        else:
            return None

    def _find_algorithm(self) -> str or None:
        split_string = self.name_string.split("_")
        candidates = []
        for fragment in split_string:
            if fragment in ALL_GRID_ALGORITHMS:
                candidates.append(fragment)
            elif fragment.lower() == "none":
                candidates.append(DEFAULT_ALGORITHM_O)
        # >= 2 algorithms found in the string
        if len(candidates) > 1:
            raise ValueError(f"Found two or more algorithm names in grid name {self.name_string}, can't decide.")
        # exactly one algorithm in the string -> return it
        elif len(candidates) == 1:
            return candidates[0]
        # no algorithm given -> None
        else:
            return None

    def _find_dimensions(self) -> str or None:
        split_string = self.name_string.split("_")
        candidates = []
        for fragment in split_string:
            if len(fragment) == 2 and fragment[-1] == "d" and fragment[0].isnumeric():
                candidates.append(int(fragment))
        if len(candidates) > 1:
            raise ValueError(f"Found two or more dimension names in grid name {self.name_string}, can't decide.")
        # exactly one algorithm in the string -> return it
        elif len(candidates) == 1:
            return candidates[0]
        # no algorithm given -> None
        else:
            return None


class FullGridNameParser:

    def __init__(self, name_string: str):
        split_name = name_string.split("_")
        self.b_grid_name = None
        self.o_grid_name = None
        self.t_grid_name = None
        for i, split_part in enumerate(split_name):
            if split_part == "b":
                try:
                    self.b_grid_name = GridNameParser(
                        f"{split_name[i + 1]}_{split_name[i + 2]}", "b").get_standard_grid_name()
                except IndexError:
                    self.b_grid_name = GridNameParser(f"{split_name[i + 1]}", "o").get_standard_grid_name()
            elif split_part == "o":
                try:
                    self.o_grid_name = GridNameParser(
                        f"{split_name[i + 1]}_{split_name[i + 2]}").get_standard_grid_name()
                except IndexError:
                    self.o_grid_name = GridNameParser(f"{split_name[i + 1]}").get_standard_grid_name()
            elif split_part == "t":
                self.t_grid_name = f"{split_name[i + 1]}"

    def get_standard_full_grid_name(self):
        return f"o_{self.o_grid_name}_b_{self.b_grid_name}_t_{self.t_grid_name}"

    def get_num_b_rot(self):
        return int(self.b_grid_name.split("_")[1])

    def get_num_o_rot(self):
        return int(self.o_grid_name.split("_")[1])


class GridNameParser(NameParser):
    """
    Differently than pure NameParser, GridNameParser raises errors if the name doesn't correspond to a standard grid
    name.
    """
    # TODO: don't simply pass if incorrectly written alg name!
    def __init__(self, name_string: str, o_or_b="o"):
        super().__init__(name_string)
        # num of points 0 or 1 -> always zero algorithm; selected zero algorithm -> always num of points is 1
        # ERROR - absolutely nothing provided
        if self.N is None:
            raise ValueError(f"The number of grid points not recognised in name {self.name_string}.")
        elif self.N <= 0:
            raise ValueError("Cannot have 0 or negative number of points")
        # algorithm not provided but > 1 points -> default algorithm
        elif self.algo is None and self.N >= 1 and o_or_b == "o":
            self.algo = DEFAULT_ALGORITHM_O
        elif self.algo is None and self.N >= 1 and o_or_b == "b":
            self.algo = DEFAULT_ALGORITHM_B


    def get_standard_grid_name(self) -> str:
        return f"{self.algo}_{self.N}"

    def get_alg(self):
        return self.algo

    def get_N(self):
        return self.N

    def get_dim(self):
        return self.dim