from my_constants import *


class NameParser:

    def __init__(self, name: str or dict):
        """
        Correct ordering: '[H2O_HF_]ico[_NO]_500_full_openMM[_extra]
        """
        # define all properties
        self.central_molecule = None
        self.rotating_molecule = None
        self.grid_type = None
        self.ordering = True
        self.num_grid_points = None
        self.traj_type = None
        self.open_MM = False
        self.is_real_run = False
        self.additional_data = None
        self.ending = None
        # parse name
        if isinstance(name, str):
            self._read_str(name)
        elif isinstance(name, dict):
            self._read_dict(name)

    def _read_str(self, name):
        try:
            if "." in name:
                name, self.ending = name.split(".")
        except ValueError:
            pass
        split_str = name.split("_")
        for split_item in split_str:
            if split_item in MOLECULE_NAMES:
                if self.central_molecule is None:
                    self.central_molecule = split_item
                else:
                    self.rotating_molecule = split_item
        for method_name in SIX_METHOD_NAMES:
            if method_name in split_str:
                self.grid_type = method_name
        if "_NO" in name:
            self.ordering = False
        for split_item in split_str:
            if split_item.isnumeric():
                self.num_grid_points = int(split_item)
                break
        for traj_type in ["circular", "full"]:
            if traj_type in split_str:
                self.traj_type = traj_type
        if "openMM" in split_str:
            self.open_MM = True
        if FULL_RUN_NAME in name:
            self.is_real_run = True
        # get the remainder of the string
        if self.central_molecule:
            split_str.remove(self.central_molecule)
        if self.rotating_molecule:
            split_str.remove(self.rotating_molecule)
        if self.grid_type:
            split_str.remove(self.grid_type)
        if self.num_grid_points:
            split_str.remove(str(self.num_grid_points))
        if self.traj_type:
            split_str.remove(self.traj_type)
        if not self.ordering:
            split_str.remove("NO")
        if self.is_real_run:
            split_str.remove(FULL_RUN_NAME)
        if self.open_MM:
            split_str.remove("openMM")
        self.additional_data = "_".join(split_str)

    def _read_dict(self, dict_name):
        self.central_molecule = dict_name.pop("central_molecule", None)
        self.rotating_molecule = dict_name.pop("rotating_molecule", None)
        self.grid_type = dict_name.pop("grid_type", None)
        self.ordering = dict_name.pop("ordering", True)
        self.num_grid_points = dict_name.pop("num_grid_points", None)
        self.traj_type = dict_name.pop("traj_type", None)
        self.open_MM = dict_name.pop("open_MM", False)
        self.is_real_run = dict_name.pop("is_real_run", False)
        self.additional_data = dict_name.pop("additional_data", None)
        self.ending = dict_name.pop("ending", None)

    def get_dict_properties(self):
        return vars(self)

    def get_standard_name(self):
        standard_name = ""
        if self.central_molecule:
            standard_name += self.central_molecule + "_"
        if self.rotating_molecule:
            standard_name += self.rotating_molecule + "_"
        if self.is_real_run:
            standard_name += FULL_RUN_NAME + "_"
        if self.grid_type:
            standard_name += self.grid_type + "_"
        if not self.ordering:
            standard_name += "NO_"
        if self.num_grid_points:
            standard_name += str(self.num_grid_points) + "_"
        if self.traj_type:
            standard_name += self.traj_type + "_"
        if self.open_MM:
            standard_name += "openMM_"
        if standard_name.endswith("_"):
            standard_name = standard_name[:-1]
        return standard_name

    def get_grid_type(self):
        if not self.grid_type:
            raise ValueError(f"No grid type given!")
        return self.grid_type

    def get_traj_type(self):
        if not self.traj_type:
            raise ValueError(f"No traj type given!")
        return self.traj_type

    def get_num(self):
        if not self.num_grid_points:
            raise ValueError(f"No number given!")
        return self.num_grid_points


if __name__ == "__main__":
    from os import walk

    f = []
    for (dirpath, dirnames, filenames) in walk("./data/"):
        f.extend(filenames)
    for el in f:
        np = NameParser(el)
        print(el, np.get_standard_name())
