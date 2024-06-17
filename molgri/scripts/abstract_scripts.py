import os
from abc import ABC

import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError

from molgri.paths import PATH_OUTPUT_LOGBOOK, PATH_OUTPUT_AUTOSAVE

import warnings
warnings.filterwarnings("ignore")


class Logbook(ABC):
    """
    For every class that is derived from ScientificObject, 1 Logbook will be started. A logbook is initiated once and
    keeps recording all instances of this object and its given input parameters.
    """

    def __init__(self, class_name: str, path_to_use: str, *, class_parameter_names: list = None):
        self.class_name = class_name
        self.class_parameter_names = class_parameter_names
        self.class_logbook_path = f"{path_to_use}{self.class_name}.csv"
        self.current_logbook = self._load_current_logbook()

    def _load_current_logbook(self) -> pd.DataFrame:
        try:
            read_csv = pd.read_csv(self.class_logbook_path, index_col=0, dtype=object)
            self.class_parameter_names = read_csv.columns
            return read_csv
        except (FileNotFoundError, EmptyDataError):
            open(self.class_logbook_path, mode="w").close()
            return pd.DataFrame(columns=self.class_parameter_names)

    def get_current_logbook(self):
        """
        Returns:
            A pandas dataframe including all instances that have been constructed so far.
        """
        return self._load_current_logbook()

    def get_class_index(self, use_saved: bool, parameter_names: list, parameter_values: list):
        assert len(parameter_names) == len(parameter_values), f"len({parameter_names})!=len({parameter_values})"
        parameter_names_values = {n: v for n, v in zip(parameter_names, parameter_values)}
        new_entry = self._get_new_entry(parameter_names_values)
        if use_saved is True:
            # try finding an existing set of data
            existing_index = None
            for index, data in self.current_logbook.iterrows():
                for i, row in new_entry.iterrows():
                    if row.equals(data):
                        existing_index = index
            if existing_index is not None:
                return existing_index
        # if not use_saved or doesn't exist yet, create new entry
        self._record_new_entry(new_entry)
        return len(self.current_logbook) - 1  # minus 1 because we just added this new one

    def _get_new_entry(self, parameter_names_values: dict):
        current_len = len(self.current_logbook)
        empty_df = pd.DataFrame(columns=self.class_parameter_names)
        for existing_title in self.class_parameter_names:
            empty_df.loc[current_len, existing_title] = np.NaN
        for title, data in parameter_names_values.items():
            empty_df.loc[current_len, title] = data
        empty_df.to_csv(f"{PATH_OUTPUT_LOGBOOK}temp.csv")
        empty_df = pd.read_csv(f"{PATH_OUTPUT_LOGBOOK}temp.csv", index_col=0, dtype=object)
        return empty_df

    def _record_new_entry(self, new_entry: pd.DataFrame):
        self.current_logbook = pd.concat([self.current_logbook, new_entry])
        # update the file immediately
        self.current_logbook.to_csv(self.class_logbook_path)
        self.current_logbook = self._load_current_logbook()


class ScriptLogbook:

    """
    The Logbook to be used for methods.
    """

    def __init__(self, my_parser):
        self.my_parser = my_parser
        root, ext = os.path.splitext(my_parser.prog)
        self.my_args = self.my_parser.parse_args().__dict__
        self.use_saved = not self.my_args.pop("recalculate")
        # before calculation is done time is zero and failed=True
        self.my_args["Time [s]"] = None
        self.my_args["failed"] = True
        self.class_name = root
        self.class_logbook_path = f"{PATH_OUTPUT_LOGBOOK}{self.class_name}.csv"
        self.current_logbook = self._load_current_logbook()
        self.is_newly_assigned = None
        self.my_index = self._get_class_index()

    def _load_current_logbook(self) -> pd.DataFrame:
        try:
            read_csv = pd.read_csv(self.class_logbook_path, index_col=0, dtype=object)
            self.class_parameter_names = read_csv.columns
            return read_csv
        except (FileNotFoundError, EmptyDataError):
            open(self.class_logbook_path, mode="w").close()
            return pd.DataFrame()

    def get_current_logbook(self):
        """
        Returns:
            A pandas dataframe including all instances that have been constructed so far.
        """
        return self._load_current_logbook()

    def _get_class_index(self):
        """
        Only calling this in __init__ and never again!
        Returns:

        """
        parameter_names = self.my_args.keys()
        parameter_values = self.my_args.values()
        assert len(parameter_names) == len(parameter_values), f"len({parameter_names})!=len({parameter_values})"
        parameter_names_values = {n: v for n, v in zip(parameter_names, parameter_values)}
        new_entry = self._get_new_entry(parameter_names_values)
        if self.use_saved is True:
            # try finding an existing set of data
            # only compare the columns that contain arguments!
            existing_index = None
            for index, data in self.current_logbook[parameter_names].iterrows():
                for i, row in new_entry[parameter_names].iterrows():
                    if row.equals(data):
                        print(existing_index)
                        existing_index = index
            if existing_index is not None:
                self.is_newly_assigned = False
                return existing_index
        # if not use_saved or doesn't exist yet, create new entry
        self.is_newly_assigned = True
        self._record_new_entry(new_entry)
        return len(self.current_logbook) - 1  # minus 1 because we just added this new one

    def _get_new_entry(self, parameter_names_values: dict):
        current_len = len(self.current_logbook)
        empty_df = pd.DataFrame()
        for title, data in parameter_names_values.items():
            empty_df.loc[current_len, title] = data
        empty_df.to_csv(f"{PATH_OUTPUT_LOGBOOK}temp.csv")
        empty_df = pd.read_csv(f"{PATH_OUTPUT_LOGBOOK}temp.csv", index_col=0, dtype=object)
        return empty_df

    def _record_new_entry(self, new_entry: pd.DataFrame):
        self.current_logbook = pd.concat([self.current_logbook, new_entry])
        # update the file immediately
        self.current_logbook.to_csv(self.class_logbook_path)
        self.current_logbook = self._load_current_logbook()

    def update_after_calculation(self, time_in_s: float):
        self.current_logbook = self._load_current_logbook()
        if not self.use_saved:
            self.current_logbook.loc[self.current_logbook.index[self.my_index], "Time [s]"] = time_in_s
        self.current_logbook.loc[self.current_logbook.index[self.my_index], "failed"] = False
        self.current_logbook.to_csv(self.class_logbook_path)

    def add_information(self, dict_of_info, overwrite=False):
        self.current_logbook = self._load_current_logbook()
        if not self.use_saved or overwrite:
            for n, v in dict_of_info.items():
                self.current_logbook.loc[self.current_logbook.index[self.my_index], n] = v
        self.current_logbook.to_csv(self.class_logbook_path)