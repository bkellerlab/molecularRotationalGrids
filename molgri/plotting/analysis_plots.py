import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike

from molgri.constants import CELLS_DF_COLUMNS
from molgri.molecules.parsers import XVGParser
from molgri.paths import PATH_OUTPUT_CELLS, PATH_INPUT_ENERGIES
from molgri.plotting.abstract import AbstractPlot


class VoranoiConvergencePlot(AbstractPlot):

    def __init__(self, data_name: str, style_type=None, plot_type="areas"):
        super().__init__(data_name, dimensions=2, style_type=style_type, plot_type=plot_type)

    def _prepare_data(self) -> object:
        return pd.read_csv(f"{PATH_OUTPUT_CELLS}{self.data_name}.csv")

    def _plot_data(self, color=None, **kwargs):
        N_points = CELLS_DF_COLUMNS[0]
        voranoi_areas = CELLS_DF_COLUMNS[2]
        ideal_areas = CELLS_DF_COLUMNS[3]
        time = CELLS_DF_COLUMNS[4]
        voranoi_df = self._prepare_data()
        sns.lineplot(data=voranoi_df, x=N_points, y=voranoi_areas, errorbar="sd", color=color, ax=self.ax)
        sns.scatterplot(data=voranoi_df, x=N_points, y=voranoi_areas, alpha=0.8, color="black", ax=self.ax, s=1)
        sns.scatterplot(data=voranoi_df, x=N_points, y=ideal_areas, color="black", marker="x", ax=self.ax)
        ax2 = self.ax.twinx()
        ax2.set_yscale('log')
        ax2.set_ylim(10**-3, 10**3)
        sns.lineplot(data=voranoi_df, x=N_points, y=time, color="black", ax=ax2)

    def create(self, *args, **kwargs):
        super().create(*args, **kwargs)
        self.ax.set_xscale('log')


class EnergyConvergencePlot(AbstractPlot):

    def __init__(self, data_name: str, test_Ns=None, property_name="Potential", no_convergence=False,
                 plot_type="energy_convergence", **kwargs):
        self.test_Ns = test_Ns
        self.property_name = property_name
        self.unit = None
        self.no_convergence = no_convergence
        super().__init__(data_name, plot_type=plot_type, **kwargs)

    def _prepare_data(self) -> pd.DataFrame:
        file_name = f"{PATH_INPUT_ENERGIES}{self.data_name}"
        file_parsed = XVGParser(file_name)
        self.property_name, correct_column = file_parsed.get_column_index_by_name(self.property_name)
        self.unit = file_parsed.get_y_unit()
        df = pd.DataFrame(file_parsed.all_values[:, correct_column], columns=[self.property_name])
        # select points that fall within each entry in test_Ns
        self.test_Ns = test_or_create_Ns(len(df), self.test_Ns)
        if self.no_convergence:
            self.test_Ns = [self.test_Ns[-1]]
        points_up_to_Ns(df, self.test_Ns, target_column=self.property_name)
        return df

    def _plot_data(self, **kwargs):
        df = self._prepare_data()
        new_column_names = [f"{i}" for i in self.test_Ns]
        sns.violinplot(df[new_column_names], ax=self.ax, scale="count", inner="stick", cut=0)
        self.ax.set_xlabel("N")
        if self.unit:
            self.ax.set_ylabel(f"{self.property_name} [{self.unit}]")
        else:
            self.ax.set_ylabel(self.property_name)


def points_up_to_Ns(df: pd.DataFrame, Ns: ArrayLike, target_column: str):
    """
    Introduce a new column into the dataframe for every N  in the Ns list. The new columns contain the value from
     target_column if the row index is <= N. Every value in the Ns list must be smaller or equal the length of the
     DataFrame. The DataFrame is changed in-place.

    Why is this helpful? For plotting convergence plots using more and more of the data.

    Example:

    # Initial dataframe df:
        index   val1    val2
        1        a       b
        2        c       d
        3        e       f
        4        g       h

    # After applying points_up_to_Ns(df, Ns=[1, 3, 4], target_column='val2'):
        index   val1    val2    1       3       4
        1        a       b      b       b       b
        2        c       d              d       d
        3        e       f              f       f
        4        g       h                      h

    # Plotting the result as separate items
    sns.violinplot(df["1", "3", "4"], ax=self.ax, scale="area", inner="stick")
    """
    # create helper columns under_N where the value is 1 if index <= N
    selected_Ns = np.zeros((len(Ns), len(df)))
    for i, point in enumerate(Ns):
        selected_Ns[i][:point] = 1
    column_names = [f"under {i}" for i in Ns]
    df[column_names] = selected_Ns.T
    # create the columns in which values from target_column are copied
    for point in Ns:
        df[f"{point}"] = df.where(df[f"under {point}"] == 1)[target_column]
    # remover the helper columns
    return df


def test_or_create_Ns(max_N: int, Ns: ArrayLike = None,  num_test_points=5) -> ArrayLike:
    if Ns is None:
        Ns = np.linspace(0, max_N, num_test_points+1, dtype=int)[1:]
    else:
        assert np.all([np.issubdtype(x, np.integer) for x in Ns]), "All tested Ns must be integers."
        assert np.max(Ns) <= max_N, "Test N cannot be larger than the number of points"
    return Ns
