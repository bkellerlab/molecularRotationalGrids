import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike

from molgri.constants import CELLS_DF_COLUMNS
from molgri.molecules.parsers import XVGParser
from molgri.paths import PATH_INPUT_ENERGIES
from molgri.plotting.abstract import RepresentationCollection


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
