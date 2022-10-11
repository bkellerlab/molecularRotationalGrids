import numpy as np

from ..grids.grid import build_grid
from ..plotting.abstract_plot import AbstractPlot
from ..my_constants import *
from ..paths import PATH_OUTPUT_GRIDPLOT, PATH_OUTPUT_GRID_ANI


class GridPlot(AbstractPlot):

    def __init__(self, data_name, empty=True, title=True, plot_type="grid", **kwargs):
        if empty:
            style_type = ["talk", "empty"]
            plot_type = "e" + plot_type
        else:
            style_type = ["talk", "half_empty"]
        super().__init__(data_name, fig_path=PATH_OUTPUT_GRIDPLOT, style_type=style_type,
                         ani_path=PATH_OUTPUT_GRID_ANI, plot_type=plot_type, **kwargs)
        self.title = title

    def _prepare_data(self) -> np.ndarray:
        num = self.parsed_data_name.get_num()
        orig_name = self.parsed_data_name.get_grid_type()
        my_grid = build_grid(orig_name, num, use_saved=True).get_grid()
        return my_grid

    def _plot_data(self, **kwargs):
        my_grid = self._prepare_data()
        self.ax.scatter(*my_grid.T, color="black", s=4)
        self.ax.view_init(elev=10, azim=30)

    def create(self, **kwargs):
        short_gt = NAME2SHORT_NAME[self.parsed_data_name.grid_type]
        title_ex = f"{short_gt} grid, {self.parsed_data_name.num_grid_points} points"
        title = title_ex if self.title else None
        if "empty" in self.style_type:
            pad_inches = -0.2
        else:
            pad_inches = 0
        super(GridPlot, self).create(equalize=True, pos_limit=1, pad_inches=pad_inches, title=title, **kwargs)


if __name__ == "__main__":
    GridPlot("cube3D_98", empty=False, title=False).create()

