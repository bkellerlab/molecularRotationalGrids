"""
Using the pickle file, quickly plot the energy surface around the molecule
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, ticker
from mendeleev import element
import MDAnalysis as mda
import matplotlib.colors as colors

from molgri.objects.molecule import H2O
from molgri.plotting.abstract_plot import AbstractPlot, create_all_plots
from molgri.my_constants import *


def _com_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the dataframe from pickle file and calculate the center of mass of the second molecule.

    Args:
        df: dataframe with 'm2_OW_x [A]' and similar elements

    Returns:
        extended DataFrame now including the xyz coordinates of center of mass 'X com [A]', 'Y com [A]', 'Z com [A]'
    """
    #df = find_correct_mirror_images(df, box=(9, 9, 9))
    second_molecule_atoms = []
    for column in df.columns.values.tolist():
        # select data connected to the second molecule
        if column.startswith("m_2"):
            second_molecule_atoms.append(column)
    x_coo = [el for el in second_molecule_atoms if "_x" in el]
    y_coo = [el for el in second_molecule_atoms if "_y" in el]
    z_coo = [el for el in second_molecule_atoms if "_z" in el]
    atomic_masses = [element(el.split("_")[2].capitalize()).atomic_weight for el in x_coo]
    total_mass = np.sum(atomic_masses)
    x_com = np.sum([df[el]*atomic_masses[i] for i, el in enumerate(x_coo)], axis=0)/total_mass
    y_com = np.sum([df[el] * atomic_masses[i] for i, el in enumerate(y_coo)], axis=0) / total_mass
    z_com = np.sum([df[el] * atomic_masses[i] for i, el in enumerate(z_coo)], axis=0) / total_mass
    df["X com [nm]"] = np.round(x_com, 2)
    df["Y com [nm]"] = np.round(y_com, 2)
    df["Z com [nm]"] = np.round(z_com, 2)
    return df


class ComPlot(AbstractPlot):

    def __init__(self, data_name, energy_type="Potential [kJ/mol]", radius=None, plot_type="com", lowest_50=False, **kwargs):
        style_type = kwargs.pop("style_type", ["dark", "talk", "half_empty"])
        plot_type = f"{plot_type}_{ENERGY_FULL2SHORT[energy_type]}"
        self.lowest_50 = kwargs.pop("lowest_50", lowest_50)
        super().__init__(data_name, fig_path=PATH_FIG_COM, style_type=style_type,
                         ani_path=PATH_ANI_COM, plot_type=plot_type, **kwargs)
        self.energy_type = energy_type
        self.radius = radius
        if self.lowest_50:
            self.plot_type += "_mini"

    def _prepare_data(self) -> pd.DataFrame:
        if "openMM" in self.data_name:
            frame_energy = pd.read_pickle(PATH_OMM_RESULTS + self.data_name + ".pkl")
        else:
            frame_energy = pd.read_pickle(PATH_GRO_RESULTS + self.data_name + ".pkl")
        frame_energy = _com_from_df(frame_energy)
        self.minmin = frame_energy[self.energy_type].min()
        self.maxmax = frame_energy[self.energy_type].max()
        if "full" in self.data_name:
            print("initial", len(frame_energy))
            # find rows with same center of mass
            grouped = frame_energy.groupby(["X com [nm]", "Y com [nm]", "Z com [nm]"])
            # find minima for this COM
            frame_energy = grouped.min()
            list_com = np.array([list(x) for x in frame_energy.index.values])
            frame_energy["X com [nm]"] = list_com[:, 0]
            frame_energy["Y com [nm]"] = list_com[:, 1]
            frame_energy["Z com [nm]"] = list_com[:, 2]
            print("grouped by point", len(frame_energy))
            self.minmin = frame_energy[self.energy_type].min()
            self.maxmax = frame_energy[self.energy_type].max()
            frame_energy["radius"] = np.round(np.sqrt(
                frame_energy["X com [nm]"] ** 2 + frame_energy["Y com [nm]"] ** 2 + frame_energy["Z com [nm]"] ** 2), 2)
            if self.radius is not None:
                print(np.unique(frame_energy["radius"].values))
                # rad = 0.3
                frame_energy = frame_energy[frame_energy["radius"] == self.radius]
                # print("selected radius", len(frame_energy))
                self.plot_type += f"_{self.radius}"
        if self.lowest_50:
            ten_pro_data = len(frame_energy) // 100
            num_points = 100
            print("num points", num_points)
            all_frame_energy = frame_energy.sort_values(self.energy_type)
            frame_energy1 = all_frame_energy[:num_points]
            all_frame_energy = frame_energy.sort_values(self.energy_type, ascending=False)
            frame_energy2 = all_frame_energy[:num_points]
            frame_energy = pd.concat((frame_energy1, frame_energy2))
        return frame_energy

    def _plot_data(self, **kwargs):
        data = self._prepare_data()
        if self.lowest_50:
            #cmap = plt.get_cmap("Blues_r")
            cmap = plt.get_cmap("coolwarm")
        else:
            cmap = plt.get_cmap("coolwarm")
        if "panel" in self.plot_type:
            if "full" in self.data_name:
                s = 1
            else:
                s = 5
        else:
            s = 20
        # plotting the water in the middle
        if self.data_name.startswith("H2O"):
            central_water = H2O()
            central_water.draw(self.ax)
            self.ax.view_init(elev=10, azim=10)
            if self.energy_type == "LJ Energy [kJ/mol]":
                norm=colors.Normalize(vmin=self.minmin, vmax=self.maxmax) #vmin=-0.6, vmax=1
                format=None
            elif self.energy_type == "Coulomb [kJ/mol]" or self.energy_type == "Potential [kJ/mol]":
                norm = colors.SymLogNorm(10, vmin=self.minmin, vmax=self.maxmax)
                #norm = colors.Normalize(vmin=self.minmin, vmax=self.maxmax)  #vmin=-30, vmax=0
                format = None
            else:
                norm = colors.SymLogNorm(0.01) #
                format = ticker.LogFormatterMathtext()
        elif self.data_name.startswith("protein"):
            # plot backbone
            u = mda.Universe(f"{PATH_GENERATED_GRO_FILES}{self.parsed_data_name.central_molecule}_{self.parsed_data_name.rotating_molecule}_run.gro")
            backbone = u.select_atoms("name CA")
            self.ax.scatter(*backbone.positions.T/10, color="black", s=1)
            self.ax.view_init(elev=10, azim=45)
            #print(min(data[self.energy_type].max(), 1e5), max(data[self.energy_type].min(), -1e5))
            #norm = colors.SymLogNorm(1, vmax=min(data[self.energy_type].max(), 1e5), vmin=max(data[self.energy_type].min(), -1e6))
            #norm = colors.SymLogNorm(0.1, vmin=data[self.energy_type].min(), vmax=data[self.energy_type].max())
            if self.energy_type == "LJ Energy [kJ/mol]":
                norm = colors.SymLogNorm(1, vmax=min(data[self.energy_type].max(), 1e5),
                                         vmin=max(data[self.energy_type].min(), -1e6))
            else:
                norm=None
            format = ticker.LogFormatterMathtext()
        else:
            norm=None
            format = None
        p = self.ax.scatter(data["X com [nm]"], data["Y com [nm]"], data["Z com [nm]"], "o",
                            c=data[self.energy_type], s=s, cmap=cmap, norm=norm, **kwargs)
        cbar = plt.gcf().colorbar(p, ax=self.ax, fraction=0.046, pad=0.04, format=format)
        #if norm is None:
        #    cbar.formatter.set_powerlimits((-2, 2))
        #    cbar.ax.yaxis.set_offset_position('left')
        # elif "protein" not in self.data_name:
        #     m0 = -30
        #     m4 = 30
        #     m1 = int(1 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 1
        #     m2 = int(2 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 2
        #     m3 = int(3 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 3
        #     cbar.set_ticks([m0, m1, m2, m3, m4])
        #     cbar.set_ticklabels([m0, m1, m2, m3, m4])
        if "protein" in self.data_name and self.energy_type == "Coulomb [kJ/mol]":
            m0 = data[self.energy_type].min()
            m4 = data[self.energy_type].max()
            m1 = int(1 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 1
            m2 = int(2 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 2
            m3 = int(3 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 3
            cbar.set_ticks(np.round([m0, m1, m2, m3, m4]))
            cbar.set_ticklabels(np.round([m0, m1, m2, m3, m4], 0))
        elif "protein" in self.data_name and self.energy_type == "LJ Energy [kJ/mol]":
            m0 = data[self.energy_type].min()
            m4 = data[self.energy_type].max()
            # m1 = int(1 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 1
            # m2 = int(2 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 2
            # m3 = int(3 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 3
            # cbar.set_ticks(np.round([m0, m1, m2, m3, m4]))
            # cbar.set_ticklabels(np.round([m0, m1, m2, m3, m4], 0))
            # tl = cbar.ax.get_yticklabels()
            # new_labels = []
            # for t in tl:
            #     t.get_text()
            #     new_labels.append(t)
            # print(new_labels)
            # new_labels[-1] = "$>10^5$"
            # cbar.set_ticklabels(tl)
        #plt.tight_layout()

    def create(self, **kwargs):
        kwargs.pop("title", None)
        #l50 = kwargs.pop("lowest_50", False)
        #self.lowest_50 = l50
        title = NAME2PRETTY_NAME[self.parsed_data_name.grid_type]
        if self.data_name.startswith("protein"):
            pos_lim = 3
        else:
            pos_lim = 0.5
        super(ComPlot, self).create(title=title, equalize=True, pos_limit=pos_lim, **kwargs)

