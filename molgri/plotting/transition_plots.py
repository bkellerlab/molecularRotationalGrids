"""
Plots of ITS, eigenvectors and eigenvalues of transition/rate matrix.

A collection of methods to visualise the SqRA or MSM objects.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def plot_eigenvalues(input_data_path: str, output_data_path: str) -> None:
    """
    This is a function to be used with snakemake (or separately) to generate eigenvalue plots.

    Args:
        input_data_path (): path to a .npy file that contains eigenvalues
        output_data_path (): path to the output file where the figure will be saved
    """
    # data
    eigenvals = np.load(input_data_path)
    xs = np.linspace(0, 1, num=len(eigenvals))

    # plotting
    fig = go.Figure()
    fig.add_scatter(x=xs, y=eigenvals, mode='markers+text', text=[f"{el:.3f}" for el in eigenvals],
                    marker=dict(size=8, color="black"), opacity=1)
    # vertical lines
    for i, eigenw in enumerate(eigenvals):
        fig.add_shape(type="line", x0=xs[i], y0=0, x1=xs[i], y1=eigenw, line=dict(color="black", width=2),
                      opacity=1)
    # horizontal infinite line
    fig.add_hline(y=0, line=dict(color="black", width=2), opacity=1)

    fig.update_traces(textposition='bottom center')
    fig.update_yaxes(title="Eigenvalues")
    fig.update_layout(xaxis_visible=False, xaxis_showticklabels=False)
    fig.write_image(output_data_path, width=600, height=600)


def plot_its_as_line(input_data_path: str, output_data_path: str) -> None:
    """
    Plot iterative timescales for SQRA methods.

    Args:
        input_data_path (): path to a .npy file that contains eigenvalues
        output_data_path (): path to the output file where the figure will be saved
    """
    # data
    eigenvals = np.load(input_data_path)[1:]    # droping the first one as it should be zero and cause issues
    its = [-1/eigenval for eigenval in eigenvals]

    fig = go.Figure()
    for it in its:
        fig.add_hline(it, line=dict(color="black", dash="dash", width=1), opacity=1)
    fig.update_layout(xaxis_title=r"$\tau$", yaxis_title="ITS", xaxis=dict(range=[0, 1]),
                      yaxis=dict(range=[0, np.max(its)+0.2]))
    fig.write_image(output_data_path, width=600, height=600)


def plot_one_eigenvector_flat(input_data_path: str, output_data_path: str, eigenvec_index: int = 1, index_tau=0):
    """
    
    Args:
        input_data_path (): path to a .npy file that contains eigenvectors
        output_data_path (): 
        eigenvec_index (): 
        index_tau (): 

    Returns:

    """
    eigenvecs = np.load(input_data_path)

    # for msm eigenvectors shape: (number_taus, number_cells, num_eigenvectors)
    # else: (number_cells, num_eigenvectors)
    if len(eigenvecs.shape) == 3:
        eigenvecs = eigenvecs[index_tau]

    data = eigenvecs.T[eigenvec_index]


    fig = go.Figure()
    fig.add_scatter(y=data)
    fig.update_layout(title=f"Eigenvector {eigenvec_index}", yaxis=dict(range=[np.min(data), np.max(data)]))
    fig.write_image(output_data_path, width=600, height=600)

def plot_its_msm(input_data_path: str, output_data_path: str) -> None:
    fig = go.Figure()
    # gray triangle
    fig.add_scatter(x=[0, 1, 1], y=[0, 0, 1], mode="lines", fill="toself", fillcolor="gray",
                    line=dict(width=0))
    pass



# class TransitionPlot(RepresentationCollection):
#
#     def __init__(self, transition_obj: (SimulationHistogram, TransitionModel), tau_array=None, *args, **kwargs):
#         self.simulation_histogram = transition_obj[0]
#         self.transition_obj = transition_obj[1]
#         self.simulation_histogram.use_saved = True
#         self.transition_obj.use_saved = True
#         data_name = self.transition_obj.get_name()
#         super().__init__(data_name, *args, **kwargs)
#
#     @plot_method
#     def plot_heatmap(self, trans_index: int = 0):
#         """
#         This method draws the array and colors the fields according to their values (red = very large,
#         blue = very small). Zero values are always white, negative ones always blue, positive ones always red.
#         """
#         transition_matrix = self.transition_obj.get_transitions_matrix()[trans_index]
#         if np.all(transition_matrix< 0):
#             cmap = "Blues"
#             norm = None
#         elif np.all(transition_matrix > 0):
#             cmap = "Reds"
#             norm = None
#         else:
#             cmap = "bwr"
#             norm = colors.TwoSlopeNorm(vcenter=0, vmax=5, vmin=-5)
#         sns.heatmap(transition_matrix, cmap=cmap, ax=self.ax, xticklabels=False, yticklabels=False, norm=norm)
#         self._equalize_axes()

#
#     def plot_eigenvectors(self, num_eigenvectors: int = 5, projection="3d"):
#         """
#         Visualize the energy surface and the first num (default=5) eigenvectors
#         """
#         self._create_fig_ax(num_columns=num_eigenvectors, projection=projection)
#         for i, subax in enumerate(self.ax.ravel()):
#             self.plot_one_eigenvector(i, ax=subax, fig=self.fig, projection=projection, save=False)
#
#     @plot3D_method
#     def plot_one_eigenvector(self, eigenvec_index: int = 1):
#         eigenvals, eigenvecs = self.transition_obj.get_eigenval_eigenvec()
#
#         # shape: (number_taus, number_cells, num_eigenvectors)
#         eigenvecs = eigenvecs[0]  # values for the first tau
#         eigenvecs = eigenvecs.T
#
#         fgp = FullGridPlot(self.simulation_histogram.full_grid, default_complexity_level="half_empty")
#         fgp.plot_position_voronoi(ax=self.ax, fig=self.fig, plot_vertex_points=False, save=False)
#         fgp.plot_positions(ax=self.ax, fig=self.fig, save=False, animate_rot=False) #, c=eigenvecs[eigenvec_index]
#         self.ax.set_title(f"Eigenv. {eigenvec_index}")
#
#     @plot_method
#     def plot_one_eigenvector_flat(self, eigenvec_index: int = 1, index_tau=0):
#         eigenvals, eigenvecs = self.transition_obj.get_eigenval_eigenvec()
#
#         # shape: (number_taus, number_cells, num_eigenvectors)
#         try:
#             eigenvecs = eigenvecs[index_tau]  # values for the first tau
#         except IndexError:
#             eigenvecs = eigenvecs
#         sns.lineplot(eigenvecs.T[eigenvec_index], ax=self.ax)
#         self.ax.set_title(f"Eigenv. {eigenvec_index}")


if __name__ == "__main__":
    plot_one_eigenvector_flat("output/data/autosave/water_water-bigger_ideal_eigenvectors.npy",
                              "output/figures/test_eval3.png")
    #plot_its_as_line("output/data/autosave/water_water-bigger_ideal_eigenvalues.npy", "output/figures/test_eval2.png")
    #plot_eigenvalues("output/data/autosave/water_water-bigger_ideal_eigenvalues.npy", "output/figures/test_eval.png")

