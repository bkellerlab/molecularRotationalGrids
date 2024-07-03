"""
Plots of ITS, eigenvectors and eigenvalues of transition/rate matrix.

A collection of methods to visualise the SqRA or MSM objects.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from scipy import sparse

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


def plot_one_eigenvector_flat(input_data_path: str, output_data_path: str, eigenvec_index: int = 1, index_tau: int = 0):
    """
    
    Args:
        input_data_path (): path to a .npy file that contains eigenvectors
        output_data_path (): path to the output file where the figure will be saved
        eigenvec_index (): which eigenvector to plot
        index_tau (): which tau index to use for plotting

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


def plot_heatmap(input_data_path: str, output_data_path: str, index_tau: int = 0):
    """
    This method draws the array and colors the fields according to their values (red = very large,
    blue = very small). Zero values are always white, negative ones always blue, positive ones always red.

    Note that this may cause memory issues for very large matrices.

    Args:
        input_data_path (): path to a .npz file that contains the (sparse) rate matrix
        output_data_path (): path to the output file where the figure will be saved
        index_tau (): for MSM transition matrices that are a 3D numpy arrays, which tau index to use for plotting
    """
    transition_matrix = sparse.load_npz(input_data_path)

    if len(transition_matrix.shape) == 3:
        transition_matrix = transition_matrix[index_tau]

    fig = px.imshow(transition_matrix.todense(), color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
    # cut-off because the values are over a range of magnitudes
    fig.update_coloraxes(cmin=-5, cmid=0, cmax=5)
    fig.write_image(output_data_path, width=600, height=600)


def plot_its_msm(input_data_path: str, output_data_path: str) -> None:
    fig = go.Figure()
    # gray triangle
    fig.add_scatter(x=[0, 1, 1], y=[0, 0, 1], mode="lines", fill="toself", fillcolor="gray",
                    line=dict(width=0))
    pass


if __name__ == "__main__":
    plot_heatmap("output/data/autosave/water_water-small_ideal_rate_matrix.npz", "output/figures/test_eval4.png")
    #plot_one_eigenvector_flat("output/data/autosave/water_water-bigger_ideal_eigenvectors.npy",
    #                          "output/figures/test_eval3.png")
    #plot_its_as_line("output/data/autosave/water_water-bigger_ideal_eigenvalues.npy", "output/figures/test_eval2.png")
    #plot_eigenvalues("output/data/autosave/water_water-bigger_ideal_eigenvalues.npy", "output/figures/test_eval.png")

