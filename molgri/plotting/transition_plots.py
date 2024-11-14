"""
Plots of ITS, eigenvectors and eigenvalues of transition/rate matrix.

A collection of methods to visualise the SqRA or MSM objects.
"""

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.express as px
from scipy import sparse
from plotly.subplots import make_subplots

WIDTH = 600
HEIGHT = 600
NUM_EIGENV = 10

class PlotlyTransitions:

    def __init__(self, is_msm: bool, path_matrix: str = None, path_eigenvalues: str or list = None,
                 path_eigenvectors: str or list = None, tau_array: NDArray = None):
        self.path_matrix = path_matrix
        self.path_eigenvalues = path_eigenvalues
        self.path_eigenvectors = path_eigenvectors
        self.tau_array = tau_array
        self.is_msm = is_msm
        pio.templates.default = "simple_white"
        self.fig = go.Figure()

    def save_to(self, path_to_save: str, width: float = WIDTH, height: float = HEIGHT):
        """
        Save everything that has been added to self.fig so far.

        Args:
            path_to_save (): path to which to save (recommended: .png, pdfs have weird errors)
        """
        self.fig.write_image(path_to_save, width=width, height=height, scale=2)
        self.fig = go.Figure()

    def plot_eigenvalues(self, index_tau: int = 0, **kwargs) -> None:
        """
        This is a function to be used with snakemake (or separately) to generate eigenvalue plots.

        Args:
            input_data_path (): path to a .npy file that contains eigenvalues
            output_data_path (): path to the output file where the figure will be saved
        """
        # data
        eigenvals = np.load(self.path_eigenvalues)[1:NUM_EIGENV+1]
        xs = np.linspace(0, 1, num=NUM_EIGENV)

        if len(eigenvals.shape) == 2:
            eigenvals=eigenvals[index_tau]

        # plotting
        self.fig.add_scatter(x=xs, y=eigenvals, mode='markers+text', text=[f"{el:.3f}" for el in eigenvals],
                        marker=dict(size=8, color="black"), opacity=1, **kwargs)
        # vertical lines
        for i, eigenw in enumerate(eigenvals):
            self.fig.add_shape(type="line", x0=xs[i], y0=0, x1=xs[i], y1=eigenw, line=dict(color="black", width=2),
                          opacity=1, **kwargs)

        # horizontal infinite line
        self.fig.add_hline(y=0, line=dict(color="black", width=2), opacity=1, **kwargs)

        # where the labels are (above or below)
        if self.is_msm:
            self.fig.update_traces(textposition='top center')
        else:
            self.fig.update_traces(textposition='bottom center')
        self.fig.update_yaxes(title="Eigenvalues")
        self.fig.update_layout(xaxis_visible=False, xaxis_showticklabels=False)

    def plot_its_as_line(self) -> None:
        """
        Plot iterative timescales for SQRA methods.

        Args:
            input_data_path (): path to a .npy file that contains eigenvalues
            output_data_path (): path to the output file where the figure will be saved
        """
        # data
        eigenvals = np.load(self.path_eigenvalues)[1:]    # dropping the first one as it should be zero and cause issues
        its = [-1/(eigenval) for eigenval in eigenvals]

        for it in its:
            self.fig.add_hline(it, line=dict(color="black", dash="dash", width=1), opacity=1)
        self.fig.update_layout(xaxis_title=r"$\tau$", yaxis_title="ITS", xaxis=dict(range=[0, 1]),
                          yaxis=dict(range=[0, np.max(its)+0.2]))

    def plot_eigenvectors_flat(self, index_tau: int = 0):
        """

        Args:
            input_data_path (): path to a .npy file that contains eigenvectors
            output_data_path (): path to the output file where the figure will be saved
            eigenvec_index (): which eigenvector to plot
            index_tau (): which tau index to use for plotting

        """
        eigenvecs = np.load(self.path_eigenvectors)
        real_num_eigenv = int(np.min([NUM_EIGENV, eigenvecs.shape[1]]))
        if real_num_eigenv == 0:
            return

        self.fig = make_subplots(rows=real_num_eigenv, cols=1, shared_yaxes='all')
        #subplot_titles=[f"Eigenvector {i}" for i in range(
        #    NUM_EIGENV)]



        # for msm eigenvectors shape: (number_taus, number_cells, num_eigenvectors)
        # else: (number_cells, num_eigenvectors)
        if len(eigenvecs.shape) == 3:
            eigenvecs = eigenvecs[index_tau]

        for i in range(real_num_eigenv):
            data = eigenvecs.T[i]
            self.fig.add_trace(go.Scatter(y=data, line_color="black"), row=i+1, col=1)  # 1-based counting
        for i in range(real_num_eigenv):
            if i == 0:
                self.fig['layout'][f'yaxis']['title'] = f"Eigenv. {i}"
            else:
                self.fig['layout'][f'yaxis{i+1}']['title'] = f"Eigenv. {i}"
        self.fig['layout'][f'xaxis6']['title'] = "Grid cell index"
        self.fig.update_layout(showlegend=False)

    def plot_heatmap(self, index_tau: int = 0):
        """
        This method draws the array and colors the fields according to their values (red = very large,
        blue = very small). Zero values are always white, negative ones always blue, positive ones always red.

        Note that this may cause memory issues for very large matrices.

        Args:
            input_data_path (): path to a .npz file that contains the (sparse) rate matrix
            output_data_path (): path to the output file where the figure will be saved
            index_tau (): for MSM transition matrices that are a 3D numpy arrays, which tau index to use for plotting
        """
        transition_matrix = sparse.load_npz(self.path_matrix)

        if len(transition_matrix.shape) == 3:
            transition_matrix = transition_matrix[index_tau]

        self.fig = px.imshow(transition_matrix.todense(), color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
        # cut-off because the values are over a range of magnitudes
        self.fig.update_coloraxes(cmin=-5, cmid=0, cmax=5)

    def plot_its_msm(self, writeout=5, time_step_ps=0.002) -> None:
        xs = self.tau_array * writeout * time_step_ps
        self.fig = make_subplots(1, 2, shared_yaxes=False)
        row=1
        all_eigenvals = []
        for el in self.path_eigenvalues:
            eigenvals = np.load(el)[1:]  # dropping the first one as it should be zero and cause issues
            eigenvals = list(eigenvals)
            while len(eigenvals) < 4:
                eigenvals.append(np.NaN)
            if len(eigenvals) > 4:
                eigenvals = eigenvals[:4]
            all_eigenvals.append(eigenvals)
        all_eigenvals = np.array(all_eigenvals)

        for col in (1, 2):
            # gray triangle
            self.fig.add_scatter(x=[0, xs[-1], xs[-1]], y=[0, 0, xs[-1]], mode="lines", fill="toself", fillcolor="gray",
                                 line=dict(width=0), row=row, col=col)
            self.fig.update_layout(showlegend=False, xaxis_title=r"$\tau [ps]$", yaxis_title=r"ITS [ps]")
            self.fig.update_xaxes(title_text=r"$\tau [ps]$", row=1, col=col)
            self.fig.update_yaxes(title_text=r"ITS [ps]", row=1, col=col)
            # eigenvalues
            cols = DEFAULT_PLOTLY_COLORS


            max_its = 0
            for i, eigenvals in enumerate(all_eigenvals.T):
                its = np.array(-self.tau_array * writeout * time_step_ps / np.log(np.abs(eigenvals)))
                if np.any(its) > max_its:
                    max_its = np.max(its)
                if col==2:
                    xs = xs[:5]
                    its = its[:5]
                    self.fig.update_xaxes(range=[np.min(xs), np.max(xs)], row=row, col=col)
                self.fig.add_scatter(x=xs, y=its, mode="lines+markers", line=dict(width=2, color=cols[i]), row=row,
                                     col=col)



