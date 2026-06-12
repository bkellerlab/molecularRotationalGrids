"""
This is a fairly chaotic collection of plotly utils.
"""

import os
from functools import wraps

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from MDAnalysis import Universe
from matplotlib import pyplot as plt
from scipy.spatial import geometric_slerp
import plotly.graph_objects as go
import plotly.express as px

from molgri.utils.arrays import normalise_vectors
from molgri.utils.spheres import sort_points_on_sphere_ccw


def normalize_marker_sizes(values, min_size=5, max_size=20):
    """Normalize an array of numbers to marker sizes between min_size and max_size"""
    vmin = np.min(values)
    vmax = np.max(values)
    if np.isclose(vmax, vmin):  # all values are the same
        return np.full_like(values, (min_size + max_size) / 2)
    norm_sizes = (values - vmin) / (vmax - vmin)  # scale to 0-1
    norm_sizes = norm_sizes * (max_size - min_size) + min_size
    return norm_sizes

def save_plotly(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        save_as = kwargs.pop("save_as", None)
        save_interactive_as = kwargs.pop("save_interactive_as", None)
        show = kwargs.pop("show", False)
        fig = func(*args, **kwargs)
        if fig is None:
            return None
        if show:
            fig.show()
        if save_as is not None:
            fig.write_image(save_as)
            name, ext = os.path.splitext(save_as)
            fig.write_html(f"{name}.html")
        return fig
    return wrapper


def draw_curve(fig, start_point, end_point, color="black"):
    norm = np.linalg.norm(start_point)
    interpolate_points = np.linspace(0, 1, 2000)
    curve = geometric_slerp(normalise_vectors(start_point), normalise_vectors(end_point), interpolate_points)
    fig.add_trace(go.Scatter3d(x=norm * curve[..., 0], y=norm * curve[..., 1], z=norm * curve[..., 2],
                               mode="lines", line=dict(color=color)))


def draw_spherical_polygon(fig, points, color="black"):
    if len(points) < 3:
        # no poligon, quietly exit
        return
    sorted_points = sort_points_on_sphere_ccw(points)
    # duplicate first point as last point to draw a closed curve
    sorted_points = np.vstack([sorted_points, sorted_points[0]])
    # draw all the arches
    for point1, point2 in zip(sorted_points, sorted_points[1:]):
        draw_curve(fig, point1, point2, color=color)

@save_plotly
def draw_points(points, fig = None, label_by_index: bool = False, custom_labels=None, marker_size=None, color="black",
                colorbar=None, equal_aspect = False, **kwargs):
    if fig is None:
        fig = go.Figure()
    if label_by_index or custom_labels is not None:
        if custom_labels is None:
            custom_labels = list(range(len(points)))
        mode = "text+markers"
        text = custom_labels
    else:
        mode = "markers"
        text = None
    if marker_size is None:
        normalized_marker_size = 5
    else:
        normalized_marker_size = normalize_marker_sizes(marker_size)
    if len(points.shape) == 2 and points.shape[1] == 4:
        # last coordinate of quaternions is shown as opacity
        for point_i, point in enumerate(points):
            opacity = np.min([(point[3] + 1)/2 +0.1, 1])
            if text is not None:
                one_point_text = text[point_i]
            else:
                one_point_text = None
            if isinstance(normalized_marker_size, np.ndarray):
                single_marker_size = normalized_marker_size[point_i]
            else:
                single_marker_size = normalized_marker_size
            fig.add_trace(go.Scatter3d(x=(point[0],), y=(point[1],), z=(point[2],), text=one_point_text, mode=mode,
                                       marker=dict(color=color, opacity=opacity, size=single_marker_size)), **kwargs)
    else:
        fig.add_trace(go.Scatter3d(x=points.T[0], y=points.T[1], z=points.T[2], text=text, mode=mode,
                                   marker=dict(color=color, size=normalized_marker_size, colorbar=colorbar),
                                   **kwargs))
    if equal_aspect:
        fig.update_scenes(aspectmode='data')
    return fig


def draw_line_between(fig, point1, point2, color="black", **kwargs):
    if len(point1) == 3:
        fig.add_trace(
            go.Scatter3d(x=[point1[0], point2[0]],
                         y=[point1[1], point2[1]],
                         z=[point1[2], point2[2]],
                         mode="lines",
                         marker=dict(color=color)), **kwargs)
    else:
        fig.add_trace(
            go.Scatter(x=[point1[0], point2[0]],
                         y=[point1[1], point2[1]],
                         mode="lines",
                         marker=dict(color=color)), **kwargs)

@save_plotly
def show_array(my_array, title: str = "", indices=None, log=False):
    if indices is None:
        indices = list(range(len(my_array)))

    if log:
        data = np.sign(my_array)*np.log10(np.abs(my_array))
    else:
        data = my_array

    fig = px.imshow(data, color_continuous_scale="RdBu") #, zmax=45, zmin=-45
    fig.update_xaxes(
        tickmode="array",
        #tickvals=indices,
        #showgrid=True,
        tickangle=45,
        tickfont=dict(size=9),
    )
    fig.update_yaxes(
        tickmode="array",
        #tickvals=indices,
        #showgrid=True,
        tickfont=dict(size=9)
    )

    nrows, ncols = my_array.shape
    if ncols < 70:
        # draw horizontal grid lines
        for y in range(nrows + 1):
            fig.add_hline(
                y=y - 0.5,
                line_color="black",
                line_width=1
            )

        # draw vertical grid lines
        for x in range(ncols + 1):
            fig.add_vline(
                x=x - 0.5,
                line_color="black",
                line_width=1
            )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=title
    )
    return fig


def show_graph(G, node_property: str = "total_index", edge_property: str = "edge_type",
               save_as = None, show=True):
    labels = {node: node_i for node_i, node in enumerate(sorted(G.nodes))}




    type_to_color = {"r": "red", "spherical": "blue", "rotational": "yellow",
                     "x": "black", "y": "violet", "z": "orange"}
    edge_colors = [type_to_color[edge_data["edge_type"]] for u, v, edge_data in G.edges(data=True)]

    pos = nx.kamada_kawai_layout(G, weight="numerical_edge_type")
    nx.draw(G, pos, labels=labels)

    if G.number_of_edges() > 1:
        if edge_property == "edge_type":
            edge_labels = {(u,v): edge_data[edge_property] for u, v, edge_data in G.edges(data=True)}
        else:
            edge_labels = {(u, v): np.round(edge_data[edge_property], 2) for u, v, edge_data in G.edges(data=True)}
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()

@save_plotly
def show_violin(data, cutoff=None, name=None):
        data = np.array(data)

        # Apply cutoff
        if cutoff is not None:
            data_to_plot = data[data <= cutoff]
        else:
            data_to_plot = data

        fig = go.Figure()

        fig.add_trace(go.Violin(y=data_to_plot, name=name, line_color="black", fillcolor="white",opacity=0.8))

        if cutoff is not None:
            fig.add_shape(type="line",x0=-0.5, x1=0.5,y0=cutoff, y1=cutoff,line=dict(color="red", dash="dash"))
            fig.add_annotation(x=0,y=cutoff, text=f"cutoff = {cutoff}", showarrow=False, yshift=10, font=dict(color="red"))

        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=70, r=20, t=60, b=60),
                          xaxis=dict(showline=True, linewidth=1, linecolor="black",showgrid=False),
                          yaxis=dict(showline=True,linewidth=1,linecolor="black",showgrid=True,gridcolor="lightgray"))
        return fig


@save_plotly
def draw_structure(fig, ag, color="black", side_lengths: NDArray = None, side_angles: NDArray = None, periodic=""):
    unique_types = np.unique(ag.atoms.types)
    some_colors = ["black", "gray", "red", "green", "blue", "yellow"]
    for type_i, atom_type in enumerate(unique_types):
        selected_indices = np.where(ag.atoms.types == atom_type)[0]
        fig = draw_points(ag.atoms.positions[selected_indices], fig=fig, equal_aspect=True, color=some_colors[type_i])

    if side_lengths is not None:
        draw_unit_cell(fig=fig, side_lengths=side_lengths, color="black", side_angles=side_angles)
        if "X" in periodic:
            ag.atoms.positions
    fig.update_layout(showlegend=False)
    return fig


def find_triclinic_vertices(side_lengths: NDArray, side_angles: NDArray = None):
    a, b, c = side_lengths
    alpha, beta, gamma = np.deg2rad(side_angles)
    va = np.array([a, 0.0, 0.0])

    # b-axis in xy-plane
    vb = np.array([
        b * np.cos(gamma),
        b * np.sin(gamma),
        0.0
    ])

    # c-axis general triclinic construction
    cx = c * np.cos(beta)

    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)

    cz_term = (
        1
        - np.cos(alpha) ** 2
        - np.cos(beta) ** 2
        - np.cos(gamma) ** 2
        + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )

    cz = c * np.sqrt(max(cz_term, 0.0)) / np.sin(gamma)

    vc = np.array([cx, cy, cz])
    return va, vb, vc


@save_plotly
def draw_unit_cell(fig: go.Figure, side_lengths: NDArray, side_angles: NDArray = None, color="blue", in_3d = True,
                   v0=[0,0,0], **kwargs):
    """
    Provide a 3x3 array where every row is a lattice vector and get a drawing of a unit cell (all edges)

    Angles should be given in degrees and as [angle(b,c), angle(a, c), angle(a, b)]
    """
    if side_angles is None:
        side_angles = np.array([90, 90, 90])

    if in_3d:
        v0 = np.array(v0)

        # Lx, Ly, Lz = side_lengths
        #
        #
        va, vb, vc = find_triclinic_vertices(side_lengths, side_angles)
        print(f"triclinic vertices {color} ", va + vb + vc)

        vertices = np.array([
            v0,
            v0+va,
            v0+vb,
            v0+vc,
            v0+va + vb,
            v0+va + vc,
            v0+vb + vc,
            v0+va + vb + vc,
        ])

        # vertices = np.array([
        #     v0,
        #     v0 + [va, 0, 0],
        #     v0 + [0, vb, 0],
        #     v0 + [0, 0, vc],
        #     v0 + [va, vb, 0],
        #     v0 + [va, 0, vc],
        #     v0 + [0, vb, vc],
        #     v0 + [va, vb, vc],
        # ])
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7),
        ]

        for start, end in edges:
            draw_line_between(fig, vertices[start], vertices[end], color=color)

        fig.update_layout(scene_aspectmode="data")
        fig.update_layout(showlegend=False)
        print("have fig")
    else:
        Lx, Ly,= side_lengths
        draw_line_between(fig, np.array([0, 0]), np.array([Lx, 0]), color="green")
        draw_line_between(fig, np.array([0, 0]), np.array([0, Ly]), color="green")
        draw_line_between(fig,np.array([Lx, Ly]),np.array([Lx, 0]),color="green")
        draw_line_between(fig,np.array([Lx, Ly]),np.array([0, Ly]),color="green")
        fig.update_xaxes(showgrid=True, range=[-Lx/2, 3*Lx/2])
        fig.update_yaxes(showgrid=True, range=[-Ly/2, 3*Ly/2])
        fig.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1
            )
        )
    return fig
