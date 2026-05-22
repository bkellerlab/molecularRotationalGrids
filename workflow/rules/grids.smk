"""
All rules here should save to network/ folder. Here we create the grid and save properties like volumes, surfaces ...
"""
import numpy as np
import pandas as pd
import matplotlib
import plotly.graph_objects as go

from molgri.network.generation import build_quaternion_network, build_translation_network, create_full_network
from molgri.images.plotting import show_graph, show_array

from workflow.helpers.io import write_object, read_object
from workflow.helpers.build_subgrids import make_grid_base


matplotlib.use('Agg')

rule save_basic_grid_information:
    """
    Grid information is a useful file because it saves properties like number of rotational, translational and total
    grid points, grid limits, sub-grids that were used to create the full grid etc.
    """
    output:
        info_material = "<outputs_network>grid_info.yaml"
    run:
        save_information = make_grid_base(config)
        write_object(save_information, output.info_material)

rule create_rotation_network:
    input:
        info_material = "<outputs_network>grid_info.yaml"
    benchmark:
        "<outputs_network>rotation_network/network_creation.txt"
    output:
        network_file = "<outputs_network>rotation_network/network.pkl"
    run:
        grid_info = read_object(input.info_material)
        upper_quaternions = np.array(grid_info["quaternions"])

        rotation_network = build_quaternion_network(upper_quaternions)
        write_object(rotation_network, output.network_file)


rule create_translation_network:
    input:
        info_material = "<outputs_network>grid_info.yaml"
    benchmark:
        "<outputs_network>translation_network/network_creation.txt"
    output:
        network_file = "<outputs_network>translation_network/network.pkl"
    run:
        grid_info = read_object(input.info_material)
        periodic_in = grid_info["periodic_in"]
        subgrids = grid_info["subgrid_points"]
        translation_network = build_translation_network(subgrids,periodic_in)
        write_object(translation_network, output.network_file)


rule create_full_network:
    input:
        rotation_network_file = "<outputs_network>rotation_network/network.pkl",
        translation_network_file = "<outputs_network>translation_network/network.pkl"
    benchmark:
        f"<outputs_network>network_creation.txt"
    output:
        network_file = f"<outputs_network>network.pkl",
    run:
        rotation_network = read_object(input.rotation_network_file)
        translation_network = read_object(input.translation_network_file)
        full_network = create_full_network(translation_network, rotation_network)
        write_object(full_network, output.network_file)

if config["sqra"]["allow_diffusion_to_bulk"]:
    rule save_to_bulk:
        input:
            network_file = "<outputs_network>network.pkl"
        output:
            boundaries_to_bulk = f"<outputs_network>boundaries_to_bulk.npy",
            volumes_to_bulk= f"<outputs_network>volumes_to_bulk.npy"
        run:
            full_network = read_object(input.network_file)
            boundaries_to_bulk = full_network.get_surface_to_bulk()
            write_object(boundaries_to_bulk,output.boundaries_to_bulk)
            volumes_to_bulk = full_network.get_volumes_to_bulk()
            write_object(volumes_to_bulk,output.volumes_to_bulk)

rule save_network_properties:
    input:
        network_file = "<outputs_network>network.pkl"
    benchmark:
        "<outputs_network>saving_properties.txt"
    output:
        grid = "<outputs_network>grid.npy",
        adjacency = "<outputs_network>adjacency.npz",
        numerical_edge_type = "<outputs_network>edge_types.npz",
        distances = "<outputs_network>distances.npz",
        surfaces = "<outputs_network>surfaces.npz",
        volumes = "<outputs_network>volumes.npz",
    run:
        full_network = read_object(input.network_file)
        write_object(full_network.grid, output.grid)
        write_object(full_network.adjacency_volume, output.volumes)

        write_object(full_network.adjacency_matrix, output.adjacency)
        write_object(full_network.adjacency_type_matrix,output.numerical_edge_type)
        write_object(full_network.distance_matrix,output.distances)
        write_object(full_network.surface_matrix,output.surfaces)

rule display_network:
    input:
        network_file = "<outputs_network>network.pkl"
    output:
        plot = "<outputs_network>network.png"
    run:
        my_network = read_object(input.network_file)
        show_graph(my_network,edge_property="distance", show=False, save_as=output.plot)


rule display_network_edge_matrices:
    input:
        adjacency = "<outputs_network>adjacency.npz",
        numerical_edge_type = "<outputs_network>edge_types.npz",
        distances = "<outputs_network>distances.npz",
        surfaces = "<outputs_network>surfaces.npz",
        volumes= "<outputs_network>volumes.npz",
    output:
        adjacency = "<outputs_network>adjacency.png",
        numerical_edge_type = "<outputs_network>edge_types.png",
        distances = "<outputs_network>distances.png",
        surfaces = "<outputs_network>surfaces.png",
        volumes= "<outputs_network>volumes.png",
    run:
        show_array(read_object(input.adjacency).toarray(), "Adjacency_type",
            save_as=output.adjacency, show=False)
        show_array(read_object(input.numerical_edge_type).toarray(),"Edge types",
            save_as=output.numerical_edge_type, show=False)
        show_array(read_object(input.distances).toarray(), "Distance_matrix",
            save_as=output.distances, show=False)
        show_array(read_object(input.surfaces).toarray(), "Surface_matrix",
            save_as=output.surfaces, show=False)
        show_array(read_object(input.volumes).toarray(),"Volume_matrix",
            save_as=output.volumes,show=False)

rule display_network_node_attributes:
    input:
        grid = "<outputs_network>grid.npy",
        volumes= "<outputs_network>volumes.npy"
    output:
        grid = "<outputs_network>grid.png",
        volumes = "<outputs_network>volumes.png"
    run:
        from molgri.images.plotting import draw_points
        grid = read_object(input.grid)
        draw_points(grid, save_as=output.grid, show=False)
        volumes = read_object(input.volumes)
        draw_points(grid, custom_labels=np.round(volumes,2), save_as=output.volumes, marker_size=volumes,
            show=False)


rule create_index_csv:
    """
    Save which quaternion and position relate to which index. Useful for debugging.
    """
    input:
        network= f"<outputs_network>network.pkl",
    output:
        energy_csv = f"<outputs_network>indices_interpretation.csv"
    run:
        my_network = read_object(input.network)

        translation_indices = my_network.get_translation_indices()
        rotation_indices = my_network.get_rotation_indices()
        coordinates = my_network.grid
        positions = coordinates[:, :3]
        quaternions = coordinates[:, 3:]

        df = pd.DataFrame(np.array([translation_indices, rotation_indices]).T,
            columns=["Translation index", "Rotation index"])


        df[["x", "y", "z"]] = pd.DataFrame(list(map(tuple, positions.astype(float))),index=df.index)
        df[["q_0", "q_1", "q_2", "q_3"]] = pd.DataFrame( list(map(tuple, quaternions.astype(float))),index=df.index)

        tuples = [
            ("Translation index", ""),
            ("Rotation index", ""),
            ("Position", "x"),
            ("Position", "y"),
            ("Position", "z"),
            ("Quaternion", "q_0"),
            ("Quaternion", "q_1"),
            ("Quaternion", "q_2"),
            ("Quaternion", "q_3"),
        ]

        df.index.name = "Total index"

        # indices should be integers
        df["Translation index"] = df["Translation index"].astype("Int64")
        df["Rotation index"] = df["Rotation index"].astype("Int64")

        df.columns = pd.MultiIndex.from_tuples(tuples)
        print(df)
        write_object(df, output.energy_csv)



rule display_geometry_properties_with_violin_distributions:
    """
    Not all that useful, maybe for small grids to see that the properties are not too wildly different.
    """
    input:
        volumes = "<outputs_network>volumes.npy",
        distances= "<outputs_network>distances.npz",
        surfaces= "<outputs_network>surfaces.npz"
    output:
        volumes = "<outputs_network>violin_plots.png"
    run:
        volume_data = read_object(input.volumes)
        distance_data = read_object(input.distances).data
        surface_data = read_object(input.surfaces).data

        fig = go.Figure()

        arrays = [distance_data, surface_data, volume_data]
        labels = ["Distance", "Surface", "Volume"]


        for i, (arr, label) in enumerate(zip(arrays, labels)):
            fig.add_trace(go.Violin(y=arr, name=label))
            x = label

            mn = np.min(arr)
            mx = np.max(arr)
            mean = np.mean(arr)
            if np.allclose([mn, mx, mean], mn):
                fig.add_annotation(
                    x=x,
                    y=mean,
                    text=f"{mean:.2f}",
                    showarrow=False,
                    font=dict(color="red",size=12),
                    yshift=0,
                    xshift=50,
                )
                continue

            # Annotate min
            fig.add_annotation(
                x=x,
                y=mn,
                showarrow=False,
                text=f"min={mn:.2f}",
                yshift=0,
                xshift=50,
            )

            # Annotate max
            fig.add_annotation(
                x=x,
                y=mx,
                showarrow=False,
                text=f"max={mx:.2f}",
                yshift=0,
                xshift=50,
            )

            # Annotate mean
            fig.add_annotation(
                x=x,
                y=mean,
                text=f"mean={mean:.2f}",
                showarrow=False,
                font=dict(color="red",size=12),
                yshift=0,
                xshift=50,
            )


        fig.write_image(output.volumes)