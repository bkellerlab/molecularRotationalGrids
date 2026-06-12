"""
All rules here should save to network/ folder. Here we create the grid and save properties like volumes, surfaces ...
"""
import numpy as np
import pandas as pd
import matplotlib
import plotly.graph_objects as go
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation, Slerp

from molgri.network.generation import build_quaternion_network, build_translation_network, create_full_network, \
    get_all_rotated_diffusion_matrices
from molgri.images.plotting import show_graph, show_array
from molgri.utils.arrays import normalise_vectors

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


rule evaluate_rotated_diffusion_matrices:
    input:
        info_material = "<outputs_network>grid_info.yaml"
    output:
        rotated_translational_diffusion = "<outputs_network>rotation_network/rotated_translational_diffusion_matrices.npy",
        rotated_rotational_diffusion= "<outputs_network>rotation_network/rotated_rotational_diffusion_matrices.npy"
    params:
        translational_diffusion = config["sqra"]["translational_diffusion_coefficient"],
        rotational_diffusion = config["sqra"]["rotational_diffusion_coefficient"]
    run:
        grid_info = read_object(input.info_material)
        upper_quaternions = np.array(grid_info["quaternions"])

        # translational diffusion
        translational_diffusion_matrix = np.diag(np.array(params.translational_diffusion))
        rotated_diffusion_trans = get_all_rotated_diffusion_matrices(upper_quaternions, translational_diffusion_matrix)
        write_object(rotated_diffusion_trans, output.rotated_translational_diffusion)

        # rotational diffusion
        rotational_diffusion_matrix = np.diag(np.array(params.rotational_diffusion))
        rotated_diffusion_rot = get_all_rotated_diffusion_matrices(upper_quaternions, rotational_diffusion_matrix)
        write_object(rotated_diffusion_rot, output.rotated_rotational_diffusion)

rule get_rotation_index_along_grid:
    """
    Save which quaternion and position relate to which index. Useful for debugging.
    """
    input:
        network= f"<outputs_network>network.pkl",
    output:
        rotation_indices = f"<outputs_network>rotation_indices.npy"
    run:
        my_network = read_object(input.network)
        rotation_indices = my_network.get_rotation_indices()
        write_object(rotation_indices, output.rotation_indices)

rule get_neighbour_diffusion_matrix:
    input:
        grid = "<outputs_network>grid.npy",
        numerical_edge_type = "<outputs_network>edge_types.npz",
        rotated_translational_diffusion = "<outputs_network>rotation_network/rotated_translational_diffusion_matrices.npy",
        rotated_rotational_diffusion= "<outputs_network>rotation_network/rotated_rotational_diffusion_matrices.npy",
        rotation_indices= f"<outputs_network>rotation_indices.npy"
    output:
        neighbour_diffusion_matrix = "<outputs_network>neighbour_diffusion_matrix.npz",
    benchmark:
        "<outputs_network>rotation_network/timing_get_neighbour_diffusion_matrix.txt"
    params:
        rotational_diffusion = config["sqra"]["rotational_diffusion_coefficient"]
    run:
        edge_type = read_object(input.numerical_edge_type).tocoo()

        grid = read_object(input.grid)[:,:3]
        quaternion = read_object(input.grid)[:, 3:]
        rotation_indices = read_object(input.rotation_indices)
        rotated_translational_diffusion_matrices = read_object(input.rotated_translational_diffusion)
        rotated_rotational_diffusion_matrices = read_object(input.rotated_rotational_diffusion)

        base_rotational_diffusion = np.diag(np.array(params.rotational_diffusion))

        diffusion_matrix_data = []
        for i, j, edge_code in zip(edge_type.row, edge_type.col, edge_type.data):
            rotation_index_of_i = rotation_indices[i]
            rotation_index_of_j = rotation_indices[j]
            if edge_code !=4:
                assert rotation_index_of_i == rotation_index_of_j
                # then this is translational edge
                diffusion_matrix = rotated_translational_diffusion_matrices[rotation_index_of_i]
                direction_vector = normalise_vectors(grid[j]-grid[i])
                final_diffusion = direction_vector.T@diffusion_matrix@direction_vector
                diffusion_matrix_data.append(final_diffusion)
            else:
                # is rotational edge, not implemented yet
                diffusion_matrix_i = rotated_rotational_diffusion_matrices[rotation_index_of_i]
                diffusion_matrix_j = rotated_rotational_diffusion_matrices[rotation_index_of_j]
                rot = Rotation.from_quat(np.array([quaternion[i], quaternion[j]]),scalar_first=True)
                my_slerp = Slerp([0, 1],rot)
                N_interpolations = 20
                t = np.linspace(0,1,N_interpolations)
                # has shape (N_interpolations, 3, 3)
                interpolated_rot = my_slerp(t).as_matrix()


                # make my diffusion matrix also the same shape
                base_rotational_diffusion_tiled = np.tile(base_rotational_diffusion,(N_interpolations, 1, 1))


                # using batch transpose, 0th axis unchanged, 1nd and 2st are swapped
                transposed_interpolated_rot = interpolated_rot.transpose(0, 2, 1)
                interpolated_Ds = interpolated_rot@base_rotational_diffusion_tiled@transposed_interpolated_rot
                #print("interpolated_Ds part1 ",np.round(interpolated_rot@base_rotational_diffusion_tiled, 3))

                # direction is now given by the quaternion connecting the start and the end quaternion
                R1 = Rotation.from_quat(np.array(quaternion[i]),scalar_first=True)
                R2 = Rotation.from_quat(np.array(quaternion[j]),scalar_first=True)
                relative_quat = R2 * R1.inv()

                # should the direction also continuisly change?
                direction_vector = normalise_vectors(relative_quat.as_rotvec())
                final_diffusion = direction_vector.T @ interpolated_Ds @ direction_vector
                average_final_diffusion = np.sum(final_diffusion)/N_interpolations

                diffusion_matrix_data.append(average_final_diffusion)

        #print(len(diffusion_matrix_data), edge_type.col.shape, edge_type.col.shape, edge_type.shape, diffusion_matrix[0], diffusion_matrix[-1])
        translational_diffusion_matrix = coo_matrix((diffusion_matrix_data, (edge_type.row, edge_type.col)),shape=edge_type.shape, dtype=np.float64)
        write_object(translational_diffusion_matrix, output.neighbour_diffusion_matrix)



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