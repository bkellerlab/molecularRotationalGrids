"""
These functions can be used directly to quickly access some information
"""
import pandas as pd
from scipy.sparse import csr_array
from scipy.sparse.linalg import eigs

from molgri.images.plotting import show_array
from molgri.molecules.bimolecular import move_universe_to_xy_plane
from molgri.molecules.rate_merger import delete_rows_columns, sqra_determine_indices_never_visited_states, \
    msm_determine_indices_never_visited_states
from molgri.molecules.transitions import DecompositionTool, SQRA, auto_determine_eigenvector_extremes
from workflow.helpers.io import read_object, write_object

rule print_lowest_energies:
    """
    Use this rule if you want to quickly look at the indices of the lowest energies.
    """
    input:
        energy_csv =f"<pseudosimulation>energy.csv"
    run:
        df = read_object(input.energy_csv)
        print(df.loc[71582], df.iloc[71582])
        df = df.sort_values(by="Energy [kJ/mol]",ascending=True)
        #print(df.head(20))
        print(df.loc[71582], df.iloc[71582])
        df = df.sort_index()
        print(df.loc[71582],df.iloc[71582])
        #df = df.sort_values(by="Energy [kJ/mol]",ascending=False)
        #print(df.head(20))

rule print_position_assignment:
    input:
        energy_csv = rules.position_assignment_csv.output.assignment_csv
    run:
        df = read_object(input.energy_csv)
        print(df.loc[83546])

rule print_assignment:
    input:
        trans_csv = f"<outputs_assignment>translation_assignment.csv",
        translation_assignment = f"<outputs_assignment>translation_assignment.npy",
        rot_assignment = f"<outputs_assignment>rotation_assignment.npy",
        full_assignment = f"<outputs_assignment>full_assignment.npy",
    run:
        my_assignments = read_object(input.rot_assignment)
        print("Rot assignment: ", my_assignments[[0,1,2,3,4,5,6]])
        my_assignments = read_object(input.translation_assignment)
        print("Trans assignment: ", my_assignments[[0,1,2,3,4,5,6]])
        my_assignments = read_object(input.full_assignment)
        print(my_assignments[[0,1,2,3,4,5,6]])

        df = read_object(input.trans_csv, header = [0,1])
        print(df)
        #print(df.loc[83546])


rule print_indices_interpretation:
    """
    Use this rule if you want to quickly look at the indices and understand them.
    """
    input:
        indices_csv =f"<outputs_network>indices_interpretation.csv"
    run:
        df = read_object(input.indices_csv)
        for index in [159, 168, 150, 125, 71]:
            print(df.loc[f"{index}"])
        print("High E")
        for index in [161, 170, 152, 224, 98, 233]:
            print(df.loc[f"{index}"])
        # for example only filter the ones with specific rotation index
        #df_filtered = df.loc[df["Rotation index"] == 5]
        #print(df_filtered.head(10))
rule print_grid_interpretation:
    """
    Use this rule if you want to quickly look at the indices and understand them.
    """
    input:
        indices_csv =f"nobackup/benzene_benzene/162_sph_272rot/network/grid.npy"
    run:
        import numpy as np
        my_grid = read_object(input.indices_csv)
        print(len(my_grid))

rule print_position_subgrids:
    input:
        grid_info = rules.save_basic_grid_information.output.info_material
    run:
        grid_info = read_object(input.grid_info)
        print("X gridpoints: ", grid_info["subgrid_points"][0])
        print("Y gridpoints: ", grid_info["subgrid_points"][1])
        print("Z gridpoints: ", grid_info["subgrid_points"][2])


import numpy as np

rule print_rate_matrix:
    input:
        rate_matrix = "<outputs_transitions>sqra/sqra.npz",
        reduced_rate_matrix = "<outputs_transitions>sqra/reduced_sqra.npz" #reduced_sqra
    run:
        matrix = read_object(input.rate_matrix).todense()
        print("Shape full matrix ", matrix.shape)
        matrix = read_object(input.reduced_rate_matrix).todense()
        print("Shape reduced matrix ", matrix.shape)


rule display_matrices:
    input:
        adjacency = "<outputs_network>adjacency.npz",
        distances = "<outputs_network>distances.npz",
        surfaces = "<outputs_network>surfaces.npz",
        volumes= "<outputs_network>volumes.npz",
        numerical_edge_type= "<outputs_network>edge_types.npz",
    run:
        some_arr = read_object(input.volumes).toarray()
        num_edge = read_object(input.numerical_edge_type).toarray()
        np.set_printoptions(precision=3,suppress=True,linewidth=np.inf)
        print(some_arr[0])
        print(num_edge[0])
        print()
        print(some_arr[20])
        print(num_edge[20])

rule display_energy_difference:
    input:
        adjacency = "<outputs_network>adjacency.npz",
        energy_csv=f"<pseudosimulation>energy.csv"
    run:
        np.set_printoptions(precision=3,suppress=True,linewidth=np.inf)
        adjacency = read_object(input.adjacency).tocoo() #.toarray()
        energies = read_object(input.energy_csv)
        my_energy_array = energies["Energy [kJ/mol]"].to_numpy()
        print(len(my_energy_array))
        print(my_energy_array[35], my_energy_array[43], my_energy_array[52], my_energy_array[62])
        diff_energies = my_energy_array[adjacency.row] - my_energy_array[adjacency.col]
        diff_energies = np.where(diff_energies < 10,diff_energies,10)
        assert len(adjacency.data) == len(diff_energies)
        adjacency.data = diff_energies
        print(np.where(adjacency.toarray()[44, :]))
        #print(adjacency.toarray()[8, 17])
        adjacency.data = np.exp(diff_energies)
        print(adjacency.toarray()[7, :],)
        print(adjacency.toarray()[8, 7],)
        print(adjacency.toarray()[9, 0], adjacency.toarray()[9, 10],)
        print(adjacency.toarray()[44, 35], adjacency.toarray()[44, 43], adjacency.toarray()[44, 53],)
        print(adjacency.toarray()[53, 44], adjacency.toarray()[53, 52], adjacency.toarray()[53, 62],)
        # show_array(adjacency.toarray(), "Energy difference",
        #     show=True, log=True) #save_as=output.adjacency,

rule print_eigenvector_statistics:
    input:
        eigenvectors = "<outputs_transitions>10/eigenvectors.npy",
        indices_csv=f"<outputs_network>indices_interpretation.csv",
    run:
        import plotly.graph_objects as go

        eigenvectors = read_object(input.eigenvectors)
        indices_csv = read_object(input.indices_csv)

        first_eigenvector = eigenvectors[:, 0].T
        lower_extremes, upper_extremes = auto_determine_eigenvector_extremes(first_eigenvector, N_extremes_to_plot=10)

        for el in lower_extremes:
            row_of_table = indices_csv.loc[str(int(el))]
            print(str(int(el)), int(row_of_table["Rotation index"]), np.round(float(row_of_table["Position"]), 2), np.round(float(row_of_table["Position.1"]), 2), np.round(float(row_of_table["Position.2"]), 2))

        for higher_i in range(1, 4):
            print("Eigenvector ", higher_i)
            first_eigenvector = eigenvectors[:, higher_i].T
            lower_extremes, upper_extremes = auto_determine_eigenvector_extremes(first_eigenvector,N_extremes_to_plot=10)

            print("Lower extremes: ", lower_extremes)
            for el in lower_extremes:
                row_of_table = indices_csv.loc[str(int(el))]
                print(str(int(el)),int(row_of_table["Rotation index"]),np.round(float(
                    row_of_table["Position"]),2),np.round(float(row_of_table["Position.1"]),2),np.round(float(
                    row_of_table["Position.2"]),2))

            print("Upper extremes: ", upper_extremes)
            for el in upper_extremes:
                row_of_table = indices_csv.loc[str(int(el))]
                print(str(int(el)),int(row_of_table["Rotation index"]),np.round(float(
                    row_of_table["Position"]),2),np.round(float(row_of_table["Position.1"]),2),np.round(float(
                    row_of_table["Position.2"]),2))


rule try_out_tiny_sqra:
    input:
        adjacency = "<outputs_network>adjacency.npz",
        energies = "<pseudosimulation>energy.csv",
    run:
        my_energy = read_object(input.energies)
        energies = my_energy["Binding energy [kJ/mol]"].to_numpy()
        transition_matrix = read_object(input.adjacency).astype(np.float64).toarray()

        #transition_matrix = np.array([[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0], [1,0,1,1,0, 0], [0,1,0,0,0,0]], dtype=np.float64)
        #energies = np.array([1, 5, 105000, 10, 70000, 7]) #[1, 5, 205000, 10, 70000, 7]
        diff_energies = energies[:, None] - energies[None, :]
        pi_exponent = np.round(diff_energies,14) / 100
        transition_matrix *= np.exp(pi_exponent)
        # normalize
        sums = transition_matrix.sum(axis=1)
        sums = np.array(sums).squeeze()
        diag_array = np.diag(-sums)
        transition_matrix = transition_matrix + diag_array
        np.set_printoptions(precision=3,suppress=True,linewidth=np.inf)
        #print(np.round(transition_matrix[:10,:10], 3))

        # for row in transition_matrix:
        #     if np.any(~np.isfinite(row)):
        #         mask = np.isfinite(row) & (row != 0)
        #
        #         count = np.count_nonzero(mask)
        #         # when it contains other elements they are extremely large or extremely small
        #         if count > 0:
        #             elements = row[mask]
        #             print("New one")
        #             print(count)
        #             print(elements)


        # too_large = np.where(transition_matrix.diagonal() < -1e100)[0]
        # not_finite = np.where(~np.isfinite(transition_matrix.diagonal()))[0]
        #
        # all_bad_ones = list(not_finite)
        # all_bad_ones.extend(list(too_large))
        # all_bad_ones.sort()
        # all_bad_ones = np.array(all_bad_ones)
        # print(all_bad_ones[:20])

        sparse_arr = csr_array(transition_matrix)


        reduced_rate_matrix, to_keep = delete_rows_columns(sparse_arr,"sqra")
        print("rate inf ",len(np.where(np.isinf(transition_matrix.data))[0]))
        print("reduced rate inf ",len(np.where(np.isinf(reduced_rate_matrix.data))[0]))
        #print(reduced_rate_matrix[10, :])
        #reduced_rate_matrix, to_keep = delete_rows_columns(reduced_rate_matrix,bad_ones,"sqra")

        reduced_rate_matrix = reduced_rate_matrix.toarray()
        #print(reduced_rate_matrix)
        #print(np.round(reduced_rate_matrix[:10,:10], 3))


rule try_out_sqra:
    input:
        energies = "<pseudosimulation>energy.csv",
        volumes = "<outputs_network>volumes.npy",
        distances = "<outputs_network>distances.npz",
        surfaces = "<outputs_network>surfaces.npz"
    params:
        T_in_K = 293,
        diffusion_coefficient = 1,
    run:
        my_energy = read_object(input.energies)
        my_energy_array = my_energy["Binding energy [kJ/mol]"].to_numpy()
        volumes = read_object(input.volumes)
        distances = read_object(input.distances)
        surfaces = read_object(input.surfaces)

        sqra = SQRA(energies=my_energy_array,volumes=volumes,distances=distances,surfaces=surfaces)


        rate_matrix = sqra.get_rate_matrix(params.diffusion_coefficient,params.T_in_K)

        reduced_rate_matrix, to_keep = delete_rows_columns(rate_matrix,"sqra")

        print("rate ",len(np.where(np.isinf(rate_matrix.data))[0]))
        print("reduced rate ",len(np.where(np.isinf(reduced_rate_matrix.data))[0]))

        dc = DecompositionTool(rate_matrix, np.arange(10000),10000)
        dc = DecompositionTool(reduced_rate_matrix, to_keep, 10000)
        eigenval, eigenvec = dc.decompose_sqra()
        print(eigenval)

        for evec in eigenvec.T:
            print("Eigenvector")
            print(pd.DataFrame(evec).describe())

rule align_to_xy:
    input:
        structure = "/home/hanaz63/2026_molgri/inputs/one_molecule_structures/benzene.gro"
    output:
        structure = "/home/hanaz63/2026_molgri/inputs/one_molecule_structures/benzene2.gro"
    run:
        input_u = read_object(input.structure)
        output_u = move_universe_to_xy_plane(input_u)
        write_object(output_u, output.structure)

rule look_at_network:
    input:
        network = "<outputs_network>network.pkl"
    run:
        my_network = read_object(input.network)
        print("these are volumes \n", my_network.adjacency_volume)

        # num_rotations = np.max(my_network.get_rotation_indices()) +1
        # print(num_rotations)
        #
        # all_areas = []
        # for node in my_nodes:
        #     if node.is_boundary_to_bulk():
        #         upper_radius = node.translation_node.r.hull[-1]
        #         print(upper_radius)
        #         unit_area = node.translation_node.sphere.unit_voronoi_area
        #         area = upper_radius ** 2 * unit_area / num_rotations
        #         #print(node, node.translation_node.hull[-1])
        #         print(area)
        #         all_areas.append(area)
        #
        # print(np.sum(all_areas), np.sqrt(np.sum(all_areas)/4/3.1415))
        #print(my_network.hulls)