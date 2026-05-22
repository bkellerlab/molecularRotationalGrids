import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from molgri.images.create_vmdlog import VMDCreator
from molgri.images.modifying_images import join_images
from molgri.images.plotting import show_array
from molgri.molecules.rate_merger import delete_rows_columns
from molgri.molecules.transitions import SQRA, auto_determine_eigenvector_extremes
from workflow.helpers.find_right_input import find_the_right_frames, find_the_right_structure, what_to_provide
from workflow.helpers.io import get_num_atoms, read_object, write_object
from molgri.molecules.transitions import DecompositionTool


def input_sqra(wc):
    base_properties = {"energies": "<pseudosimulation>energy.csv",
            "volumes": "<outputs_network>volumes.npz",
            "distances": "<outputs_network>distances.npz",
            "surfaces": "<outputs_network>surfaces.npz"}
    if config["sqra"]["allow_diffusion_to_bulk"]:
        base_properties["surfaces_to_bulk"] = f"<outputs_network>boundaries_to_bulk.npy"
        base_properties["volumes_to_bulk"] = f"<outputs_network>volumes_to_bulk.npy"
    return base_properties

rule make_sqra:
    input:
        unpack(input_sqra)
    output:
        rate_matrix = f"<outputs_transitions>sqra/sqra.npz",
    params:
        T_in_K = 293,
        diffusion_coefficient = config["sqra"]["diffusion_coefficient"],
        capping_factor = config["sqra"]["capping_factor"],
        flow_to_bulk = config["sqra"]["flow_to_bulk"],
        allow_diffusion_to_bulk = config["sqra"]["allow_diffusion_to_bulk"],
        energy_kJ_mol_bulk = config["sqra"]["energy_kJ_mol_bulk"],
    run:
        my_energy = read_object(input.energies)
        my_energy_array = my_energy["Energy [kJ/mol]"].to_numpy()
        volumes = read_object(input.volumes)
        distances = read_object(input.distances)
        surfaces = read_object(input.surfaces)
        flow_to_bulk = float(params.flow_to_bulk)
        energy_kJ_mol_bulk = float(params.energy_kJ_mol_bulk)

        sqra = SQRA(energies=my_energy_array,volumes=volumes,distances=distances,surfaces=surfaces, T=params.T_in_K)
        rate_matrix = sqra.get_rate_matrix(params.diffusion_coefficient,
            capping_factor=params.capping_factor)
        print("pre ", pd.DataFrame(rate_matrix.data).describe())
        if bool(params.allow_diffusion_to_bulk):
            surfaces_to_bulk = read_object(input.surfaces_to_bulk)
            volumes_to_bulk = read_object(input.volumes_to_bulk)
            rate_matrix = sqra.add_bulk(flow_to_bulk, surfaces_to_bulk, energy_kJ_mol_bulk, volumes_to_bulk)
            print("post ", pd.DataFrame(rate_matrix.data).describe())
        write_object(rate_matrix, output.rate_matrix)

rule reduce_sqra_size:
    """
    Here we remove rows & indices with too high energies to enable diagonalization of rate matrix.
    """
    input:
        sqra= f"<outputs_transitions>sqra/sqra.npz",
    output:
        reduced_sqra= f"<outputs_transitions>sqra/reduced_sqra.npz",
        indices_to_keep = f"<outputs_transitions>sqra/indices_to_keep.npy",
    params:
        cutting_factor = config["sqra"]["cutting_factor"],
        tolerance_rate_matrix_row_sum = config["sqra"]["tolerance_rate_matrix_row_sum"]
    run:
        sqra = read_object(input.sqra)

        # if we don't have any idea how to set the cutting factor we can just try out a few options
        if params.cutting_factor == "auto":
            logarithmically_spaced_cutting_factors = np.logspace(2, 300, num=298)
            logarithmically_spaced_cutting_factors = logarithmically_spaced_cutting_factors [::-1]

            for cutting_factor in logarithmically_spaced_cutting_factors:
                reduced_sqra, indices_to_keep = delete_rows_columns(sqra,"sqra", cutting_factor)
                # we demand the row-sum of rate matrix to be close to zero in order to get eigenvectors that behave as such
                if np.abs(np.max(np.sum(reduced_sqra, axis=1))) < float(params.tolerance_rate_matrix_row_sum):
                    print("FOUND CUTTING FACTOR ", cutting_factor)
                    break
            else:
                raise ValueError(f"No cutting factor could reach the required tolerance: {np.abs(np.max(np.sum(reduced_sqra, axis=1)))} > {float(params.tolerance_rate_matrix_row_sum)}")
        else:
            print("CUTTING FACTOR is set to be ", params.cutting_factor)
            reduced_sqra, indices_to_keep = delete_rows_columns(sqra,"sqra", float(params.cutting_factor))

        print("Written matrix has shape ", reduced_sqra.shape)
        write_object(reduced_sqra, output.reduced_sqra)
        write_object(np.array(indices_to_keep),output.indices_to_keep)

rule run_decomposition_sqra:
    """
    As output we want to have eigenvalues, eigenvectors. Es input we get a (sparse) rate matrix.
    """
    input:
        reduced_sqra= f"<outputs_transitions>sqra/reduced_sqra.npz",
        indices_to_keep = f"<outputs_transitions>sqra/indices_to_keep.npy",
        grid_info= "<outputs_network>grid_info.yaml",
    benchmark:
        f"<outputs_transitions>sqra/timing_decomposition.txt"
    output:
        eigenvalues_sum_0 = f"<outputs_transitions>sqra/eigenvalues_sum_0.npy",
        eigenvectors_sum_0 = f"<outputs_transitions>sqra/eigenvectors_sum_0.npy",
        eigenvalues_sum_other= f"<outputs_transitions>sqra/eigenvalues_sum_other.npy",
        eigenvectors_sum_other= f"<outputs_transitions>sqra/eigenvectors_sum_other.npy",
    params:
        tolerance = config["sqra"]["tolerance_eigendecomposition"],
        sigma_sqra = config["sqra"]["sigma"],
        allow_diffusion_to_bulk= config["sqra"]["allow_diffusion_to_bulk"]
    run:
        grid_info = read_object(input.grid_info)
        total_length = int(grid_info["N_total"])
        if params.allow_diffusion_to_bulk:
            total_length += 1
        print("total_length", total_length)
        kept_indices = read_object(input.indices_to_keep)
        my_matrix = read_object(input.reduced_sqra)
        print("Read out matrx has shape ", my_matrix.shape)
        dt = DecompositionTool(my_matrix, kept_indices, total_length)
        sum_to_0, sum_to_other = dt.decompose_sqra(sigma=float(params.sigma_sqra), tolerance=float(params.tolerance))
        write_object(sum_to_0[0], output.eigenvalues_sum_0)
        write_object(sum_to_0[1],output.eigenvectors_sum_0)
        write_object(sum_to_other[0], output.eigenvalues_sum_other)
        write_object(sum_to_other[1],output.eigenvectors_sum_other)

rule plot_sqra_eigenvectors_as_lines:
    input:
        eigenvectors_sum_0 = f"<outputs_transitions>sqra/eigenvectors_sum_0.npy",
        eigenvectors_sum_other= f"<outputs_transitions>sqra/eigenvectors_sum_other.npy",
    output:
        plot = f"<outputs_other_plots>eigenvectors_sqra.png"
    params:
        N_interesting_eigenvectors = config["eigenvectors"]["num_interesting_eigenvectors_as_lines"]
    run:

        eigenvector_array_0 = read_object(input.eigenvectors_sum_0)
        eigenvector_array_other = read_object(input.eigenvectors_sum_other)
        all_eigenvector_arrays = (eigenvector_array_0, eigenvector_array_other)
        print("line plot ", eigenvector_array_0.shape, eigenvector_array_other.shape)
        print("last el ", np.max(eigenvector_array_0.T[0]), np.min(eigenvector_array_0.T[0]), np.max(eigenvector_array_other.T[0]), eigenvector_array_0.T[0][-1], eigenvector_array_other.T[0][-1])

        N_interesting_eigenvectors = params.N_interesting_eigenvectors

        fig = make_subplots(rows=N_interesting_eigenvectors,cols=2, column_titles=["Sum=0", "Sum=Other"])

        for col in range(2):
            selected_array = all_eigenvector_arrays[col]
            for row in range(min(N_interesting_eigenvectors, selected_array.shape[1])):
                data_eigenvector = selected_array[:, row]
                selected_x = np.where(~np.isclose(data_eigenvector,0, rtol=1e-3, atol=1e-5))[0]
                print(len(selected_x))
                selected_y = data_eigenvector[selected_x]


                xs = np.empty(3 * len(selected_x))
                ys = np.empty(3 * len(selected_y))

                xs[0::3] = selected_x
                xs[1::3] = selected_x
                xs[2::3] = np.nan

                ys[0::3] = 0
                ys[1::3] = selected_y
                ys[2::3] = np.nan


                fig.add_trace(
                    go.Scattergl(x=xs,y=ys, line=dict(color="black"),
                        mode="lines"),row=1+row,col=1+col)
                fig.update_xaxes(range=[0, len(data_eigenvector)], showticklabels=False, ticks="",col=1 + col,row=1+row)
                # hline - for all values that are zero
                fig.add_hline(y=0,line=dict(color="black",width=1),opacity=1,col=1 + col,row=1+row)
            # todo names of peaks
        # fig.add_hline(
        #     y=1,
        #     line_color="gray",
        #     line_width=1,
        #     line_dash="dot"
        # )
        # fig.add_hline(
        #     y=-1,
        #     line_color="gray",
        #     line_width=1,
        #     line_dash="dash"
        # )
        fig.update_layout(showlegend=False,plot_bgcolor="white", autosize=False,
    width=800,
    height=1000,)
        fig.update_yaxes(showticklabels=False, ticks="", range=[-0.4, 0.4]) #range=[-1, 1],
        #fig.update_xaxes(showticklabels=False,ticks="")
        fig.write_image(output.plot, scale=3)

rule display_rate_matrix:
    input:
        rate_matrix = f"<outputs_transitions>sqra/sqra.npz",
        reduced_sqra= f"<outputs_transitions>sqra/reduced_sqra.npz",
        indices_to_keep= f"<outputs_transitions>sqra/indices_to_keep.npy",
    output:
        plot = f"<outputs_other_plots>array_sqra.png",
        plot_reduced = f"<outputs_other_plots>array_reduced_sqra.png",
    run:
        as_array = read_object(input.rate_matrix).toarray()
        show_array(as_array, "Rate Matrix",
            save_as=output.plot, show=False)
        for i, row in enumerate(as_array):
            if i in [7, 8, 9, 43, 44, 53]:
                print(i, np.unique(row))
        print("---- REDUCED ---")
        as_array = read_object(input.reduced_sqra).toarray()
        show_array(as_array,title="Reduced Rate Matrix",
            save_as=output.plot_reduced,show=False, indices=read_object(input.indices_to_keep))
        for i, row in enumerate(as_array):
            if i in [7, 8, 9, 43, 44, 53]:
                print(i, np.unique(row))

rule plot_sqra_eigenvalues:
    input:
        eigenvalues_sum_0 = f"<outputs_transitions>sqra/eigenvalues_sum_0.npy",
        # eigenvalues_sum_1= f"<outputs_transitions>sqra/eigenvalues_sum_1.npy",
        eigenvalues_sum_other= f"<outputs_transitions>sqra/eigenvalues_sum_other.npy",
    output:
        plot = f"<outputs_other_plots>eigenvalues_sqra.png"
    params:
        N_interesting_eigenvectors = config["eigenvectors"]["num_interesting_eigenvectors"]
    run:
        eigenvals_0 = read_object(input.eigenvalues_sum_0)
        # eigenvals_1 = read_object(input.eigenvalues_sum_1)
        eigenvals_other = read_object(input.eigenvalues_sum_other)
        all_eigenvalues = (eigenvals_0, eigenvals_other)

        fig = make_subplots(rows=1,cols=2,column_titles=["Sum=0", "Sum=Other"],
            horizontal_spacing=0.03, vertical_spacing=0.01)

        max_num = int(params.N_interesting_eigenvectors)

        xs = np.linspace(0, 1, num=max_num)

        for col in range(2):
            eigenvalue_array = all_eigenvalues[col]
            actual_max_num = min(max_num, len(eigenvalue_array))
            # vertical lines
            for i, eigenw in enumerate(eigenvalue_array[:actual_max_num]):
                fig.add_shape(type="line", x0=xs[i], y0=0, x1=xs[i], y1=eigenw, line=dict(color="black", width=5),
                              opacity=1, col=1+col, row=1)

            # horizontal infinite line
            fig.add_hline(y=0, line=dict(color="black", width=5), opacity=1, col=1+col, row=1)

            # plotting (after the axes so that the numbers are on top if there is any overlap)
            fig.add_scatter(x=xs, y=eigenvalue_array[:actual_max_num], mode='markers+text', text=[f"{el:.1e}" for el in eigenvalue_array[:actual_max_num]],
                            marker=dict(size=14, color="black"), opacity=1, col=1+col, row=1)
        fig.update_layout(margin=dict(l=100, r=40, t=30, b=70))


        fig.update_layout(xaxis_visible=False, xaxis_showticklabels=False, font=dict(size=12), yaxis_visible=False, showlegend=False)

        fig.update_traces(textposition='bottom center', textfont=dict(size=12))

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        fig.update_layout(
            width=1500,
            height=300,
            font = dict(size=18)
        )
        fig.update_xaxes(showticklabels=False,ticks="")
        fig.update_yaxes(showticklabels=False,ticks="")

        fig.write_image(output.plot, scale=3)

rule save_sqra_its:
    input:
        eigenvalues_sum_0 = f"<outputs_transitions>sqra/eigenvalues_sum_0.npy",
    output:
        its = f"<outputs_transitions>sqra/its.txt"
    run:
        eigenvals = read_object(input.eigenvalues_sum_0)
        its = np.array([-1 / (eigenval) for eigenval in eigenvals])
        for i, it in enumerate(its):
            print(f"{i}-th ITS: {it} ns")
        write_object(its, output.its)

rule plot_sqra_its:
    input:
        eigenvalues_sum_0 = f"<outputs_transitions>sqra/eigenvalues_sum_0.npy",
    output:
        plot = f"<outputs_other_plots>its_sqra.png"
    run:
        eigenvals = read_object(input.eigenvalues_sum_0)
        its = [-1 / (eigenval) for eigenval in eigenvals]

        fig = go.Figure()

        for it in its[:4]:
            fig.add_hline(it,line=dict(color="black",dash="dash",width=1),opacity=1)
        fig.update_layout(xaxis_title=r"",yaxis_title="ITS [ns]",xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, np.max(its) + 0.2]))

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=18)
        )
        fig.update_layout(
            xaxis=dict(
                showline=True,# show axis spine
                linecolor="black",
                ticks="",# hide ticks
                showticklabels=False,# hide tick labels
                title=None  # no axis label
            ),
            yaxis=dict(
                showline=True,
                linecolor="black",
            ),
            plot_bgcolor="white"
        )

        fig.write_image(output.plot, scale=3)

checkpoint sqra_find_indices_dominant_eigenvectors:
    """
    For each eigenvector find the structures that contribute the most to the eigenvector.
    """
    input:
        # eigenvectors_sum_1=f"<outputs_transitions>sqra/eigenvectors_sum_1.npy",
        eigenvectors_sum_0=f"<outputs_transitions>sqra/eigenvectors_sum_0.npy",
        eigenvectors_sum_other= f"<outputs_transitions>sqra/eigenvectors_sum_other.npy",
    output:
        # abs_e_indices=expand(f"<outputs_indices>sqra/{{i}}_eigenvector_sum_1_{{j}}_largest_abs_values.txt",
        #     j=config["eigenvectors"]["num_extremes_to_plot"], i=range(config["eigenvectors"]["num_interesting_eigenvectors"])),
        pos_e_indices= expand(f"<outputs_indices>sqra/{{i}}_eigenvector_sum_0_{{j}}_most_positive.txt",
            i=range(config["eigenvectors"]["num_interesting_eigenvectors"]),j=config["eigenvectors"]["num_extremes_to_plot"]),
        neg_e_indices= expand(f"<outputs_indices>sqra/{{i}}_eigenvector_sum_0_{{j}}_most_negative.txt",
            i=range(config["eigenvectors"]["num_interesting_eigenvectors"]),j=config["eigenvectors"]["num_extremes_to_plot"],),
        abs_e_indices_other=expand(f"<outputs_indices>sqra/{{i}}_eigenvector_sum_other_{{j}}_largest_abs_values.txt",
            j=config["eigenvectors"]["num_extremes_to_plot"],i=range(
                config["eigenvectors"]["num_interesting_eigenvectors"])),
    params:
        N_interesting_eigenvectors=config["eigenvectors"]["num_interesting_eigenvectors"],
        N_extremes_to_plot=config["eigenvectors"]["num_extremes_to_plot"]
    run:
        # eigenvectors = read_object(input.eigenvectors_sum_1)
        N_interesting_eigenvectors = int(params.N_interesting_eigenvectors)
        N_extremes_to_plot = params.N_extremes_to_plot

        # eigenvectors with sum 1 should have only positive or only negative contributions
        # for i in range(N_interesting_eigenvectors):
        #     if i < len(eigenvectors.T):
        #         eigenvector = eigenvectors.T[i]
        #         pos_e, pos_neg = auto_determine_eigenvector_extremes(np.abs(eigenvector),N_extremes_to_plot)
        #         # save the absolute
        #         write_object(np.array(pos_e),output.abs_e_indices[i])
        #     else:
        #         write_object(np.array([]),output.abs_e_indices[i])

        eigenvectors = read_object(input.eigenvectors_sum_0)
        for i in range(N_interesting_eigenvectors):
            if i < len(eigenvectors.T):
                eigenvector = eigenvectors.T[i]
                pos_e, neg_e = auto_determine_eigenvector_extremes(eigenvector,N_extremes_to_plot)
                # save the absolute
                write_object(np.array(pos_e),output.pos_e_indices[i])
                write_object(np.array(neg_e),output.neg_e_indices[i])
            else:
                write_object(np.array([]),output.pos_e_indices[i])
                write_object(np.array([]),output.neg_e_indices[i])

        eigenvectors = read_object(input.eigenvectors_sum_other)

        # eigenvectors with sum 1 should have only positive or only negative contributions
        for i in range(N_interesting_eigenvectors):
            if i < len(eigenvectors.T):
                eigenvector = eigenvectors.T[i]
                pos_e, pos_neg = auto_determine_eigenvector_extremes(np.abs(eigenvector),N_extremes_to_plot)
                print(i, pos_e)
                # save the absolute
                write_object(np.array(pos_e),output.abs_e_indices_other[i])
            else:
                write_object(np.array([]),output.abs_e_indices_other[i])

def collect_box_information(input):
    grid_info = read_object(input.grid_info)
    subgrid_limits = grid_info["subgrid_limits_A"]
    N_rotations = int(grid_info["N_rotations"])
    my_grid = read_object(input.grid)[:, :3]
    my_grid = my_grid[::N_rotations]

    box_limits = [subgrid_limits[0][1], subgrid_limits[1][1], subgrid_limits[2][1]]
    return box_limits, my_grid

def input_base(where, what, wc):
    structure_path = find_the_right_structure(what)
    return {"structure": structure_path,
            "structure1": "<pseudosimulation>molecule1.gro",
            "grid_info": "<outputs_network>grid_info.yaml",
            "translation_rotation_script": f"<inputs_vmd>script{wc.view_index}.log",
            "grid": "<outputs_network>grid.npy"}

################# EVERYTHING FOR EIGENVECTORS WITH SUM 1

rule get_eigenvectors_sum_1:
    input:
        expand(f"<outputs_molecular_plots>eigenvectors/sqra/ALL_EIGENVECTORS_SUM_1_view{{view_index}}_{{COM_or_full}}.png",
        COM_or_full=["full"], view_index=config["analysis"]["view_index"])

rule stack_all_eigenvectors_sum_1:
    input:
        expand(f"<outputs_molecular_plots>eigenvectors/sqra/individual/{{i}}th_eigenvector_sum_1_view{{view_index}}_{{COM_or_full}}.png",
            i=list(range(config["eigenvectors"]["num_interesting_eigenvectors"])), allow_missing=True),
    output:
        all_eigenvectors_one_tau = f"<outputs_molecular_plots>eigenvectors/sqra/ALL_EIGENVECTORS_SUM_1_view{{view_index}}_{{COM_or_full}}.png"
    run:
        join_images(input,output.all_eigenvectors_one_tau,flip=False)

def input_zeroth_eigenvector(wc):
    where = "<pseudosimulation>"
    what = what_to_provide(wc.COM_or_full, for_a_structure=True)
    result = input_base(where, what, wc)

    indices_file = checkpoints.sqra_find_indices_dominant_eigenvectors.get(i=wc.i).output.abs_e_indices
    indices_file = indices_file[int(wc.i)]
    indices = read_object(indices_file).astype(int)
    what = what_to_provide(wc.COM_or_full,for_a_structure=False)
    result["all_frame_gros"] = find_the_right_frames(where, what, indices, grid_len=NUM_GRID_POINTS)
    return result

rule eigenvector_sum_1_overlapping_frames:
    input:
        unpack(input_zeroth_eigenvector)
    output:
        vmdlog=f"<outputs_vmd>eigenvectors/sqra/{{i}}th_eigenvector_sum_1_view{{view_index}}_{{COM_or_full}}",
        frame_plot=f"<outputs_molecular_plots>eigenvectors/sqra/individual/{{i}}th_eigenvector_sum_1_view{{view_index}}_{{COM_or_full}}.tga",
        frame_plot_png = f"<outputs_molecular_plots>eigenvectors/sqra/individual/{{i}}th_eigenvector_sum_1_view{{view_index}}_{{COM_or_full}}.png"
    params:
        zoom_level = config["analysis"]["zoom_level"],
        draw_m1 = config["analysis"]["plot_m1_as"],
        draw_m2 = config["analysis"]["plot_m2_as"],
    run:
        n1 = get_num_atoms(input.structure1)
        box_limits, gridpoints = collect_box_information(input)

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")


        my_vmd.prepare_frame_script(vmd_name=output.vmdlog, plot_name=output.frame_plot,
            num_frames=len(input.all_frame_gros),
            box_limits=box_limits, draw_m1=params.draw_m1, draw_m2=params.draw_m2,
            draw_rectangular_box=False, gridpoints=None,
            zoom_level=int(params.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_all_frames = ' '.join(input.all_frame_gros)

        shell("vmd  -dispdev text {input.structure} {names_all_frames} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")

################# EVERYTHING FOR EIGENVECTORS WITH SUM other
#TODO
rule get_eigenvectors_sum_other:
    input:
        expand(f"<outputs_molecular_plots>eigenvectors/sqra/ALL_EIGENVECTORS_SUM_other_view{{view_index}}_{{COM_or_full}}.png",
        COM_or_full=["full"], view_index=config["analysis"]["view_index"])

rule stack_all_eigenvectors_sum_other:
    input:
        expand(f"<outputs_molecular_plots>eigenvectors/sqra/individual/{{i}}th_eigenvector_sum_other_view{{view_index}}_{{COM_or_full}}.png",
            i=list(range(config["eigenvectors"]["num_interesting_eigenvectors"])), allow_missing=True),
    output:
        all_eigenvectors_one_tau = f"<outputs_molecular_plots>eigenvectors/sqra/ALL_EIGENVECTORS_SUM_other_view{{view_index}}_{{COM_or_full}}.png"
    run:
        join_images(input,output.all_eigenvectors_one_tau,flip=False)

def input_sum_other_eigenvector(wc):
    where = "<pseudosimulation>"
    what = what_to_provide(wc.COM_or_full, for_a_structure=True)
    result = input_base(where, what, wc)

    indices_file = checkpoints.sqra_find_indices_dominant_eigenvectors.get(i=wc.i).output.abs_e_indices_other
    indices_file = indices_file[int(wc.i)]
    indices = read_object(indices_file).astype(int)
    what = what_to_provide(wc.COM_or_full,for_a_structure=False)
    result["all_frame_gros"] = find_the_right_frames(where, what, indices, grid_len=NUM_GRID_POINTS)
    return result

rule eigenvector_sum_other_overlapping_frames:
    input:
        unpack(input_sum_other_eigenvector)
    output:
        vmdlog=f"<outputs_vmd>eigenvectors/sqra/{{i}}th_eigenvector_sum_other_view{{view_index}}_{{COM_or_full}}",
        frame_plot=f"<outputs_molecular_plots>eigenvectors/sqra/individual/{{i}}th_eigenvector_sum_other_view{{view_index}}_{{COM_or_full}}.tga",
        frame_plot_png = f"<outputs_molecular_plots>eigenvectors/sqra/individual/{{i}}th_eigenvector_sum_other_view{{view_index}}_{{COM_or_full}}.png"
    params:
        zoom_level = config["analysis"]["zoom_level"],
        draw_m1= config["analysis"]["plot_m1_as"],
        draw_m2= config["analysis"]["plot_m2_as"],
    run:
        n1 = get_num_atoms(input.structure1)
        box_limits, gridpoints = collect_box_information(input)

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")


        my_vmd.prepare_frame_script(vmd_name=output.vmdlog, plot_name=output.frame_plot,
            num_frames=len(input.all_frame_gros),
            box_limits=box_limits, draw_m1=params.draw_m1, draw_m2=params.draw_m2,
            draw_rectangular_box=False, gridpoints=None,
            zoom_level=int(params.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_all_frames = ' '.join(input.all_frame_gros)

        shell("vmd  -dispdev text {input.structure} {names_all_frames} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")

################# EVERYTHING FOR EIGENVECTORS WITH SUM 0

rule get_eigenvectors_sum_0:
    input:
        expand(f"<outputs_molecular_plots>eigenvectors/sqra/ALL_EIGENVECTORS_SUM_0_view{{view_index}}_{{COM_or_full}}.png",
        COM_or_full=["full"], view_index=config["analysis"]["view_index"])

rule stack_all_eigenvectors_sum_0:
    input:
        expand(f"<outputs_molecular_plots>eigenvectors/sqra/individual/{{i}}th_eigenvector_sum_0_view{{view_index}}_{{COM_or_full}}.png",
            i=list(range(config["eigenvectors"]["num_interesting_eigenvectors"])), allow_missing=True),
    output:
        all_eigenvectors_one_tau = f"<outputs_molecular_plots>eigenvectors/sqra/ALL_EIGENVECTORS_SUM_0_view{{view_index}}_{{COM_or_full}}.png"
    run:
        join_images(input,output.all_eigenvectors_one_tau,flip=False)


def input_eigenvector_sum_0(wc):
    where = "<pseudosimulation>"
    what = what_to_provide(wc.COM_or_full, for_a_structure=True)
    result = input_base(where, what, wc)

    indices_pos_file = checkpoints.sqra_find_indices_dominant_eigenvectors.get(i=wc.i).output.pos_e_indices
    indices_neg_file = checkpoints.sqra_find_indices_dominant_eigenvectors.get(i=wc.i).output.neg_e_indices

    indices_pos = read_object(indices_pos_file[int(wc.i)]).astype(int)
    indices_neg = read_object(indices_neg_file[int(wc.i)]).astype(int)

    what = what_to_provide(wc.COM_or_full,for_a_structure=False)
    result["pos_e_structures"] = find_the_right_frames(where, what, indices_pos, NUM_GRID_POINTS)
    result["neg_e_structures"] = find_the_right_frames(where, what, indices_neg, NUM_GRID_POINTS)
    return result

rule eigenvector_sum_0_overlapping_frames:
    input:
        unpack(input_eigenvector_sum_0)
    output:
        vmdlog=f"<outputs_vmd>eigenvectors/sqra/{{i}}th_eigenvector_sum_0_view{{view_index}}_{{COM_or_full}}",
        frame_plot=f"<outputs_molecular_plots>eigenvectors/sqra/individual/{{i}}th_eigenvector_sum_0_view{{view_index}}_{{COM_or_full}}.tga",
        frame_plot_png = f"<outputs_molecular_plots>eigenvectors/sqra/individual/{{i}}th_eigenvector_sum_0_view{{view_index}}_{{COM_or_full}}.png"
    params:
        zoom_level = config["analysis"]["zoom_level"],
        draw_m1= config["analysis"]["plot_m1_as"],
        draw_m2= config["analysis"]["plot_m2_as"],
    run:
        from molgri.images.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms, read_object

        n1 = get_num_atoms(input.structure1)
        box_limits, gridpoints = collect_box_information(input)


        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")

        my_vmd.prepare_eigenvector_script(num_red=len(input.pos_e_structures), num_blue=len(input.neg_e_structures),
            vmd_name=output.vmdlog, plot_name=output.frame_plot, draw_m1=params.draw_m1, draw_m2=params.draw_m2,
            box_limits=box_limits, draw_rectangular_box=False, gridpoints=None,
            zoom_level=int(params.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_red = ' '.join(input.pos_e_structures)
        names_blue = ' '.join(input.neg_e_structures)
        shell("vmd  -dispdev text {input.structure} {names_red} {names_blue} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")