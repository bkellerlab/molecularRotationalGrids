import numpy as np
from plotly.tools import DEFAULT_PLOTLY_COLORS

from molgri.molecules.rate_merger import delete_rows_columns
from molgri.molecules.transitions import MSM
from workflow.helpers.find_right_input import what_to_provide
from workflow.helpers.io import read_object, write_object, read_from_mdrun


rule make_msm:
    input:
        assignments = f"<outputs_assignment>full_assignment.npy",
        grid_info= "<outputs_network>grid_info.yaml",
    output:
        msm= f"<outputs_transitions>{{tau}}/msm.npz",
    run:
        full_assignments = read_object(input.assignments)
        grid_info = read_object(input.grid_info)
        N_gridpoints = grid_info["N_total"]

        my_msm = MSM(full_assignments, N_gridpoints)
        transition_matrix = my_msm.get_one_tau_transition_matrix(int(wildcards.tau), noncorrelated_windows=False)
        write_object(transition_matrix, output.msm)

rule reduce_msm_size:
    input:
        msm= f"<outputs_transitions>{{tau}}/msm.npz",
    wildcard_constraints:
        tau= r"[1-9]\d*"
    output:
        reduced_msm= f"<outputs_transitions>{{tau}}/reduced_msm.npz",
        indices_to_keep = f"<outputs_transitions>{{tau}}/indices_to_keep.npy",
    run:
        msm = read_object(input.msm)
        reduced_msm, indices_to_keep = delete_rows_columns(msm,"msm")
        write_object(reduced_msm, output.reduced_msm)
        write_object(np.array(indices_to_keep),output.indices_to_keep)


rule run_decomposition_msm:
    """
    As output we want to have eigenvalues, eigenvectors. Es input we get a (sparse) rate matrix.
    """
    input:
        reduced_msm=f"<outputs_transitions>{{tau}}/reduced_msm.npz",
        indices_to_keep=f"<outputs_transitions>{{tau}}/indices_to_keep.npy",
        grid_info= "<outputs_network>grid_info.yaml",
    wildcard_constraints:
        tau= r"[1-9]\d*"
    benchmark:
        f"<outputs_transitions>{{tau}}/timing_decomposition.txt"
    output:
        eigenvalues = f"<outputs_transitions>{{tau}}/eigenvalues.npy",
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy"
    run:
        from molgri.molecules.transitions import DecompositionTool

        grid_info = read_object(input.grid_info)
        total_length = int(grid_info["N_total"])
        kept_indices = read_object(input.indices_to_keep)
        my_matrix = read_object(input.reduced_msm)

        # calculation
        dt = DecompositionTool(my_matrix, kept_indices, total_length)
        all_eigenval, all_eigenvec = dt.decompose_msm()

        write_object(all_eigenval, output.eigenvalues)
        write_object(all_eigenvec,output.eigenvectors)



TAUS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200]

rule get_implied_timescales:
    input:
        eigenvalues = f"<outputs_transitions>{{tau}}/eigenvalues.npy",
        runfile=f"<simulation>production.mdp"
    output:
        its = f"<outputs_transitions>{{tau}}/its.npy",
    params:
        N_eigenvec = config["eigenvectors"]["num_interesting_eigenvectors"]
    run:
        eigenvalues = read_object(input.eigenvalues)
        eigenvalues = eigenvalues[1:]  # dropping the first one as it should be zero and cause issues

        writeout = int(read_from_mdrun(input.runfile,"nstxout-compressed"))
        time_step_ps = float(read_from_mdrun(input.runfile,"dt"))

        num_interesting_timescales = int(params.N_eigenvec)

        # save only the interesting ones
        while len(eigenvalues) < num_interesting_timescales:
            eigenvalues.append(np.nan)
        if len(eigenvalues) > 4:
            eigenvalues = eigenvalues[:4]

        its = - int(wildcards.tau) * writeout * time_step_ps / np.log(np.abs(eigenvalues))

        write_object(its, output.its)



rule plot_all_eigenvectors_as_lines:
    input:
        expand(f"<outputs_other_plots>grouped_by_zcoo_eigenvectors_for_tau_{{tau}}.png", tau=[10])

rule plot_vmd_eigenvectors_as_lines_grouped_by_zcoo:
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
        grid_info= rules.save_basic_grid_information.output.info_material
    output:
        plot = f"<outputs_other_plots>grouped_by_zcoo_eigenvectors_for_tau_{{tau}}.png"
    params:
        N_eigenvec = config["eigenvectors"]["num_interesting_eigenvectors"]
    run:
        import plotly.graph_objects as go
        import numpy as np
        from plotly.subplots import make_subplots

        grid_info = read_object(input.grid_info)
        N_rotations = grid_info["N_rotations"]
        N_translations = grid_info["N_translations"]
        subgrids = grid_info["subgrid_points"]
        len_x, len_y, len_z = len(subgrids[0]), len(subgrids[1]), len(subgrids[2])

        eigenvector_array = read_object(input.eigenvectors)
        N_interesting_eigenvectors = int(params.N_eigenvec)

        groups_by_rotation_index = np.repeat(np.arange(N_translations), N_rotations)
        groups_by_rotation_index = groups_by_rotation_index % len_z
        print(groups_by_rotation_index)
        print(groups_by_rotation_index[70:90])
        print(groups_by_rotation_index[1990:2010])

        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for row in range(N_interesting_eigenvectors):
            eigenvector = eigenvector_array[:, row]
            out = np.bincount(groups_by_rotation_index ,weights=eigenvector,minlength=len_z)
            fig.add_trace(
                go.Bar(x=np.arange(len_z),y=out, text=[f"{i:>2}" for i in np.arange(len_z)]),row=1+row,col=1)
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        #fig.update_yaxes(range=[-5, 5])
        #fig.update_yaxes(range=[-5, 0], row=1, col=1)
        fig.update_xaxes(showticklabels=False)
        fig.write_image(output.plot, scale=3)

rule plot_vmd_eigenvectors_as_lines_grouped_by_trans:
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
        grid_info= rules.save_basic_grid_information.output.info_material
    output:
        plot = f"<outputs_other_plots>grouped_by_trans_eigenvectors_for_tau_{{tau}}.png"
    params:
        N_eigenvec = config["eigenvectors"]["num_interesting_eigenvectors"]
    run:
        import plotly.graph_objects as go
        import numpy as np
        from plotly.subplots import make_subplots

        grid_info = read_object(input.grid_info)
        N_rotations = grid_info["N_rotations"]
        N_translations = grid_info["N_translations"]
        subgrids = grid_info["subgrid_points"]
        len_x, len_y, len_z = len(subgrids[0]), len(subgrids[1]), len(subgrids[2])

        eigenvector_array = read_object(input.eigenvectors)
        N_interesting_eigenvectors = int(params.N_eigenvec)

        groups_by_rotation_index = np.repeat(np.arange(N_translations), N_rotations)
        print(groups_by_rotation_index)
        print(groups_by_rotation_index[70:90])

        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for row in range(N_interesting_eigenvectors):
            eigenvector = eigenvector_array[:, row]
            out = np.bincount(groups_by_rotation_index ,weights=eigenvector,minlength=N_translations)
            fig.add_trace(
                go.Bar(x=np.arange(N_translations),y=out, text=[f"{i:>2}" for i in np.arange(N_translations)]),row=1+row,col=1)
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        #fig.update_yaxes(range=[-5, 5])
        #fig.update_yaxes(range=[-5, 0], row=1, col=1)
        fig.update_xaxes(showticklabels=False)
        fig.write_image(output.plot, scale=3)

rule plot_vmd_eigenvectors_as_lines_grouped_by_rotation:
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
        grid_info= rules.save_basic_grid_information.output.info_material
    output:
        plot = f"<outputs_other_plots>grouped_by_rotation_eigenvectors_for_tau_{{tau}}.png"
    params:
        N_eigenvec = config["eigenvectors"]["num_interesting_eigenvectors"]
    run:
        import plotly.graph_objects as go
        import numpy as np
        from plotly.subplots import make_subplots

        grid_info = read_object(input.grid_info)
        N_rotations = grid_info["N_rotations"]
        N_translations = grid_info["N_translations"]


        eigenvector_array = read_object(input.eigenvectors)
        N_interesting_eigenvectors = int(params.N_eigenvec)

        groups_by_rotation_index = np.tile(np.arange(N_rotations), N_translations)


        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for row in range(N_interesting_eigenvectors):
            eigenvector = eigenvector_array[:, row]
            out = np.bincount(groups_by_rotation_index ,weights=eigenvector,minlength=N_rotations)
            fig.add_trace(
                go.Bar(x=np.arange(N_rotations),y=out, text=[f"{i:>2}" for i in np.arange(N_rotations)]),row=1+row,col=1)
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        fig.update_yaxes(range=[-5, 5])
        fig.update_yaxes(range=[-5, 0], row=1, col=1)
        fig.update_xaxes(showticklabels=False)
        fig.write_image(output.plot, scale=3)


# rule plot_vmd_eigenvectors_as_lines:
#     input:
#         eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
#     output:
#         plot = f"<outputs_other_plots>eigenvectors_for_tau_{{tau}}.png"
#     params:
#         N_eigenvec = config["eigenvectors"]["num_interesting_eigenvectors"]
#     run:
#         import plotly.graph_objects as go
#         from plotly.subplots import make_subplots
#
#         eigenvector_array = read_object(input.eigenvectors)
#
#         N_interesting_eigenvectors = int(params.N_eigenvec)
#
#         fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)
#
#         for row in range(N_interesting_eigenvectors):
#             fig.add_trace(
#                 go.Scatter(x=np.arange(eigenvector_array.shape[0]),y=eigenvector_array[:, row], line=dict(color="black"),
#                     mode="lines"),row=1+row,col=1)
#
#         fig.update_layout(showlegend=False, plot_bgcolor="white",)
#         fig.write_image(output.plot, scale=3)

rule plot_msm_eigenvectors_as_lines:
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
    output:
        plot = f"<outputs_other_plots>eigenvectors_for_tau_{{tau}}.png"
    params:
        N_interesting_eigenvectors = config["eigenvectors"]["num_interesting_eigenvectors_as_lines"]
    run:
        eigenvector_array = read_object(input.eigenvectors)

        N_interesting_eigenvectors = params.N_interesting_eigenvectors

        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for col in range(1):
            selected_array = eigenvector_array
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
        #     y=0.5,
        #     line_color="gray",
        #     line_width=1,
        #     line_dash="dot"
        # )
        # fig.add_hline(
        #     y=-0.5,
        #     line_color="gray",
        #     line_width=1,
        #     line_dash="dot"
        # )
        fig.update_layout(showlegend=False,plot_bgcolor="white", autosize=False,
    width=800,
    height=500,)
        fig.update_yaxes(showticklabels=False, ticks="", range=[-0.4, 0.4]) #range=[-1, 1],
        #fig.update_xaxes(showticklabels=False,ticks="")
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=18)
        )
        fig.write_image(output.plot, scale=3)

rule run_plot_its_msm:
    input:
        its = expand(f"<outputs_transitions>{{tau}}/its.npy", tau=config["msm"]["taus"]),
        runfile=f"<simulation>production.mdp"
    output:
        plot_its = f"<outputs_other_plots>its.png"
    run:
        from plotly.subplots import make_subplots

        writeout = int(read_from_mdrun(input.runfile,"nstxout-compressed"))
        time_step_ps = float(read_from_mdrun(input.runfile,"dt"))

        xs = np.array(config["msm"]["taus"]) * writeout * time_step_ps
        all_its = np.array([read_object(its_file) for its_file in input.its])


        fig = make_subplots(1, 2, shared_yaxes=False)
        for col in (1, 2):
            # gray triangle
            fig.add_scatter(x=[0, xs[-1], xs[-1]], y=[0, 0, xs[-1]], mode="lines", fill="toself", fillcolor="gray",
                                 line=dict(width=0), row=1, col=col)
            fig.update_layout(showlegend=False, xaxis_title=r"$\tau [ps]$", yaxis_title=r"ITS [ps]")
            fig.update_xaxes(title_text=r"$\tau [ps]$", row=1, col=col)
            fig.update_yaxes(title_text=r"ITS [ps]", row=1, col=col)
            # eigenvalues
            cols = DEFAULT_PLOTLY_COLORS

            for i, its in enumerate(all_its.T):
                if col==2:
                    xs = xs[:9]
                    its = its[:9]
                    fig.update_xaxes(range=[0, np.max(xs)], row=1, col=col)
                    fig.update_yaxes(range=[0, 20],row=1,col=col)
                fig.add_scatter(x=xs, y=its, mode="lines+markers", line=dict(width=2, color=cols[i]), row=1,
                                     col=col)
        fig.update_layout(
            xaxis=dict(
                showline=True,# show axis spine
                linecolor="black",
            ),
            yaxis=dict(
                showline=True,
                linecolor="black",
            ),
            plot_bgcolor="white",
            width=800, height=500
        )
        fig.write_image(output.plot_its, scale=3)



# def input_eigenvector_msm(wc):
#     where = "<pseudosimulation>"
#     what = what_to_provide(wc.COM_or_full, for_a_structure=True)
#     result = input_base(where, what, wc)
#
#     indices_pos_file = checkpoints.find_indices_dominant_eigenvectors.get(i=wc.ic, tau=wc.tau).output.pos_e_indices
#     indices_neg_file = checkpoints.find_indices_dominant_eigenvectors.get(i=wc.i, tau=wc.tau).output.neg_e_indices
#
#     indices_pos = read_object(indices_pos_file[int(wc.i)]).astype(int)
#     indices_neg = read_object(indices_neg_file[int(wc.i)]).astype(int)
#
#     what = what_to_provide(wc.COM_or_full,for_a_structure=False)
#     result["pos_e_structures"] = find_the_right_frames(where, what, indices_pos, NUM_GRID_POINTS)
#     result["neg_e_structures"] = find_the_right_frames(where, what, indices_neg, NUM_GRID_POINTS)
#     return result
#
# rule eigenvector_sum_0_overlapping_frames:
#     input:
#         unpack(input_eigenvector_msm)
#     output:
#         vmdlog=f"<outputs_vmd>eigenvectors/{{tau}}/{{i}}th_eigenvector_view{{view_index}}_{{COM_or_full}}",
#         frame_plot=f"<outputs_molecular_plots>eigenvectors/{{tau}}/individual/{{i}}th_eigenvector_sum_0_view{{view_index}}_{{COM_or_full}}.tga",
#         frame_plot_png = f"<outputs_molecular_plots>eigenvectors/{{tau}}/individual/{{i}}th_eigenvector_sum_0_view{{view_index}}_{{COM_or_full}}.png"
#     params:
#         zoom_level = config["analysis"]["zoom_level"],
#         draw_m1= config["analysis"]["plot_m1_as"],
#         draw_m2= config["analysis"]["plot_m2_as"],
#     run:
#         from molgri.images.create_vmdlog import VMDCreator
#         from workflow.helpers.io import get_num_atoms, read_object
#
#         n1 = get_num_atoms(input.structure1)
#         box_limits, gridpoints = collect_box_information(input)
#
#
#         my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")
#
#         my_vmd.prepare_eigenvector_script(num_red=len(input.pos_e_structures), num_blue=len(input.neg_e_structures),
#             vmd_name=output.vmdlog, plot_name=output.frame_plot, draw_m1=params.draw_m1, draw_m2=params.draw_m2,
#             box_limits=box_limits, draw_rectangular_box=False, gridpoints=None,
#             zoom_level=int(params.zoom_level), translation_rotation_script=input.translation_rotation_script)
#
#         names_red = ' '.join(input.pos_e_structures)
#         names_blue = ' '.join(input.neg_e_structures)
#         shell("vmd  -dispdev text {input.structure} {names_red} {names_blue} < {output.vmdlog}")
#         shell("convert {output.frame_plot} {output.frame_plot_png}")