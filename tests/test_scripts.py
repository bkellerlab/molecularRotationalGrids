import os


def test_scripts_run():
    os.system("python -m molgri.scripts.set_up_io --examples")
    os.system("python -m molgri.scripts.generate_grid -N 250 -algo ico --draw --animate --animate_ordering --statistics --readable")
    os.system("python -m molgri.scripts.generate_grid -N 7 -algo ico --recalculate")
    os.system("python -m molgri.scripts.generate_pt -m1 H2O -m2 NH3 -origingrid 15 -bodygrid ico_10 -transgrid 'range(1, 5, 2)' --as_dir")
    os.system("python -m molgri.scripts.generate_pt -m1 NH3 -m2 NH3 -origingrid cube3D_9 -bodygrid zero -transgrid 'range(1, 5, 2)' --extension_trajectory 'xyz' --extension_topology 'gro'")
    os.system("python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --animate")
    os.system('python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --animate --convergence' )
    os.system(
        'python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --animate --convergence --Ns_o "(50, 100, 500)"')