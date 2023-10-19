import os
import subprocess
import pytest


def test_set_up_io():
    subprocess.run("python -m molgri.scripts.set_up_io --examples", shell=True, check=True)


def test_generate_grid():
    # all these commands should pass with no error
    commands = [
        "python -m molgri.scripts.generate_grid -N 250 -d 3",
        "python -m molgri.scripts.generate_grid -N 15 -d 4",
        "python -m molgri.scripts.generate_grid -N 0 -d 3",
        "python -m molgri.scripts.generate_grid -N 0 -d 4",
        "python -m molgri.scripts.generate_grid -N 45 -d 3 --draw --animate --animate_ordering --animate_translation",
        "python -m molgri.scripts.generate_grid -N 45 -d 4 --draw --animate --animate_ordering --animate_translation",
        "python -m molgri.scripts.generate_grid -N 250 -d 3 --algorithm cube3D --recalculate --statistics"
    ]

    expected_error_commands = [
        "python -m molgri.scripts.generate_grid -N 250",
        "python -m molgri.scripts.generate_grid -N 250 -d 2",
        "python -m molgri.scripts.generate_grid -N 250 -d 5",
        "python -m molgri.scripts.generate_grid -d 2",
        "python -m molgri.scripts.generate_grid -N 250 -d 4 --algorithm cube3D",
        "python -m molgri.scripts.generate_grid -N 250 -d 3 --algorithm randomQ"
    ]

    for command in commands:
        subprocess.run(command, shell=True, check=True)

    for err_command in expected_error_commands:
        with pytest.raises(Exception):
            subprocess.run(err_command, shell=True, check=True)



def scripts_run():
    os.system("python -m molgri.scripts.generate_pt -m1 H2O -m2 NH3 -origingrid 15 -bodygrid ico_10 -transgrid 'range(1, 5, 2)' --as_dir")
    os.system("python -m molgri.scripts.generate_pt -m1 NH3 -m2 NH3 -origingrid cube3D_9 -bodygrid zero -transgrid 'range(1, 5, 2)' --extension_trajectory 'xyz' --extension_structure 'gro'")
    os.system("python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --animate")
    os.system('python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --convergence' )
    os.system(
        'python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671.xvg --p1d --p2d --p3d --convergence')
    os.system(
        'python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671.xvg --p1d --p2d --p3d')
    os.system(
        'python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --animate --Ns_o "(50, 100, 500)"')