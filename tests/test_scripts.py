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

def test_generate_pt():
    # all these commands should pass with no error
    commands = [
        'python -m molgri.scripts.generate_pt -m1 NH3 -m2 CL -o ico_15 -b 10 -t "range(1, 5, 2)" --as_dir',
        'python -m molgri.scripts.generate_pt -m1 NA -m2 NH3 -o 15 -b randomQ_10 -t "range(1, 5, 2)"',
        'python -m molgri.scripts.generate_pt -m1 H2O -m2 NH3 -o 5 -b zero -t "[1, 5, 10]" --recalculate',
        'python -m molgri.scripts.generate_pt -m1 H2O -m2 H2O -o 1 -b 1 -t "linspace(0.1, 2, 5)" --extension_trajectory xyz --extension_structure xyz'
    ]

    expected_error_commands = [
        'python -m molgri.scripts.generate_pt -m2 NH3 -o ico_15 -b 10 -t "range(1, 5, 2)"',
        'python -m molgri.scripts.generate_pt -m1 H2O -m2 NH3 -o 15 -b zero_10 -t "range(1, 5, 2)"',
        'python -m molgri.scripts.generate_pt -m1 H2O -m2 NH3 -o 5 -b randomS_1 -t "[1, 5, 10]"',
        'python -m molgri.scripts.generate_pt -m1 H2O -m2 H2O -o 1 -b 1 -t "[-7]"'
    ]

    for command in commands:
        subprocess.run(command, shell=True, check=True)

    for err_command in expected_error_commands:
        with pytest.raises(Exception):
            print(err_command)
            subprocess.run(err_command, shell=True, check=True)

def test_generate_energy():

    commands = [
        'python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --animate',
        'python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --convergence',
        'python -m molgri.scripts.generate_energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --Ns_o "(50, 100, 500)" --convergence'
    ]

    for command in commands:
        subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    test_set_up_io()
    test_generate_grid()
    test_generate_pt()
    test_generate_energy()