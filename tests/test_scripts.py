import os


def test_scripts_run():
    os.system("python -m molgri.scripts.set_up_io --examples")
    os.system("python -m molgri.scripts.generate_grid -N 250 -algo ico --draw --animate --animate_ordering --statistics --readable")
    os.system("python -m molgri.scripts.generate_grid -N 7 -algo ico --recalculate")
    os.system("python -m molgri.scripts.generate_pt -m1 H2O -m2 NH3 -origingrid cube3D_15 -bodygrid ico_10 -transgrid 'range(1, 5, 2)' --as_dir")
    os.system("python -m molgri.scripts.generate_pt -m1 NH3 -m2 NH3 -origingrid cube3D_9 -bodygrid zero -transgrid 'range(1, 5, 2)'")