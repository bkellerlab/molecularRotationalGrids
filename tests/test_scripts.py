import subprocess
import os


def test_scripts_run():
    pass
    # cwd = os.getcwd()
    # os.chdir("..")
    # #subprocess.call(['python', 'molgri/scripts/set_up_io.py', "--examples"])
    # #subprocess.run(["python", "-m molgri.scripts.set_up_io --examples"])
    # print(subprocess.Popen(["molgri/scripts/set_up_io.py", "--examples"], shell=True).communicate())
    # print(subprocess.Popen(["python", "-m ..molgri.scripts.generate_grid -N 250 -algo ico --draw --animate --animate_ordering --statistics"], shell=True).communicate())
    # #subprocess.run("python -m .molgri.scripts.generate_grid -N 250 -algo ico --draw --animate --animate_ordering --statistics")
    # #subprocess.run("python -m .molgri.scripts.generate_pt -m1 H2O -m2 NH3 -origingrid cube3D_15 -bodygrid ico_10 -transgrid 'range(1, 5, 2)'")
    # os.chdir(cwd)