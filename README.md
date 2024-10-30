![Coverage Status](https://img.shields.io/coverallsCoverage/github/bkellerlab/molecularRotationalGrids)
![issues](https://img.shields.io/github/issues/bkellerlab/molecularRotationalGrids)
![license](https://img.shields.io/github/license/bkellerlab/molecularRotationalGrids)
![activity](https://img.shields.io/github/last-commit/bkellerlab/molecularRotationalGrids)
[![Documentation Status](https://readthedocs.org/projects/molgri/badge/?version=main)](https://molgri.readthedocs.io/en/main/?badge=main)
![release](https://img.shields.io/github/v/release/bkellerlab/molecularRotationalGrids)

# molecularRotationalGrids

The python package ```molgri``` can be used to discretize the translational and rotational space of a rigid body - the SE(3) space.
We use this discretization to systematically generate sets of two-molecule structures and thus investigate the
space and pathways of molecular association. Furthermore, Markov state models can be built based on the 
tesselation that translational and rotational grids induce and used to analyse most populated states and
slowest transitions between them.
All of this functionality is incorporated in ```molgri```.


<p float="left">
    <img src="/readme_images/ico_630_grid.gif" width="48%">
    <img src="/readme_images/H2O_H2O_o_ico_500_b_ico_5_t_3830884671_trajectory_energies.gif" width="48%">
</p>

<p float="left">
    <img src="/readme_images/molgri_demonstration.png" width="30%">
    <img src="/readme_images/set_up_30_full_color.png" width="30%">
    <img src="/readme_images/clustering_only.png" width="30%">
</p>



## Installation

```molgri``` is a python package and can be easily installed using:

```
pip install molgri
```


# Using molgri

We offer three ways of using molgri, sorted here from easiest to use to most adaptable:
 - running scripts
 - running snakemake pipelines
 - importing the package

Read on for details on each of the three user interfaces.

## Running scrips

For quick and easy generation of spherical grids and/on pseudo-trajectories with systematically positioned pairs of molecules use the provided scripts ```molgri-grid``` and ```molgri-pt```. The scripts are installed alongside the package (see Installation above).

To explore the capabilities of the package, the user is encouraged to run
the following example commands (the commands should all be executed in the
same directory, we recommend an initially empty directory).

```
molgri-io --examples
molgri-grid -N 250 -algo ico
molgri-pt -m1 H2O -m2 NH3 -o 15 -b 10 -t "range(1, 5, 2)"
```

The first-line command ```molgri-io``` creates the 📂 input/ and
📂 output/ folder structure. This command should be run in each new
directory before running other commands. The optional
flag ```--examples``` provides some sample input files that are used by the rest of the commands.

The second command ```molgri-grid``` is used to generate rotation grids. It is
necessary to specify the number of grid points ```-N``` and the algorithm 
```-algo``` (select from: systemE, randomE, randomQ, cube4D, cube3D, ico; we
recommend ico). Other flags describe optional figures and animations to save. All
generated files can be found in ```output/grid_files/```, statistics in the 
```output/statistics_files/``` and visualisations in ```output/figures/``` and ```output/animations/```. 
Note: you do not need to use this function if you are only interested in pseudotrajectories. All required
grids will be automatically generated when running the ```molgri-pt``` command.

The third command ```molgri-pt``` creates a pseudotrajectory. By default, this is a single 
trajectory-like file containing
all frames. Alternatively, with an optional command ```--as_dir``` a directory of single-frame files is
created. In addition, the first frame of the pseudo-trajectory is also written out as a structure file.
Default trajectory format is .xtc and default structure format .gro, user can change this behaviour with
optional commands ```--extension_trajectory``` and ```--extension_structure```.

This scripts needs
two file inputs that should be provided in input/, each containing a single molecule. Standard formats
like .gro, .xyz, .pdb and others are accepted. In this example, due to the flag
```-m1 H2O``` the program will look for a file input/H2O with any standard extension
and use it as the fixed molecule in the pseudotrajectory. The flag ```-m2```
gives the name of the file with the other molecule, which will be mobile
in the simulation. Finally, the user needs to specify the two rotational grids
in form ```-o algorithm_N``` (for rotations around the origin) and 
```-b algorithm_N``` , see algorithm names above. If you want to use the default algorithms (currently
icosahedron algorithm),
specify only the number of points, e.g. ```-o 15 -b 10```. Finally, the translational grid after the
flag ```-t``` should be supplied in one of the
following formats: a list of distances (in nm), linspace(start, stop, num) 
or range(start, stop, step). The argument should be surrounded by quotation
marks. Some acceptable translation grid arguments would be:

```-t "(1, 3, 5)"``` -> use distances 1nm, 3nm and 5nm

```-t "linspace(1, 3, 5)"``` -> use 5 equally spaced points
between 1nm and 3nm

```-t "range(1, 3, 0.5)"``` -> use distances between 1nm and 3nm
in 0.5nm increments

Remember that you can always add the flag ```--help``` to get
further instructions and full range of options.

The pseudotrajectory .xtc and .gro files can be used as regularly generated trajectory files. We show how they can be
displayed with VMD or used for GROMACS calculations, but the user is free to use them as inputs to any other
tool.

### Displaying pseudotrajectory

To display the example pseudotrajectory we created in the previous section with VMD, change to
directory ```output/pt_files``` and run

```
vmd structure.gro pseudotrajectory.xtc
```

or on a windows computer

```
start vmd vmd structure.gro pseudotrajectory.xtc
```

Then, to fully display a pseudotrajectory, it is often helpful to change the display style and to display
several or all frames at once. We suggest using the following commands within the VMD command line:

```
mol modstyle 0 0 VDW
mol drawframes 0 0 0:1:300
```

The first one displays the molecules as spheres with van der Waals radii and the second draws frames of
the pseudotrajectory in a form &lt;start>:&lt;step>:&lt;stop>.
A useful trick is to use the number of rotations as &lt;step> (in this case that would be 15) - this displays one structure per mass point
without considering internal rotations. This number is also written in the name of the .gro file.
If you want to display all frames, you can use any large number
for &lt;num_frames>, it does not need to correspond exactly to the number of frames.

### Calculating energy along a pseudotrajectory
Often, a pseudotrajectory is used to explore where regions of high and low energies lie when molecules
approach each other. Since a range of timesteps sampling important rotations and translations
is already provided in a PT, there is no need to run a real
simulation. Therefore, the flag ```-rerun``` is always used while dealing with PTs in GROMACS. This
setting saves time that would else be used for running an integrator and propagating positions.


To use GROMACS with PTs, the user must also provide a topology file which includes both molecules used in a pseudotrajectory.
We will assume that this file is named topol.top. Lastly, we need a GROMACS run file that we will name
mdrun.mdp. This file records GROMACS parameters and can be used as in normal simulations, but note that
some parameters (e.g. integrator, dt) are meaningless for a pseudotrajectory without actual dynamics.
Then, the energy along a pseudotrajectory can be calculated as follows, using the molgri-generated
<structure_file> (eg. structure.gro) and 
<trajectory_file> (eg. pseudotrajectory.xtc):

```
gmx22 grompp -f mdrun.mdp -c <structure_file> -p topol.top -o result.tpr   
gmx22 mdrun -s result.tpr -rerun <trajectory_file>
gmx22 energy -f ener.edr -o full_energy.xvg
```

## Using snakemake pipelines

The pre-requirements for fully functional pipelines are:
- python with correct packages (obtained through pip install molgri)
- GROMACS (the command *gmx22* should be functional in the terminal where the pipeline is run)
- VMD (the command *vmd* should be functional in the terminal where the pipeline is run)

For reproducing the entire set of computational experiments of our publications see the section Citations and reproducing papers.

To run your own computational experiment, prepare a configuration file by copying and modifying the template you can find in
```molgri/examples/default_configuration_file.yaml```. 

Three snakefiles are then prepared for you:
* *workflow/run_grid* to create a particular discretization of SE(3) space including adjacency matrices, volumes of cells etc.
* *workflow/run_sqra* to generate a grid-based pseudo-trajectory, calculate the energies along it in GROMACS, build a rate matrix and visualize its eigenvectors
* *workflow/run_msm* to generate a trajectory in GROMACS, assign the structures to closest grid cells, build a transition matrix and visualize its eigenvectors

Note: to include different molecules, the first few rules that deal with copying input files need to be adopted.

```
snakemake --snakefile workflow/run_sqra --cores 4 --configfile <configuration file>
```

### Creating records
Records are useful to have an overview of experiments that have been set-up so far and their parameters. To obtain (or update) summary files run:

```
snakemake --snakefile workflow/run_records --cores 4
```

This creates an overview .csv file in each sub-folder of experiments/

## Importing the package

Users who would like to build custom grids, pseudotrajectories, Markov state models and enjoy more
flexibility with available tools can import ```molgri```
as a python package (following installation described above) and work with all provided modules. Documentation
of all modules is available online
[via ReadTheDocs](https://molgri.readthedocs.io/en/main/).

To get a feeling for the use of the modules we recommend first reading through our workflows ```run_grid```, ```run_sqra``` and ```run_msm```.

# Citations and reproducing papers

To get additional background on the use of grids for systematic generation of bi-molecular structures and 
construction of trajectory-free Markov state models we recommend reading through our open-access paper


> Zupan, Hana, Frederick Heinz, and Bettina G. Keller. "Grid-based state space exploration for molecular binding." Canadian Journal of Chemistry (2022)

and

> [to be published (2024/2025)]

To reproduce the functionality demonstrated in the 2022 paper, please install an old version of molgri: ```pip install molgri==1.3.4```. 
While most of the old code remains, some has been updated or removed.

To use the full functionality of our current work, please install the last available version of molgri, versions
```> 2.0.0 ```. 

We believe computational experiments should be easily reproducible! To reproduce the computational experiments
featured in our latest publication (TODO: as publication is not complete yet, note that this is still somewhat in development), please install the latest version of ```molgri``` and run:

 > snakemake --cores 10

> **Notes:**
 > * in additional to internal python dependencies, two external programs must be installed and available at the command line to run this replication study: GROMACS 2022 must be available on your system as command *gmx22* and VMD as command *vmd*
 > * this process starts running 48 computational experiments - some only take seconds, but some take over a day of computational time and/or use over 10GB of memory and/or write out over 3GB of data ... so you probably want to use a computer cluster if you are running a full replication (TODO: we plan to include more accurate estimate of resources needed)
 > * if you want to only replicate a part of our work or run modified versions of experiments, use different molecules etc. the section *Using molgri* will be more useful for you
  > * feel free to contact us if you need support running the replication
  
If this repository has helped you, please cite our publications!

# Dependencies

This section is probably only useful for developers. 

Dependencies of molgri are managed with poetry. Usage:

1) install pipx:

```
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

And then open a new terminal. If there are problems, pipx reinstall-all is also useful.

2) install poetry in a separate environment with pipx: 

```
pipx install poetry
```

4) now you add/remove dependencies via poetry:

```
poetry add <dependency>
poetry remove <dependency>
```

If a dependency is only needed for a specific part of the process you please add ```--group <group_name>``` to the 
commands. Currently, there are groups *notebooks*, *test* and *workflows*.
3) To make sure no other dependencies are present in the environment, occasionally run

```
poetry install --sync
```

and to update to latest compatible versions occasionally run
```
poetry update package
```
4) Now the set-up is complete you can:
   - run a specific module, e.g. ```poetry run python -m molgri.plotting.other_plots```
   - run tests, e.g. ```poetry run pytest```
   - run workflows, e.g. ```poetry run snakemake --snakefile workflow/run_grid --configfile molgri/examples/default_configuration_file.yaml --cores 4```