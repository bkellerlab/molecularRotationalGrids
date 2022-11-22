![Coverage Status](https://img.shields.io/coverallsCoverage/github/bkellerlab/molecularRotationalGrids)
![issues](https://img.shields.io/github/issues/bkellerlab/molecularRotationalGrids)
![license](https://img.shields.io/github/license/bkellerlab/molecularRotationalGrids)
![activity](https://img.shields.io/github/last-commit/bkellerlab/molecularRotationalGrids)
![pypi](https://img.shields.io/pypi/format/molgri)
![release](https://img.shields.io/github/v/release/bkellerlab/molecularRotationalGrids)

This repository is connected to the publication:
Hana Zupan, Frederick Heinz, Bettina G. Keller: "Grid-based state space exploration for molecular binding",
arXiv preprint: [https://arxiv.org/abs/2211.00566](https://arxiv.org/abs/2211.00566)

# molecularRotationalGrids

The python package ```molgri``` has three main purposes: 1) generation of rotation grids, 2) analysis of
said grids and 3) generation of pseudotrajectories (PTs). PTs are .gro files with several timesteps in
which the interaction space of two molecules is systematically explored. We provide user-friendly,
one-line scripts for rotation grid generation and analysis as well as pseudotrajectory generation and
give instructions how to use PTS with external tools like VMD and GROMACS for further analysis.

Below, we show examples of rotation grids created with different algorithms and figures of
a protein-ion pseudotrajectory as well as some analysis plots. All plots and animations are created
directly with the ```molgri``` package, except the PT plot where the output of ```molgri``` is drawn
using VMD.


<p float="left">
    <img src="/readme_images/ico_630_grid.gif" width="48%">
    <img src="/readme_images/systemE_1000_grid.gif" width="48%">
</p>

<p float="left">
    <img src="/readme_images/systemE_1000_uniformity.png" width="30%">
    <img src="/readme_images/set_up_30_full_color.png" width="30%">
    <img src="/readme_images/systemE_1000_convergence.png" width="30%">
</p>



## Installation

This is a python library that can be easily installed using:

```
pip install molgri
```



## Running examples

To explore the capabilities of the package, the user is encouraged to run
the following example commands (the commands should all be executed in the
same directory, we recommend an initially empty directory).

```
molgri-io --examples
molgri-grid -N 250 -algo ico --draw --animate --animate_ordering --statistics
molgri-pt -m1 H2O -m2 NH3 -origingrid cube3D_15 -bodygrid ico_10 -transgrid "range(1, 5, 2)"
```

****The first-line command ```molgri-io``` creates the 📂 input/ and
📂 output/ folder structure. This command should be run in each new
directory before running other commands. The optional
flag ```--examples``` provides some sample input files that we will use later.

The second command ```molgri-grid``` is used to generate rotation grids. It is
necessary to specify the number of grid points ```-N``` and the algorithm 
```-algo``` (select from: systemE, randomE, randomQ, cube4D, cube3D, ico; we
recommend ico). Other flags describe optional figures and animations to save. All
generated files can be found in the output/ folder.

The last command ```molgri-pt``` creates a pseudotrajectory. As a default, this is a single file with
all frames, with an optional command ```--as_dir``` a directory of single-frame .gro files is
created. This scripts needs
two file inputs that should be provided in input/. Both must be
.gro files, each containing a single molecule. Due to the flag
```-m1 H2O``` the program will look for a file input/H2O.gro
and use it as a fixed molecule in the pseudotrajectory. The flag ```-m2```
gives the name of the file with the other molecule, which will be mobile
in the simulation. Finally, the user needs to specify the two rotational grids
in form ```-origingrid algorithm_N``` (for rotations around the origin) and 
```-bodygrid algorithm_N``` , see algorithm names above. Finally, the translational grid after the
flag ```-transgrid``` should be supplied in one of the
following formats: a list of distances (in nm), linspace(start, stop, num) 
or range(start, stop, step). The argument should be surrounded by quotation
marks. Some acceptable translation grid arguments would be:

```-transgrid "(1, 3, 5)"``` -> use distances 1nm, 3nm and 5nm

```-transgrid "linspace(1, 3, 5)"``` -> use 5 equally spaced points
between 1nm and 3nm

```-transgrid "range(1, 3, 0.5)"``` -> use distances between 1nm and 3nm
in 0.5nm increments

All flags starting with ```--``` are optional and can be omitted for faster
calculations. Remember that you can always add the flag ```--help``` to get
further instructions.****


## Using outputs

The pseudotrajectory .gro files can be used as regularly generated .gro files. We show how they can be
displayed with VMD or used for GROMACS calculations, but the user is free to use them as inputs to any other
tool.

#### Displaying pseudotrajectory

To display the example pseudotrajectory we created in the previous section with VMD, stay in the same
directory and run

```
vmd output/pt_files/H2O_NH3_cube3D_15_full.gro
```

or on a windows computer

```
start vmd output/pt_files/H2O_NH3_cube3D_15_full.gro
```

Then, to fully display a pseudotrajectory, it is often helpful to change the display style and to display
several or all frames at once. We suggest using the following commands within the VMD command line:

```
mol modstyle 0 0 VDW
mol drawframes 0 0 0:1:241
```

The first one displays the molecules as spheres with van der Waals radii and the second draws frames of
the pseudotrajectory in a form &lt;start>:&lt;step>:&lt;stop>.
A useful trick is to use the number of rotations as &lt;step> (in this case that would be 15) - this displays one structure per mass point
without considering internal rotations. This number is also written in the name of the .gro file.
If you want to display all frames, you can use any large number
for &lt;num_frames>, it does not need to correspond exactly to the number of frames.

#### Calculating energy along a pseudotrajectory
Often, a pseudotrajectory is used to explore where reagions of high and low energies lie when molecules
approach each other. Since a range of timesteps sampling important rotations and translations
is already provided in a PT, there is no need to run a real
simulation. Therefore, the flag ```-rerun``` is always used while dealing with PTs in GROMACS. This
setting saves time that would else be used for running an integrator and propagating positions.


To use GROMACS with PTs, the user must also provide a topology file which includes both molecules used in a pseudotrajectory.
We will assume that this file is named topol.top. Lastly, we need a GROMACS run file that we will name
mdrun.mdp. This file records GROMACS parameters and can be used as in normal simulations, but note that
some parameters (e.g. integrator, dt) are meaningless for a pseudotrajectory without actual dynamics.
Then, the energy along a pseudotrajectory can be calculated as follows:

```
gmx22 grompp -f mdrun.mdp -c H2O_NH3_cube3D_15_full.gro -p topol.top -o result.tpr   
gmx22 trjconv -f H2O_NH3_cube3D_15_full.gro -s result.tpr -o result.trr
gmx22 mdrun -s result.tpr -rerun result.trr
gmx22 energy -f ener.edr -o full_energy.xvg
```


## Complex applications: using python package

Users who would like to build custom grids,  pseudotrajectories or sets of rotations can import ```molgri```
as a python package (following installation described above) and work with all provided modules. Documentation
is provided in form of docstrings or available in a compiled version at our github repository 
[in docs folder](https://github.com/bkellerlab/molecularRotationalGrids/tree/main/docs/molgri).