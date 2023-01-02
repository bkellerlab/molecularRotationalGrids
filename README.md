![Coverage Status](https://img.shields.io/coverallsCoverage/github/bkellerlab/molecularRotationalGrids)
![issues](https://img.shields.io/github/issues/bkellerlab/molecularRotationalGrids)
![license](https://img.shields.io/github/license/bkellerlab/molecularRotationalGrids)
![activity](https://img.shields.io/github/last-commit/bkellerlab/molecularRotationalGrids)
[![Documentation Status](https://readthedocs.org/projects/molgri/badge/?version=main)](https://molgri.readthedocs.io/en/main/?badge=main)
![release](https://img.shields.io/github/v/release/bkellerlab/molecularRotationalGrids)

This repository is connected to the publication:
Hana Zupan, Frederick Heinz, Bettina G. Keller: "Grid-based state space exploration for molecular binding",
arXiv preprint: [https://arxiv.org/abs/2211.00566](https://arxiv.org/abs/2211.00566)

# molecularRotationalGrids

The python package ```molgri``` has three main purposes: 1) generation of rotation grids, 2) analysis of
said grids and 3) generation of pseudotrajectories (PTs). PTs are files in .xtc or similar format 
consisting of several timesteps in
which the interaction space of two molecules is systematically explored. We provide user-friendly,
one-line scripts for rotation grid generation and analysis as well as pseudotrajectory generation and 
intutive visual inspection.
In this short tutorial, we also give instructions how to use PTS with external tools like 
VMD and GROMACS for further analysis.

In the figures below, we show examples of rotation grids and pseudotrajectories as well as some analysis plots. All plots and animations are created
directly with the ```molgri``` package, except the PT plot in the middle where the output of ```molgri``` is drawn
using VMD.


<p float="left">
    <img src="/readme_images/ico_630_grid.gif" width="48%">
    <img src="/readme_images/H2O_H2O_o_ico_500_b_ico_5_t_3830884671_trajectory_energies.gif" width="48%">
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
molgri-pt -m1 H2O -m2 NH3 -o cube3D_15 -b ico_10 -t "range(1, 5, 2)"
molgri-energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --animate --convergence --Ns_o "(50, 100, 500)"
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

All flags starting with ```--``` are optional and can be omitted for faster
calculations. Remember that you can always add the flag ```--help``` to get
further instructions.

The last command ```molgri-energy``` is discussed further in section "Visualising energy distribution and convergence".

## Using outputs

The pseudotrajectory .xtc and .gro files can be used as regularly generated trajectory files. We show how they can be
displayed with VMD or used for GROMACS calculations, but the user is free to use them as inputs to any other
tool.

#### Displaying pseudotrajectory

To display the example pseudotrajectory we created in the previous section with VMD, change to
directory ```output/pt_files``` and run

```
vmd H2O_NH3_o_cube3D_15_b_ico_10_t_3203903466.gro H2O_NH3_o_cube3D_15_b_ico_10_t_3203903466.xtc
```

or on a windows computer

```
start vmd H2O_NH3_o_cube3D_15_b_ico_10_t_3203903466.gro H2O_NH3_o_cube3D_15_b_ico_10_t_3203903466.xtc
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

#### Calculating energy along a pseudotrajectory
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
<structure_file> (eg. H2O_NH3_o_cube3D_15_b_ico_10_t_3203903466.gro) and 
<trajectory_file> (eg. H2O_NH3_o_cube3D_15_b_ico_10_t_3203903466.xtc):

```
gmx22 grompp -f mdrun.mdp -c <structure_file> -p topol.top -o result.tpr   
gmx22 mdrun -s result.tpr -rerun <trajectory_file>
gmx22 energy -f ener.edr -o full_energy.xvg
```

## Visualising energy distribution and convergence

After calculating energy for each point along the pseudotrajectory (see a GROMACS example above), the
.xvg file can be copied to the ```input/``` folder and used to visualise the distribution of energies.
Visualisation can be performed with the command ```molgri-energy``` and flags
 - ```--p1d``` in 1D (violin plot), 
 - ```--p2d``` in 2D (color-coded Hammer projection) and/or 
 - ```--p3d``` in 3D (color-coded 3D plot).
In the last case we also recommend using the ```--animate``` flag so that the plot can be observed from all sides.

A XVG file is required for the visualisation and must be supplied after the ```-xvg``` flag.
It is recommended that the .xvg file has the same name as the pseudotrajectory for the full functionality.
An example provided with the package can be run like:
```
molgri-energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --animate
``` 

Additionaly, energy visualisation can be used to check for convergence / determine how many rotational 
points are truly necessary. For this purpose, it is useful to visualise energy distributions using only a fraction 
of points and visually inspecting for sufficient convergence of energy surface (see example pictured).

<p float="left">
<img src="/readme_images/hammer_energies_hammer.png">
</p>

To perform convergence tests, add the flag ```--convergence```. You can also select specific number of
points tested with a flag ```--Ns_o```, for example

```
molgri-energy -xvg H2O_H2O_o_ico_500_b_ico_5_t_3830884671 --p1d --p2d --p3d --animate --convergence --Ns_o "(50, 100, 500)"
``` 


## Complex applications: using python package

Users who would like to build custom grids, pseudotrajectories or sets of rotations and enjoy more
flexibility with visualisation tools can import ```molgri```
as a python package (following installation described above) and work with all provided modules. Documentation
of all modules is available online
[via ReadTheDocs](https://molgri.readthedocs.io/en/main/).