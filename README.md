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
   <img src="/readme_images/set_up_30_full_color.png" width="30%">
  <img src="/readme_images/systemE_1000_uniformity.png" width="30%">
  <img src="/readme_images/systemE_1000_convergence.png" width="30%">
</p>



## Installation

Currently the project is private but it is possible to install it with the command

```
pip install git+https://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@github.com/bkellerlab/molecularRotationalGrids.git@main
```

Once released to PyPI, you should be able to install the
project using

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
molgri-pt -m1 H2O -m2 NH3 -rotgrid cube3D_15 -transgrid "range(1, 5, 2)"
```

The first-line command ```molgri-io``` creates the 📂 input/ and
📂 output/ folder structure. This command should be run in each new
directory before running other commands. The optional
flag ```--examples``` provides some sample input files that we will use later.

The second command ```molgri-grid``` is used to generate rotation grids. It is
necessary to specify the number of grid points ```-N``` and the algorithm 
```-algo``` (select from: systemE, randomE, randomQ, cube4D, cube3D, ico; we
recommend ico). Other flags describe optional figures and gifs to save. All
generated files can be found in the output/ folder.

The last command ```molgri-pt``` creates a pseudotrajectory. This scripts needs
two file inputs that should be provided in inputs/base_gro_files. Both must be
.gro files, each containing a single molecule. Due to the flag
```-m1 H2O``` the program will look for a file inputs/base_gro_files/H2O.gro
and use it as a fixed molecule in the pseudotrajectory. The flag ```-m2```
gives the name of the file with the other molecule, which will be mobile
in the simulation. Finally, the user needs to specify the rotational grid
in form ```-rotgrid algorithm_N``` and the translational grid after the
flag ```-transgrid``` in one of the
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
further instructions.

## Units

Please note that the unit system we employ here is the same as used by GROMACS 
[as described here](https://manual.gromacs.org/documentation/2019/reference-manual/definitions.html). 
Specifically, all distances should be provided as nanometers (nm).

## Simple applications: using scripts

For users that simply wish to create rotational grids or psudotrajectories
for their own calculations, we provide scripts with user-friendly, one-line
use. All scripts are provided for you in directory  📂 scripts/

#### Generating input/output folders

Having installed molgri, you are almost ready to start generating rotation grids and/or pseudotrajectories.
Move to a directory where the calculation should be performed and run the script

```
define-io
```

This will create the following folder system in your directory:

 
📂 input  
 * 📂 base_gro_files 

📂 output  
 * 📂 animations_grid_ordering  
 * 📂 animations_grids    
 * 📂 figures_grids  
 * 📂 grid_files  
 * 📂 pt_files  

When running the rest of the scripts, outputs will be saved in the output directory and inputs
can be provided in the input directory.

#### Generating rotation grids
Rotation grids can be generated with the script `gen-grid` by specifying the number of points and
the generation algorithm (default icosahedron grid). For example, for a
45-point grid generated with a cube 3D algorithm, the command would be

```
gen-grid -N 45 -a cube3D
```

To summarize, required flags are:

`-N` followed by the integer giving the number of points in a rotation grid

`-algorithm` followed by the name of one of six provided algorithms
(ico, cube3D, cube4D, randomQ, randomE, systemE)

User can specify additional optional flags to get additional outputs

`--recalculate` to overwrite the previously saved grid with identical parameters if it exists 

`--statistics` to get more information about grid properties

`--draw` to save a plot of the grid

`---animate` to save a gif of the plot rotating for 360°

`---animate_ordering` to save a gif of the order in which the grid points
are plotted

#### Generating pseudotrajectories
To generate a pseudotrajectory (PT), the user must first prepare two .gro input files in a standard
format, each containing the data for one molecule. In a pseudotrajectory, one of them will remain
fixed at origin and the other will undergo translations and rotations. The rotation and translation
grids also need to be provided. If they do not exist yet, they will be generated in that step.
To generate a pseudotrajectory where a central molecule is H2O and the rotating molecule NH3,
using a ico_50 rotation grid and a translational grid from 1 to 5 nm with step 2, run

```
gen-pt -m1 H2O -m2 NH3 -rot ico_50 -trans "(1, 5, 2)"
```
Required flags are therefore:

`-m1` followed by the name of the .gro file (without the ending) saved in input/base_gro_files
      where the molecule fixed at origin is saved

`-m1` followed by the name of the .gro file (without the ending) saved in input/base_gro_files
      where the molecule that moves in PT is saved

`-rotgrid` followed by the name of the grid written as &lt;algorithm>_&lt;num_points>, see also previous section

`-transgrid` followed by the translational grid description in format '(start, stop, num_points)'

And the optional additional flags are:

`--recalculate` to overwrite the previously saved grid with identical parameters if it exists 

`--only_origin` to use only rotations around origin and not around body for PT generation (recommended
only if the -m2 particle is a single atom or ion)



## Using outputs

The pseudotrajectory .gro files can be used as regularly generated .gro files. We show how they can be
displayed with VMD or used for GROMACS calculations, but the user is free to use them as inputs to any other
tool.

#### Displaying pseudotrajectory

To display the example pseudotrajectory we created in the previous section with VMD, run

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
mol drawframes 0 0 1:1:241
```

A useful trick is to use the number of rotations as &lt;step> - this displays one structure per mass point
without considering internal rotations. This number is also written in the name of the .gro file.
If you want to display all frames, you can use any large number
for &lt;num_frames>, it does not need to correspond exactly to the number of frames.

#### Calculating energy along a pseudotrajectory

The user must also provide a topology file which includes both molecules used in a pseudotrajectory.
We will assume that this file is named topol.top. Lastly, we need a GROMACS run file that we will name
mdrun.mdp. This file records GROMACS parameters and can be used as in normal simulations, but note that
some parameters (e.g. integrator, dt) are meaningless for a pseudotrajectory without actual dynamics.

```
gmx22 grompp -f mdrun.mdp -c <pt_file_name>.gro -p topol.top -o result.tpr   
gmx22 trjconv -f <pt_file_name>.gro -s result.tpr -o result.trr
gmx22 mdrun -s result.tpr -rerun result.trr
gmx22 energy -f ener.edr -o full_energy.xvg
```

With these commands, energy along each point of the pseudotrajectory is recorded.

## Complex applications: using python package

Users who would like to build custom grids,  pseudotrajectories or sets of rotations can import ```molgri```
as a python package (following installation described above) and work with all provided modules. Documentation
is provided in form of docstrings or available in a compiled version at our github repository 
[in docs folder](https://github.com/bkellerlab/molecularRotationalGrids/tree/main/docs/molgri).