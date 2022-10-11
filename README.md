# molecularRotationalGrids
package to generate relative rotational positions and orientations of two molecular structures

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

To display the generated pseudotrajectory with VMD, run

```
vmd <path-to-pt>
```

or on a windows computer

```
start vmd <path-to-pt>
```

Then, to fully display a pseudotrajectory, it is often helpful to change the display style and to display
several or all frames at once. We suggest using the following commands within the VMD command line:

```
mol modstyle 0 0 VDW
mol drawframes 0 0 1:<step>:<num_frames>
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