# molecularRotationalGrids
package to generate relative rotational positions and orientations of two molecular structures

## Installation
[TODO]

## Simple applications: using scripts

For users that simply wish to create rotational grids or psudotrajectories
for their own calculations, we provide scripts with user-friendly, one-line
use. All scripts are provided for you in directory  📂 scripts/

#### Generating rotation grids
Rotation grids can be generated by specifying the number of points and
the generation algorithm (default icosahedron grid). For example, for a
45-point grid generated with a cube 3D algorithm, the command would be

```
python -m scripts.generate_grid -N 45 -a cube3D
```
User can specify additional optional flags to get additional outputs

`--statistics` to get more information about grid properties

`--draw` to save a plot of the grid

`---animate` to save a gif of the plot rotating for 360°

`---animate_ordering` to save a gif of the order in which the grid points
are plotted

#### Generating pseudotrajectories



#### Changing default input/output folders
Optionally, if one wishes to change the references to directories where input
(gro files) and output (gro, npy, pdf, gif files) are saved. To do so, run
the command

```
python -m scripts.set_up_io_directories
```

and follow the input commands.

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

A useful trick is to use the number of rotations as <step> - this displays one structure per mass point
without considering internal rotations. This number is also written in the name of the .gro file.
If you want to display all frames, you can use any large number
for <num_frames>, it does not need to correspond exactly to the number of frames.

#### Calculating energy along a pseudotrajectory



## Complex applications: using python package