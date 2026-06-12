"""
The sub-package molgri.space is only concerned with geometry in 3- and 4-dimensional spaces. It includes general
functionality and no references to molecular structures.

Most of the modules are intended for internal use. The module that is the interface between grid implementation and
pseudo-trajectory implementation is the fullgrid module.

Modules:
 - analysis: implements statistical tests of uniformity in 3D and 4D spherical grids
 - **fullgrid: combines position grid with orientation grid and thus fully discretises approach space in Voronoi cells**
 - polytopes: implements 3D and 4D polytopes: cube, hypercube and icosahedron - needed for rotobj module
 - rotations: implements transformation between grid objects and rotation objects
 - **rotobj: implements SphereGridNDim object using varius algorithms to distribute points on N-dim sphere**
 - translations: parses user inputs of translations (linear space discretisation)
 - utils: provides other useful functions, eg normalisation, random quaternions
"""