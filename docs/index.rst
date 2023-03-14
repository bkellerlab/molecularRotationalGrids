.. molgri documentation master file, created by
   sphinx-quickstart on Tue Nov 29 10:56:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to molgri's documentation!
==================================

Molgri is a python package assisting you in studying approach vectors and orientations of two interacting
molecules. Some of its main functions include:
 - generating uniform grids on 3- and 4-dimensional spheres (**molgri.space**)
 - generating pseudo-trajectories by positioning molecules at automatically generated distances and orientations (**molgri.molecules**)
 - performing analysis and visualisation of this data (**molgri.plotting**)

The package can be simply used as a command-line program. Users interesting in generating sphere grids,
pseudo-trajoctories or corresponding visualistions are invited to read a quick introduction with many examples
at `our GitHub page <https://github.com/bkellerlab/molecularRotationalGrids>`_
.

Advanced users who may wish to import the package and extend it for their purposes are invited to read the
following API.

Sub-packages
~~~~~~~~~~

Interested in discretisation of 3D and 4D spheres, analysis of uniformity, generation of full rotational
and translational grids independent of application?

 -> read on in the sub-package :mod:`molgri.space`

Interested in applying the grids to molecular structure files, generating pseudo-trajectories and analysing
transitions based on the Markov state model (MSM) or Square-root approximation (SqRA)?

 -> read on in the sub-package :mod:`molgri.molecules`

Interested in visualising any of the above?

 -> read on in the sub-package :mod:`molgri.plotting`



Full API
~~~~~~~~~~


.. toctree::
   :glob:
   :maxdepth: 3

   api