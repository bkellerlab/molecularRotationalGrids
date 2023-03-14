"""
The sub-package molgri.molecules is largely based on MDAnalysis handling of molecular structure and trajectory files.
However, the parsers are wrapped in own objects and passed to Pseudotrajectory generators that write out a sequence
of structures according to FullGrid prescription. Analysis of (pseudo)trajectories is possible with the module
transitions.
"""