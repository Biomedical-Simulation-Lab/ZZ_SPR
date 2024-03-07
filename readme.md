# Zienkiewicz-Zhu Superconvergent Patch Recovery

## About

This script computes the ZZ superconvergent patch recovery method for the nodal gradients in P1. 
First, the mesh is partitioned (random number due to poor partitioning scheme) and neighbours are found in parallel. This is pretty fast.
Next, the monomials (serial) and the A matrices (40 procs) for each patch are precomputed. A little slow.
Next, the nodal gradients are computed on each patch and each nodal gradient is averaged over all its containing patches. Very slow.

## TODO:
- Upgrade to P2 patches
- Look into switching over to petsc4py, which may be faster due to the availability of preconditioners. The condition number of A is very high.
- Improve the code structure (one day...)

