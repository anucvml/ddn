# `ddn` Package

This document provides a brief description of the modules and utilities within the `ddn` package.
For an overview of deep declarative network concepts and demonstration of using the library see the
comprehensive [tutorials](https://nbviewer.jupyter.org/github/anucvml/ddn/tree/master/tutorials/).

## Basic

The `ddn.basic` package contains standard python code for experimenting with deep declarative nodes. The
implementation assumes that all inputs and outputs are vectors (or more complicated data structures
have been vectorized).

* `ddn.basic.composition`: implements wrapper code for composing nodes in series or parallel (i.e., building a network).
* `ddn.basic.node`: defines the interface for data processing nodes and declarative nodes (see [tutorial](https://nbviewer.jupyter.org/github/anucvml/ddn/blob/master/tutorials/05_ddn_basic_node.ipynb)).
* `ddn.basic.robust_nodes`: implements nodes for robust pooling.
* `ddn.basic.sample_nodes`: provided examples of deep declarative nodes used for testing and in the tutorials.


## PyTorch

The `ddn.pytorch` package includes efficient implementations of deeep declarative nodes suitable for including
in an end-to-end learnable model. The code builds on the PyTorch framework and conventions.

* `ddn.pytorch.geometry_utilities`: utility functions for geometry applications.
* `ddn.pytorch.leastsquares`: differentiable weighted least-squares nodes (see [tutorial](https://nbviewer.jupyter.org/github/anucvml/ddn/blob/master/tutorials/10_least_squares.ipynb)).
* `ddn.pytorch.node`: defines the PyTorch interface for data processing nodes and declarative nodes (see [tutorial](https://nbviewer.jupyter.org/github/anucvml/ddn/blob/master/tutorials/08_ddn_pytorch_node.ipynb)).
* `ddn.pytorch.pnp_node`: differentiable projection-n-point algorithm.
* `ddn.pytorch.optimal_transport`: differentiable entropy regularized optimal transport layer (see [tutorial](https://nbviewer.jupyter.org/github/anucvml/ddn/blob/master/tutorials/11_optimal_transport.ipynb)).
* `ddn.pytorch.projections`: differentiable Euclidean projection layers onto Lp balls and spheres.
* `ddn.pytorch.robustpool`: differentiable robust pooling layers treating each dimension independently.
* `ddn.pytorch.robust_vec_pool`: differentiable robust vector pooling layers (see [tutorial](https://nbviewer.jupyter.org/github/anucvml/ddn/blob/master/tutorials/09_robust_vector_pooling.ipynb)).
* `ddn.pytorch.sample_nodes`: simple example implementations of deep declarative nodes for PyTorch.
