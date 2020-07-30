# `ddn` Package

This document provides a brief description of the modules and utilities within the `ddn` package.
For an overview of deep declarative network concepts and demonstration of using the library see the
[tutorials](https://nbviewer.jupyter.org/github/anucvml/ddn/tree/master/tutorials/).

## Basic

The `ddn.basic` package contains standard python code for experimenting with deep declarative nodes. The
implementation assumes that all inputs and outputs are vectors (or more complicated data structures
have been vectorized).

* `ddn.basic.composition`: implements wrapper code for composing nodes in series or parallel (i.e., building a network).
* `ddn.basic.node`: defines the interface for data processing nodes and declarative nodes.
* `ddn.basic.robust_nodes`: implements nodes for robust pooling.
* `ddn.basic.sample_nodes`: provided examples of deep declarative nodes used for testing and in the tutorials.


## PyTorch

The `ddn.pytorch` package includes efficient implementations of deeep declarative nodes suitable for including
in an end-to-end learnable model. The code builds on the PyTorch framework and conventions.

* `ddn.geometry_utilities`: utility functions for geometry applications.
* `ddn.pytorch.node`: defines the PyTorch interface for data processing nodes and declarative nodes.
* `ddn.pytorch.pnp_node`: differentiable projection-n-point algorithm.
* `ddn.pytorch.projections`: differentiable Euclidean projection layers onto Lp balls and spheres.
* `ddn.pytorch.robostpool`: differentiable robust pooling layers.
* `ddn.pytorch.sample_nodes`: simple example implementations of deep declarative nodes for PyTorch.
