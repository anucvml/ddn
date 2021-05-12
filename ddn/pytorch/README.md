# PyTorch layers

## ToDo:
### Robust pooling:
- [x] Add RobustGlobalPool2d
- [ ] Add RobustPool2d
- [ ] Add RobustAdaptivePool2d
- [ ] Multi-random-start options for non-convex functions
- [x] Add RobustVectorPool2d

### Euclidean projection:
- [x] Add EuclideanProjection

### Generic node:
- [x] Implement switch to allow full computation of B for small problems
- [ ] Implement LBFGS fall-back solver
- [ ] Handle NaN values after solves
- [x] Multiple inequality constraints with batch computation
- [ ] Filter (first-order) duplicate constraints

### Other useful nodes:
- [x] Weighted least-squares regression
- [ ] Eigendecomposition: minimize f(x, y) = tr(y^T x y), subject to h(y) = y^T y = I, [link](https://arxiv.org/pdf/1903.11240.pdf)
- [ ] SVD node
- [x] Sinkhorn node
- [ ] Limited multi-label node
- [ ] Triangulation node (Delaunay)
