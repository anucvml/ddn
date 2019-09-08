# ROBUST AVERAGING DEEP DECLARATIVE NODES
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
#

import autograd.numpy as np
from autograd import grad
import scipy.optimize as opt

from ddn.basic.node import *

class HuberRobustAverage(NonUniqueDeclarativeNode):
    """
    Solves for the one-dimensional robust average using the Huber penalty,
        minimize f(x, y) = \sum_{i=1}^{n} phi(y - x_i)
    where phi is the Huber function
        phi(z) = 1/2 z^2 for |z| <= 1 and |z| - 1/2 otherwise
    """
    def __init__(self, n):
        super().__init__(n, 1)
        self.phi = lambda z: np.where(np.abs(z) <= 1.0, 0.5 * np.power(z, 2.0), np.abs(z) - 0.5)

    def objective(self, x, y):
        assert (len(x) == self.dim_x) and (len(y) == self.dim_y)
        return np.sum([self.phi(y - xi) for xi in x])

    def solve(self, x):
        x = x.copy()
        assert(len(x) == self.dim_x)

        J = lambda y : self.objective(x, y)
        fY = grad(self.objective, 1)
        dJ = lambda y : fY(x, y)

        result = opt.minimize(J, np.mean(x), args=(), method='L-BFGS-B', jac=dJ, options={'maxiter': 100, 'disp': False})
        return result.x, None

    def exact_gradient(self, x, y=None):
        if y is None:
            y, _ = self.solve(x)
        dy = np.array([1.0 if np.abs(y - xi) <= 1.0 else 0.0 for xi in x])
        return dy.reshape((1, self.dim_x)) / np.sum(dy)
