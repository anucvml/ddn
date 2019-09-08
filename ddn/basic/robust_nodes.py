# ROBUST AVERAGING DEEP DECLARATIVE NODES
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
#

import autograd.numpy as np
from autograd import grad
import scipy.optimize as opt

from ddn.basic.node import *

class RobustAverage(NonUniqueDeclarativeNode):
    """
    Solves for the one-dimensional robust average,
        minimize f(x, y) = \sum_{i=1}^{n} phi(y - x_i; alpha)
    where phi(z; alpha) is one of the following robust penalties,
        'psuedo-huber': alpha^2 (\sqrt(1 + (z/alpha)^2 - 1)
        'huber':        1/2 z^2 for |z| <= alpha and alpha |z| - 1/2 alpha^2 otherwise
        'welsch':       1 - exp(-z^2 / 2 alpha^2)
        'trunc-quad':   1/2 z^2 for |z| <= alpha and 1/2 alpha^2 otherwise
    """
    def __init__(self, n, penalty='huber', alpha=1.0):
        super().__init__(n, 1)
        self.alpha = alpha
        self.alpha_sq = alpha ** 2
        self.penalty = penalty.lower()
        if (self.penalty == 'pseudo-huber'):
            self.phi = lambda z: self.alpha_sq * (np.sqrt(1.0 + np.power(z, 2.0) / self.alpha_sq) - 1.0)
        elif (self.penalty == 'huber'):
            self.phi = lambda z: np.where(np.abs(z) <= alpha, 0.5 * np.power(z, 2.0), alpha * np.abs(z) - 0.5 * self.alpha_sq)
        elif (self.penalty == 'welsch'):
            self.phi = lambda z: 1.0 - np.exp(-0.5 * np.power(z, 2.0) / self.alpha_sq)
        elif (self.penalty == 'trunc-quad'):
            self.phi = lambda z: np.minimum(0.5 * np.power(z, 2.0), 0.5 * self.alpha_sq)
        else:
            assert False, "unrecognized penalty function {}".format(penalty)

    def objective(self, x, y):
        assert (len(x) == self.dim_x) and (len(y) == self.dim_y)
        return np.sum([self.phi(y - xi) for xi in x])

    def solve(self, x):
        assert(len(x) == self.dim_x)

        J = lambda y : self.objective(x, y)
        dJ = lambda y : self.fY(x, y)

        # TODO: ransac for welsch and trunc-quad
        result = opt.minimize(J, np.mean(x), args=(), method='L-BFGS-B', jac=dJ,
                              options={'maxiter': 100, 'disp': False, 'gtol': self.eps})
        if not result.success: print(result.message)
        return result.x, None

    def exact_gradient(self, x, y=None):
        """Computes the analytic gradient of the optimal solution for testing."""
        if y is None:
            y, _ = self.solve(x)

        if (self.penalty == 'pseudo-huber'):
            dy = np.array([np.power(1.0 + np.power(y - xi, 2.0) / self.alpha_sq, -1.5) for xi in x])
        elif (self.penalty == 'huber') or (self.penalty == 'trunc-quad'):
            dy = np.array([1.0 if np.abs(y - xi) <= self.alpha else 0.0 for xi in x])
        elif (self.penalty == 'welsch'):
            z = np.power(x - y, 2.0)
            dy = np.array([(self.alpha_sq - zi) / (self.alpha_sq * self.alpha_sq) * np.exp(-0.5 * zi / self.alpha_sq) for xi in x])

        return dy.reshape((1, self.dim_x)) / np.sum(dy)
