# SAMPLE DEEP DECLARATIVE NODES
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
#

from ddn.basic.node import *

class SquaredErrorNode(AbstractNode):
    """Computes the squared difference between the input and a given target vector."""
    def __init__(self, n, x_target=None):
        super().__init__(n, 1)
        self.x_target = np.zeros((n,)) if x_target is None else x_target

    def solve(self, x):
        return 0.5 * np.sum(np.square(x - self.x_target)), None

    def gradient(self, x, y=None, ctx=None):
        return (x - self.x_target).T


class UnconstPolynomial(AbstractDeclarativeNode):
    """Solves min. f(x, y) = xy^4 + 2x^2y^3 - 12y^2  from Gould et al., 2016. Takes smallest x over the three
    stationary points."""
    def __init__(self):
        super().__init__(1, 1)
        
    def objective(self, x, y):
        return (x[0] * y[0] ** 2.0 + 2 * x[0] ** 2.0 * y[0] - 12) * y[0] ** 2.0

    def solve(self, x):
        delta = np.sqrt(9.0 * x[0] ** 4.0 + 96.0 * x[0])
        y_stationary = [0.0, (-3.0 * x[0] ** 2.0 - delta) / (4.0 * x[0]), (-3.0 * x[0] ** 2.0 + delta) / (4.0 * x[0])]
        y_min_indx = np.argmin([self.objective(x, [y]) for y in y_stationary])
        return np.array([y_stationary[y_min_indx]]), None

    def gradient(self, x, y=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        if y is None:
            y, ctx = self.solve(x)
        return np.array([-1.0 * (y ** 3 + 3.0 * x[0] * y ** 2) / (3.0 * x[0] * y ** 2 + 3.0 * x[0] ** 2 * y - 6.0)])


class LinFcnOnUnitCircle(EqConstDeclarativeNode):
    """
    Solves the problem
        minimize   f(x, y) = (1, x)^Ty
        subject to h(y) = \|y\|^2 = 1
    for 1d input (x) and 2d output (y).
    """
    def __init__(self):
        super().__init__(1, 2)

    def objective(self, x, y):
        return y[0] + y[1] * x[0]

    def constraint(self, x, y):
        return np.dot(y, y) - 1.0

    def solve(self, x):
        y_star = -1.0 / np.sqrt(1.0 + x[0]**2.0) * np.array([1.0, x[0]])
        ctx = {'nu': -0.5 * np.sqrt(1 + x[0]**2.0)}
        return y_star, ctx

    def gradient(self, x, y=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        return 1.0 / np.power(1 + x[0]**2.0, 1.5) * np.array([x[0], -1.0])


class ConstLinFcnOnParameterizedCircle(EqConstDeclarativeNode):
    """
    Solves the problem
        minimize   f(x, y) = (1, 1)^Ty
        subject to h(y) = \|y\|^2 = x^2
    for 1d input (x) and 2d output (y).
    """
    def __init__(self):
        super().__init__(1, 2)

    def objective(self, x, y):
        return y[0] + y[1]

    def constraint(self, x, y):
        return np.dot(y, y) - x[0]**2.0

    def solve(self, x):
        y_star = -1.0 * np.fabs(x[0]) / np.sqrt(2.0) * np.ones((2,))
        ctx = {'nu': 0.0 if x[0] == 0.0 else 0.5 / y_star[0]}
        return y_star, ctx

    def gradient(self, x, y=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        return -1.0 * np.sign(x[0]) / np.sqrt(2.0) * np.ones((2,))


class LinFcnOnParameterizedCircle(EqConstDeclarativeNode):
    """
    Solves the problem
        minimize   f(x, y) = (1, x_1)^Ty
        subject to h(y) = \|y\|^2 = x_2^2
    for 2d input (x) and 2d output (y).
    """
    def __init__(self):
        super().__init__(2, 2)

    def objective(self, x, y):
        return y[0] + x[0] * y[1]

    def constraint(self, x, y):
        return np.dot(y, y) - x[1]**2.0

    def solve(self, x):
        y_star = -1.0 * np.fabs(x[1]) / np.sqrt(1.0 + x[0]**2.0) * np.array([1.0, x[0]])
        ctx = {'nu': 0.0 if x[1] == 0.0 else 0.5 / y_star[0]}
        return y_star, ctx

    def gradient(self, x, y=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        return np.vstack( (np.abs(x[1]) / np.power(1 + x[0]**2.0, 1.5) * np.array([x[0], -1.0]),
             -1.0 * np.sign(x[1]) / np.sqrt(1.0 + x[0]**2.0) * np.array([1.0, x[0]])) ).T


class QuadFcnOnSphere(EqConstDeclarativeNode):
    """
    Solves the problem
        minimize   f(x, y) = 0.5 * y^Ty - x^T y
        subject to h(y) = \|y\|^2 = 1
    """

    def __init__(self, n=2, m=2):
        super().__init__(n, m)

    def objective(self, x, y):
        return 0.5 * np.dot(y, y) - np.dot(y, x)

    def constraint(self, x, y):
        return np.dot(y, y) - 1.0

    def solve(self, x):
        y_star = 1.0 / np.sqrt(np.dot(x, x)) * x
        return y_star, None

    def gradient(self, x, y=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        return 1.0 / np.power(np.dot(x, x), 1.5) * (np.dot(x, x) * np.eye(self.dim_x) - np.outer(x, x))


class QuadFcnOnBall(IneqConstDeclarativeNode):
    """
    Solves the (inequality constrained) problem
        minimize   f(x, y) = 0.5 * y^Ty - x^T y
        subject to h(y) = \|y\|^2 <= 1
    """

    def __init__(self, n=2, m=2):
        super().__init__(n, m)

    def objective(self, x, y):
        return 0.5 * np.dot(y, y) - np.dot(y, x)

    def constraint(self, x, y):
        return np.dot(y, y) - 1.0

    def solve(self, x):
        x_norm_sq = np.dot(x, x)
        y_star = x.copy() if (x_norm_sq <= 1.0) else 1.0 / np.sqrt(x_norm_sq) * x
        return y_star, None

    def gradient(self, x, y=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        x_norm_sq = np.dot(x, x)
        if (x_norm_sq < 1.0):
            return np.eye(self.dim_x)
        return 1.0 / np.power(x_norm_sq, 1.5) * (x_norm_sq * np.eye(self.dim_x) - np.outer(x, x))


class CosineDistance(NonUniqueDeclarativeNode):
    """
    Solves the unconstrained problem
       minimize f(x, y) = x^T y / \|y\|
    """
    def __init__(self):
        super().__init__(2, 2)
        self.alpha = 1.0

    def objective(self, x, y):
        return -1.0 * np.dot(x, y) / np.sqrt(np.dot(y, y))

    def solve(self, x):
        return self.alpha * x.copy(), None

    def gradient(self, x, y=None, ctx=None):
        """Since D_{YY}^2 f is singular computes one possible gradient for testing."""
        return self.alpha * (np.eye(2, 2) - np.outer(x, x) / np.dot(x, x))
