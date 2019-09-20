# DEEP DECLARATIVE NODES
# Defines the interface for data processing nodes and declarative nodes
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
#

import autograd.numpy as np
from autograd import grad, jacobian
import warnings

class AbstractNode:
    """
    Minimal interface for generic data processing node that produces an output vector given an input vector.
    """

    def __init__(self, n=1, m=1):
        """
        Create a node
        :param n: dimensionality of the input (parameters)
        :param m: dimensionality of the output (optimization solution)
        """
        assert (n > 0) and (m > 0)
        self.dim_x = n # dimensionality of input variable
        self.dim_y = m # dimensionality of output variable

    # TODO: rename to evaluate
    def solve(self, x):
        """Computes the output of the node given the input."""
        raise NotImplementedError()
        return None, None

    def gradient(self, x, y = None):
        """Computes the output of the node given input x and, optional, output y. If y is not provided then
        it is recomputed from x."""
        raise NotImplementedError()
        return None


class AbstractDeclarativeNode(AbstractNode):
    """
    A general deep declarative node defined by an unconstrained parameterized optimization problems of the form
        minimize (over y) f(x, y)
    where x is given (as a vector) and f is a scalar-valued function. Derived classes must implement the `objective`
    and `solve` functions.
    """

    eps = 1.0e-6 # tolerance for checking that optimality conditions are satisfied

    def __init__(self, n=1, m=1):
        """
        Creates an declarative node with optimization problem implied by the objecive function. Initializes the
        partial derivatives of the objective function for use in computing gradients.
        """
        super().__init__(n, m)

        # partial derivatives of objective
        self.fY = grad(self.objective, 1)
        self.fYY = jacobian(self.fY, 1)
        self.fXY = jacobian(self.fY, 0)

    def objective(self, x, y):
        """Evaluates the objective function on a given input-output pair."""
        warnings.warn("objective function not implemented.")
        return 0.0

    def solve(self, x):
        """
        Solves the optimization problem
            y \in argmin_u f(x, u)
        and returns two outputs. The first is the optimal solution y and the second is the largrange multipliers in
        the case of a constrained problem and None otherwise.
        """

        raise NotImplementedError()
        return None, None

    def gradient(self, x, y_star=None):
        """
        Computes the gradient of the output (problem solution) with respect to the problem
        parameters. The returned gradient is an ndarray of size (self.dim_y, self.dim_x). In
        the case of 1-dimensional parameters the gradient is a vector of size (self.dim_y,).
        Can be overridden by the derived class to provide a more efficient implementation.
        """

        # compute optimal value if not already done so
        if y_star is None:
            y_star, _ = self.solve(x)
        assert self._check_optimality_cond(x, y_star)

        # TODO: replace with symmetric matrix solver
        return -1.0 * np.linalg.solve(self.fYY(x, y_star), self.fXY(x, y_star))

    def _check_optimality_cond(self, x, y_star, nu_star=None):
        """Checks that the problem's first-order optimality condition is satisfied."""
        return (abs(self.fY(x, y_star)) <= self.eps).all()


class EqConstDeclarativeNode(AbstractDeclarativeNode):
    """
    A general deep declarative node defined by a parameterized optimization problem with single (non-linear)
    equality constraint of the form
        minimize (over y) f(x, y)
        subject to        h(x, y) = 0
    where x is given (as a vector) and f and h are scalar-valued functions. Derived classes must implement the
    `objective`, `constraint` and `solve` functions.
    """

    def __init__(self, n, m):
        super().__init__(n, m)

        # partial derivatives of constraint function
        self.hY = grad(self.constraint, 1)
        self.hX = grad(self.constraint, 0)
        self.hYY = jacobian(self.hY, 1)
        self.hXY = jacobian(self.hY, 0)

    def constraint(self, x, y):
        """Evaluates the equality constraint function on a given input-output pair."""
        warnings.warn("constraint function not implemented.")
        return 0.0

    def solve(self, x):
        """
        Solves the optimization problem
            y \in argmin_u f(x, u) subject to h(x, u) = 0
        and returns the vector y. Optionally, also returns the Lagrange multiplier associated
        with the equality constraint where the Lagrangian is defined as
            L(x, y, nu) = f(x, y) - nu * h(x, y)
        Otherwise, should return None as second return variable.
        If the calling function only cares about the optimal solution then call as
            y_star, _ = self.solve(x)
        """

        raise NotImplementedError()
        return None, None

    def gradient(self, x, y_star=None, nu_star=None):
        """Compute the gradient of the output (problem solution) with respect to the problem
        parameters. The returned gradient is an ndarray of size (prob.dim_y, prob.dim_x). In
        the case of 1-dimensional parameters the gradient is a vector of size (prob.dim_y,)."""

        # compute optimal value if not already done so
        if y_star is None:
            y_star, nu_star = self.solve(x)
        assert self._check_constraints(x, y_star), [x, y_star, abs(self.constraint(x, y_star))]
        assert self._check_optimality_cond(x, y_star, nu_star), [x, y_star, nu_star]

        if nu_star is None:
            nu_star = self._get_nu_star(x, y_star)

        # return unconstrained gradient if undefined
        if np.isnan(nu_star):
            return -1.0 * np.linalg.solve(self.fYY(x, y_star), self.fXY(x, y_star))

        H = self.fYY(x, y_star) - nu_star * self.hYY(x, y_star)
        a = self.hY(x, y_star)
        B = self.fXY(x, y_star) - nu_star * self.hXY(x, y_star)
        C = self.hX(x, y_star)
        try:
            # TODO: replace with symmetric solver
            v = np.linalg.solve(H, np.concatenate((a.reshape((self.dim_y, 1)), B), axis=1))
        except:
            return np.full((self.dim_y, self.dim_x), np.nan).squeeze()
        return (np.outer(v[:, 0], (v[:, 0].dot(B) - C) / v[:, 0].dot(a)) - v[:, 1:self.dim_x + 1]).squeeze()

    def _get_nu_star(self, x, y_star):
        """Compute nu_star if not provided by the problem's solver."""
        indx = np.nonzero(self.hY(x, y_star))
        if len(indx[0]) == 0:
            return 0.0
        return self.fY(x, y_star)[indx[0][0]] / self.hY(x, y_star)[indx[0][0]]

    def _check_constraints(self, x, y_star):
        """Check that the problem's constraints are satisfied."""
        return abs(self.constraint(x, y_star)) <= self.eps

    def _check_optimality_cond(self, x, y_star, nu_star=None):
        """Checks that the problem's first-order optimality condition is satisfied."""
        if nu_star is None:
            nu_star = self._get_nu_star(x, y_star)

        if np.isnan(nu_star):
            return (abs(self.fY(x, y_star)) <= self.eps).all()

        # check for invalid lagrangian (gradient of constraint zero at optimal point)
        if (abs(self.hY(x, y_star)) <= self.eps).all():
            warnings.warn("gradient of constraint function vanishes at the optimum.")
            return True
        return (abs(self.fY(x, y_star) - nu_star * self.hY(x, y_star)) <= self.eps).all()


class IneqConstDeclarativeNode(EqConstDeclarativeNode):
    """
    A general deep declarative node defined by a parameterized optimization problem with single (non-linear)
    inequality constraint of the form
        minimize (over y) f(x, y)
        subject to        h(x, y) <= 0
    where x is given (as a vector) and f and h are scalar-valued functions. Derived classes must implement the
    `objective`, `constraint` and `solve` functions.
    """

    def __init__(self, n, m):
        super().__init__(n, m)

    def _get_nu_star(self, x, y_star):
        """Compute nu_star if not provided by the problem's solver."""
        if np.all(np.abs(self.fY(x, y_star)) < self.eps):
            return np.nan # flag that unconstrained gradient should be used
        indx = np.nonzero(self.hY(x, y_star))
        if len(indx[0]) == 0:
            return 0.0 # still use constrained gradient
        return self.fY(x, y_star)[indx[0][0]] / self.hY(x, y_star)[indx[0][0]]

    def _check_constraints(self, x, y_star):
        """Check that the problem's constraints are satisfied."""
        return self.constraint(x, y_star) <= self.eps


class LinEqConstDeclarativeNode(AbstractDeclarativeNode):
    """
    A deep declarative node defined by a linear equality constrained parameterized optimization problem of the form:
        minimize (over y) f(x, y)
        subject to        A y = b
    where x is given. Derived classes must implement the objective and solve functions.
    """

    def __init__(self, n, m, A, b):
        super().__init__(n, m)
        assert A.shape[1] == m, "second dimension of A must match dimension of y"
        assert A.shape[0] == b.shape[0], "dimension of A must match dimension of b"
        self.A, self.b = A, b

    def gradient(self, x, y_star=None, nu_star=None):
        """Compute the gradient of the output (problem solution) with respect to the problem
        parameters. The returned gradient is an ndarray of size (prob.dim_y, prob.dim_x). In
        the case of 1-dimensional parameters the gradient is a vector of size (prob.dim_y,)."""

        # compute optimal value if not already done so
        if y_star is None:
            y_star, nu_star = self.solve(x)
        assert self._check_constraints(x, y_star)
        assert self._check_optimality_cond(x, y_star, nu_star)

        # TODO: replace with symmetric matrix solver and avoid explicit inverse matrix computations
        invH = np.linalg.inv(self.fYY(x, y_star))
        invHAT = np.dot(invH, self.A.T)
        w = np.dot(np.dot(invHAT, np.linalg.inv(np.dot(self.A, invHAT))), invHAT.T) - invH
        return np.dot(w, self.fXY(x, y_star))

    def _check_constraints(self, x, y_star):
        """Check that the problem's constraints are satisfied."""
        residual = np.dot(self.A, y_star) - self.b
        return np.all(np.abs(residual) <= self.eps)

    def _check_optimality_cond(self, x, y_star, nu_star=None):
        """Checks that the problem's first-order optimality condition is satisfied."""
        warnings.warn("optimality check not implemented yet")
        return True


class NonUniqueDeclarativeNode(AbstractDeclarativeNode):
    """
    A general deep declarative node having non-unique solutions so that the pseudo-inverse is required
    in computing the gradient.
    """
    def __init__(self, n, m):
        super().__init__(n, m)

    def gradient(self, x, y_star=None):
        """
        Computes the gradient of the output (problem solution) with respect to the problem parameters
        using a pseudo-inverse. The returned gradient is an ndarray of size (self.dim_y, self.dim_x).
        In the case of 1-dimensional parameters the gradient is a vector of size (self.dim_y,).
        """

        # compute optimal value if not already done so
        if y_star is None:
            y_star, _ = self.solve(x)
        assert self._check_optimality_cond(x, y_star), abs(self.fY(x, y_star))

        return -1.0 * np.linalg.pinv(self.fYY(x, y_star)).dot(self.fXY(x, y_star))
