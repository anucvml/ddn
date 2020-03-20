# DEEP DECLARATIVE NODES
# Defines the PyTorch interface for data processing nodes and declarative nodes
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>

import torch
from torch.autograd import grad
import warnings

class AbstractNode:
    """Minimal interface for generic data processing node
    that produces an output vector given an input vector.
    """
    def __init__(self):
        """Create a node"""

    def solve(self, *xs):
        """Computes the output of the node given the inputs.
        The second returned object provides context for computing the gradient
        if necessary. Otherwise it is None.
        """
        raise NotImplementedError()
        return None, None

    def gradient(self, *xs, y=None, v=None, ctx=None):
        """Computes the vector--Jacobian product of the node for given inputs xs
        and, optional, output y, gradient vector v and context cxt.
        If y or ctx is not provided then they are recomputed from x as needed.
        Implementation must return a tuple.
        """
        raise NotImplementedError()
        return None

    def _expand_as_batch(self, x, b):
        """Helper function to replicate tensor along a new batch dimension
        without allocating new memory.

        Arguments:
            x: (...) Torch tensor,
                input tensor
            
            b: scalar,
                batch size

        Return Values:
            batched tensor: (b, ...) Torch tensor
        """
        return x.expand(b, *x.size())

class AbstractDeclarativeNode(AbstractNode):
    """A general deep declarative node defined by unconstrained parameterized
    optimization problems of the form
        minimize (over y) f(x, y)
    where x is given (as a vector) and f is a scalar-valued function.
    Derived classes must implement the `objective` and `solve` functions.
    """
    eps = 1.0e-4 # Tolerance to check that optimality conditions are satisfied

    def __init__(self):
        """Create a declarative node
        """
        super().__init__()

    def objective(self, *xs, y):
        """Evaluates the objective function on a given input-output pair.
        Multiple input tensors can be passed as arguments, but the final
        argument must be the output tensor.
        """
        warnings.warn("objective function not implemented")
        return None

    def solve(self, *xs):
        """Solves the optimization problem
            y in argmin_u f(x, u)
        and returns two outputs. The first is the optimal solution y and the
        second contains the context for computing the gradient, such as the
        Lagrange multipliers in the case of a constrained problem, or None
        if no context is available/needed.
        Multiple input tensors can be passed as arguments.
        Implementation should wrap function contents in "with torch.no_grad():"
        to prevent graph creation
        """
        raise NotImplementedError()
        # Todo: LBFGS fall-back solver
        return None, None

    def gradient(self, *xs, y=None, v=None, ctx=None):
        """Computes the vector--Jacobian product, that is, the gradient of the
        loss function with respect to the problem parameters. The returned
        gradient is a tuple of batched Torch tensors. Can be overridden by the
        derived class to provide a more efficient implementation.

        Arguments:
            xs: ((b, ...), ...) tuple of Torch tensors,
                tuple of batches of input tensors

            y: (b, ...) Torch tensor or None,
                batch of minima of the objective function
            
            v: (b, ...) Torch tensor or None,
                batch of gradients of the loss function with respect to the
                problem output J_Y(x,y)

            ctx: dictionary of contextual information used for computing the
                 gradient

        Return Values:
            gradients: ((b, ...), ...) tuple of Torch tensors or Nones,
                batch of gradients of the loss function with respect to the
                problem parameters;
                strictly, returns the vector--Jacobian products J_Y(x,y) * y'(x)
        """
        # Compute optimal value if have not already done so:
        if y is None:
            y, ctx = self.solve(*xs)
            y.requires_grad = True
        # Set incoming gradient v = J_Y(x,y) to one if not specified:
        if v is None:
            v = torch.ones_like(y)

        # Compute relevant derivatives with autograd:
        b = y.size(0)
        m = y.view(b, -1).size(-1)
        with torch.enable_grad():
            # Split each input x into a tuple of n tensors of size bx1:
            # Required since gradients can only be computed wrt individual
            # tensors, not slices of a tensor. See:
            # https://discuss.pytorch.org/t/how-to-calculate-gradients-wrt-one-of-inputs/24407
            xs_split, xs_sizes = self._split_inputs(xs)

            # Evaluate objective function at (xs,y):
            f = self.objective(*self._cat_inputs(xs_split, xs_sizes), y) # b

        # Compute partial derivative of f wrt y at (xs,y):
        fY = grad(f, y, grad_outputs=torch.ones_like(f), create_graph=True
            )[0].view(b, -1) # bxm
        
        if not self._check_optimality_cond(fY):
            warnings.warn("Non-zero objective function gradient {} at y".format(
                fY.detach().squeeze().cpu().numpy()))
        
        # Compute second-order partial derivative of f wrt y at (xs,y):
        fYY = self._batch_jacobian(fY, y)

        # Solve u = -H^-1 v:
        H = fYY.detach()
        H = 0.5 * (H + H.transpose(1, 2)) # Ensure that H is symmetric
        v = v.view(b, -1, 1)
        u = self._solve_linear_system(H, -1.0 * v) # bxmx1
        u = u.squeeze(-1) # bxm

        # ToDo: check for NaN values in u

        # Compute -b_i^T H^-1 v (== b_i^T u) for all i:
        gradients = []
        for x_split, x_size in zip(xs_split, xs_sizes): # Loop over input tuple
            if isinstance(x_split[0], torch.Tensor) and x_split[0].requires_grad:
                n = len(x_split)
                gradient = x_split[0].new_zeros(b, n) # bxn
                # 2nd-order partial derivative of f wrt y and x at (xs,y):
                fXiY = torch.zeros_like(fY) # bxm
                grad_outputs = torch.ones_like(fY)
                for i in range(n):
                    with torch.enable_grad():
                        fXiY = grad(fY, x_split[i], grad_outputs=grad_outputs,
                            create_graph=True)[0] # bxm
                    bi = fXiY.detach()
                    gradient[:, i] = torch.einsum('bm,bm->b', (bi, u))
                # Reshape gradient to size(x):
                gradients.append(gradient.view(x_size))
            else:
                gradients.append(None)
        return tuple(gradients)

    def jacobian(self, *xs, y=None, ctx=None):
        """Computes the Jacobian, that is, the derivative of the output with
        respect to the problem parameters. The returned Jacobian is a tuple of
        batched Torch tensors. Can be overridden by the derived class to provide
        a more efficient implementation.
        Note: this function is highly inefficient so should be used for learning
        purposes only (computes the vector--Jacobian product multiple times).

        Arguments:
            xs: ((b, ...), ...) tuple of Torch tensors,
                tuple of batches of input tensors

            y: (b, ...) Torch tensor or None,
                batch of minima of the objective function

            ctx: dictionary of contextual information used for computing the
                 gradient

        Return Values:
            jacobians: ((b, ...), ...) tuple of Torch tensors or Nones,
                batch of Jacobians of the loss function with respect to the
                problem parameters
        """
        v = torch.zeros_like(y) # v: bxm1xm2x...
        b = v.size(0)
        v = v.view(b, -1) # v: bxm
        m = v.size(-1)
        jacobians = [[] for x in xs]
        for i in range(m):
            v[:, i] = 1.0
            gradients = self.gradient(*xs, y=y, v=v.view_as(y), ctx=ctx)
            v[:, i] = 0.0
            for j in range(len(xs)):
                jacobians[j].append(gradients[j])
        jacobians = [torch.stack(jacobian, dim=1).reshape(
            y.shape + xs[i].shape[1:]
            ) for i, jacobian in enumerate(jacobians)] # bxm1xm2x...xn1xn2
        return tuple(jacobians)

    def _split_inputs(self, xs):
        """Split inputs into a sequence of tensors by input dimension
        For each input x in xs, generates a tuple of n tensors of size bx1
        """
        xs_split, xs_sizes = [], []
        for x in xs: # Loop over input tuple
            if isinstance(x, torch.Tensor) and x.requires_grad:
                b = x.size(0)
                xs_split.append(x.view(b, -1).split(1, dim=-1))
                xs_sizes.append(x.size())
            else:
                xs_split.append((x,))
                xs_sizes.append(None) # May not be a tensor
        return tuple(xs_split), tuple(xs_sizes)

    def _cat_inputs(self, xs_split, xs_sizes):
        """Concatenate inputs from a sequence of tensors
        """
        xs = []
        for x_split, x_size in zip(xs_split, xs_sizes): # Loop over input tuple
            if len(x_split) > 1:
                xs.append(torch.cat(x_split, dim=-1).view(x_size))
            else:
                xs.append(x_split[0])
        return tuple(xs)

    def _check_optimality_cond(self, fY):
        """Checks that the problem's 1st-order optimality condition is satisfied
        """
        return torch.allclose(fY, torch.zeros_like(fY), rtol=0.0, atol=self.eps)

    def _solve_linear_system(self, A, B):
        """Solves linear system AX = B.
        If B is a tuple (B1, B2, ...), returns tuple (X1, X2, ...).
        Otherwise returns X.
        """
        B_sizes = None
        # If B is a tuple, concatenate into single tensor:
        if isinstance(B, (tuple, list)):
            B_sizes = list(map(lambda x: x.size(-1), B))
            B = torch.cat(B, dim=-1)
        # Ensure B is 2D (bxmxn):
        if len(B.size()) == 2:
            B = B.unsqueeze(-1)
        try: # Batchwise Cholesky solve
            A_decomp = torch.cholesky(A, upper=False)
            X = torch.cholesky_solve(B, A_decomp, upper=False) # bxmxn
        except: # Revert to loop if batchwise solve fails
            X = torch.zeros_like(B)
            for i in range(A.size(0)):
                try: # Cholesky solve
                    A_decomp = torch.cholesky(A[i, ...], upper=False)
                    X[i, ...] = torch.cholesky_solve(B[i, ...], A_decomp,
                        upper=False) # mxn
                except: # Revert to LU solve
                    X[i, ...], _ = torch.solve(B[i, ...], A[i, ...]) # mxn
        if B_sizes is not None:
            X = X.split(B_sizes, dim=-1)
        return X

    @torch.enable_grad()
    def _batch_jacobian(self, y, x, create_graph=False):
        """Compute Jacobian of y with respect to x and reduce over batch
        dimension.

        Arguments:
            y: (b, m1, m2, ...) Torch tensor,
                batch of output tensors

            x: (b, n1, n2, ...) Torch tensor,
                batch of input tensors

            create_graph: Boolean
                if True, graph of the derivative will be constructed,
                allowing the computation of higher order derivative products

        Return Values:
            jacobian: (b, m, n) Torch tensor,
                batch of Jacobian matrices, collecting the partial derivatives
                of y with respect to x
                m = product(m_i)
                n = product(n_i)

        Assumption:
            If x is not in graph for y[:, 0], then x is not in the graph for
            y[:, i], for all i
        """
        b = y.size(0)
        y = y.view(b, -1) # bxm
        m = y.size(-1)
        n = x.view(b, -1).size(-1)
        jacobian = y.new_zeros(b, m, n) # bxmxn
        for i in range(m):
            grad_outputs = torch.zeros_like(y, requires_grad=False) # bxm
            grad_outputs[:, i] = 1.0
            yiX, = grad(y, x, grad_outputs=grad_outputs, retain_graph=True,
                create_graph=create_graph, allow_unused=True) # bxn1xn2x...
            if yiX is None: # grad returns None instead of zero
                return None # If any are None, all are None
            jacobian[:, i:(i+1), :] = yiX.view(b, -1).unsqueeze(1) # bx1xn
        return jacobian # bxmxn

class EqConstDeclarativeNode(AbstractDeclarativeNode):
    """A general deep declarative node defined by a parameterized optimization
    problem with at least one (non-linear) equality constraint of the form
        minimize (over y) f(x, y)
        subject to        h_i(x, y) = 0
    where x is given (as a vector) and f and h_i are scalar-valued functions.
    Derived classes must implement the `objective`, `equality_constraints` and
    `solve` functions.
    """

    def __init__(self):
        """Create an equality constrained declarative node
        """
        super().__init__()

    def equality_constraints(self, *xs, y):
        """Evaluates the equality constraint functions on a given input-output
        pair. Multiple input tensors can be passed as arguments, but the final
        argument must be the output tensor.
        """
        warnings.warn("equality constraint function not implemented")
        return None

    def solve(self, *xs):
        """Solves the optimization problem
            y in argmin_u f(x, u) subject to h_i(x, u) = 0
        and returns the vector y. Optionally, also returns the Lagrange
        multipliers associated with the equality constraints where the
        Lagrangian is defined as
            L(x, y, nu) = f(x, y) - sum_i ctx['nu'][i] * h_i(x, y)
        Otherwise, should return None as second return variable.
        If the calling function only cares about the optimal solution
        (and not the context) then call as
            y_star, _ = self.solve(x)
        Multiple input tensors can be passed as arguments.
        Implementation should wrap function contents in "with torch.no_grad():"
        to prevent graph creation
        """
        raise NotImplementedError()
        return None, None

    def gradient(self, *xs, y=None, v=None, ctx=None):
        """Computes the vector--Jacobian product, that is, the gradient of the
        loss function with respect to the problem parameters. The returned
        gradient is a tuple of batched Torch tensors. Can be overridden by the
        derived class to provide a more efficient implementation.

        Arguments:
            xs: ((b, ...), ...) tuple of Torch tensors,
                tuple of batches of input tensors

            y: (b, ...) Torch tensor or None,
                batch of minima of the objective function
            
            v: (b, ...) Torch tensor or None,
                batch of gradients of the loss function with respect to the
                problem output J_Y(x,y)

            ctx: dictionary of contextual information used for computing the
            gradient

        Return Values:
            gradients: ((b, ...), ...) tuple of Torch tensors or Nones,
                batch of gradients of the loss function with respect to the
                problem parameters;
                strictly, returns the vector--Jacobian products J_Y(x,y) * y'(x)
        """

        # Compute optimal value if have not already done so:
        if y is None:
            y, ctx = self.solve(*xs)
            y.requires_grad = True
        # Set incoming gradient v = J_Y(x,y) to one if not specified:
        if v is None:
            v = torch.ones_like(y)

        # Compute relevant derivatives with autograd:
        b = y.size(0)
        m = y.view(b, -1).size(-1)
        with torch.enable_grad():
            # Split each input x into a tuple of n tensors of size bx1:
            # Required since gradients can only be computed wrt individual
            # tensors, not slices of a tensors. See:
            # https://discuss.pytorch.org/t/how-to-calculate-gradients-wrt-one-of-inputs/24407
            xs_split, xs_sizes = self._split_inputs(xs)
            xs = self._cat_inputs(xs_split, xs_sizes)

            # Evaluate constraint function(s) at (xs,y):
            h = self._get_constraint_set(xs, y) # bxp
            if h is None: # If None, use unconstrained gradient
                return super().gradient(xs, y=y, v=v, ctx=ctx)

            # Evaluate objective function at (xs,y):
            f = self.objective(*xs, y) # b

        # Compute partial derivative of f wrt y at (xs,y):
        fY = grad(f, y, grad_outputs=torch.ones_like(f), create_graph=True
            )[0].view(b, -1) # bxm
        if not fY.requires_grad: # if fY is independent of y
            fY.requires_grad = True

        # Compute partial derivative of h wrt y at (xs,y):
        hY = self._batch_jacobian(h, y, create_graph=True)
        if not hY.requires_grad: # if hY is independent of y
            hY.requires_grad = True

        # Compute nu (b, p):
        nu = self._get_nu(fY, hY) if (ctx is None or
            'nu' not in ctx) else ctx['nu']
        nu = nu.unsqueeze(-1) if len(nu.size()) == 1 else nu # Force p dimension

        if not self._check_optimality_cond(fY, hY, nu):
            warnings.warn(
                "Non-zero Lagrangian gradient {} at y. fY: {}, hY: {}, nu: {}".format(
                    (fY - torch.einsum('ab,abc->ac',
                        (nu, hY))).detach().squeeze().cpu().numpy(),
                    fY.detach().squeeze().cpu().numpy(),
                    hY.detach().squeeze().cpu().numpy(),
                    nu.detach().squeeze().cpu().numpy()))

        # Compute second-order partial derivative of f wrt y at (xs,y):
        fYY = self._batch_jacobian(fY, y)

        # Compute 2nd-order partial derivative of h wrt y at (xs,y) and form H:
        H = fYY.detach() if fYY is not None else 0.0 # Shares storage with fYY
        p = h.size(-1)
        for i in range(p):
            with torch.enable_grad(): # Needed when looping over output
                hiYY = self._batch_jacobian(hY[:, i, :], y, create_graph=False)
            if hiYY is not None:
                H -= torch.einsum('b,bmn->bmn', (nu[:, i], hiYY))
        assert isinstance(H, torch.Tensor)

        # Solve u = -H^-1 v (bxm) and t = H^-1 A^T (bxmxp):
        H = 0.5 * (H + H.transpose(1, 2)) # Ensure that H is symmetric
        A = hY.detach() # Shares storage with hY
        v = v.view(b, -1, 1) # bxmx1
        u, t = self._solve_linear_system(H, (-1.0 * v, A.transpose(-2, -1)))
        u = u.squeeze(-1) # bxm

        # ToDo: check for NaN values in u and t

        # Solve s = (A H^-1 A^T)^-1 A H^-1 v = -(A t)^-1 A u:
        s = self._solve_linear_system(torch.einsum('bpm,bmq->bpq', (A, t)),
            torch.einsum('bpm,bm->bp', (A, -1.0 * u))) # bxpx1
        s = s.squeeze(-1) # bxp
        
        # ToDo: check for NaN values in s

        # Compute u + ts:
        uts = u + torch.einsum('bmp,bp->bm', (t, s)) # bxm

        # Compute bi^T (u + ts) - ci^T s for all i:
        gradients = []
        for x_split, x_size in zip(xs_split, xs_sizes): # Loop over input tuple
            if isinstance(x_split[0], torch.Tensor) and x_split[0].requires_grad:
                n = len(x_split)
                gradient = x_split[0].new_zeros(b, n) # bxn
                for i in range(n):
                    # 2nd-order partial derivative of f wrt y and xi at (xs,y):
                    fXiY = self._batch_jacobian(fY, x_split[i]) # bxmx1
                    bi = fXiY.detach().squeeze(-1) if (
                        fXiY is not None) else (
                        torch.zeros_like(fY)) # Shares storage with fXiY
                    for j in range(p):
                        # 2nd-order partial derivative of hj wrt y and xi at (xs,y):
                        with torch.enable_grad():
                            hjXiY = self._batch_jacobian(hY[:, j, :],
                                x_split[i]) # bxmx1
                        if hjXiY is not None:
                            bi -= torch.einsum('b,bm->bm', (nu[:, j],
                                hjXiY.detach().squeeze(-1))) # bxm
                    # Compute partial derivative of h wrt xi at (xs,y):
                    hXi = self._batch_jacobian(h, x_split[i]) # bxpx1
                    if hXi is None:
                        gradient[:, i] = torch.einsum('bm,bm->b', (bi, uts))
                    else:
                        ci = hXi.detach().squeeze(-1) # Shares storage with hXi
                        gradient[:, i] = (torch.einsum('bm,bm->b', (bi, uts))
                            - torch.einsum('bp,bp->b', (ci, s)))
                # Reshape gradient to size(x):
                gradients.append(gradient.view(x_size))
            else:
                gradients.append(None)
        return tuple(gradients)

    def _get_constraint_set(self, xs, y):
        """Filters constraints.
        """
        # ToDo: remove duplicate constraints (first-order identical)
        h = self.equality_constraints(*xs, y)
        if h is not None:
            h = h.unsqueeze(-1) if len(h.size()) == 1 else h
            if not self._check_equality_constraints(h):
                warnings.warn("Constraints not satisfied {}".format(
                    h.detach().squeeze().cpu().numpy()))
        return h

    def _get_nu(self, fY, hY):
        """Compute nu (ie lambda) if not provided by the problem's solver.
        That is, solve: hY^T nu = fY^T.
        """
        b = hY.size(0)
        p = hY.size(1)
        nu = fY.new_zeros(b, p)
        for i in range(b): # loop over batch
            solution,_ = torch.lstsq(fY[i, :].unsqueeze(-1), hY[i, :, :].t())
            nu[i, :] = solution[:p, :].squeeze() # extract first p values
        return nu

    def _check_equality_constraints(self, h):
        """Check that the problem's constraints are satisfied.
        """
        return torch.allclose(h, torch.zeros_like(h), rtol=0.0, atol=self.eps)

    def _check_optimality_cond(self, fY, hY, nu=None):
        """Checks that the problem's first-order optimality condition is
        satisfied.
        """
        nu = self._get_nu(fY, hY) if (nu is None) else nu
        # Check for invalid Lagrangian (gradient of constraint zero at optimum)
        if torch.allclose(hY, torch.zeros_like(hY), rtol=0.0, atol=self.eps):
            warnings.warn(
                "Gradient of constraint function vanishes at the optimum.")
            return True
        LY = fY - torch.einsum('ab,abc->ac', (nu, hY)) # bxm - bxp * bxpxm
        return torch.allclose(LY, torch.zeros_like(fY), rtol=0.0, atol=self.eps)

class LinEqConstDeclarativeNode(EqConstDeclarativeNode):
    """A deep declarative node defined by a linear equality constrained
    parameterized optimization problem of the form:
        minimize (over y) f(x, y)
        subject to        A y = d
    where x is given, and A and d are independent of x. Derived classes must
    implement the objective and solve functions.
    """
    def __init__(self):
        """Create a linear equality constrained declarative node
        """
        super().__init__()

    def linear_constraint_parameters(self, y):
        """Defines the linear equality constraint parameters A and d, where the
        constraint is given by Ay = d.

        Arguments:
            y: (b, ...) Torch tensor,
                batch of minima of the objective function

        Return Values:
            (A, d): ((p, m), (p)) tuple of Torch tensors,
                linear equality constraint parameters
        """
        raise NotImplementedError()
        return None, None

    def gradient(self, *xs, y=None, v=None, ctx=None):
        """Computes the vector--Jacobian product, that is, the gradient of the
        loss function with respect to the problem parameters. The returned
        gradient is a tuple of batched Torch tensors. Can be overridden by the
        derived class to provide a more efficient implementation.

        Arguments:
            xs: ((b, ...), ...) tuple of Torch tensors,
                tuple of batches of input tensors

            y: (b, ...) Torch tensor or None,
                batch of minima of the objective function
            
            v: (b, ...) Torch tensor or None,
                batch of gradients of the loss function with respect to the
                problem output J_Y(x,y)

            ctx: dictionary of contextual information used for computing the
            gradient

        Return Values:
            gradients: ((b, ...), ...) tuple of Torch tensors or Nones,
                batch of gradients of the loss function with respect to the
                problem parameters;
                strictly, returns the vector--Jacobian products J_Y(x,y) * y'(x)
        """

        # Compute optimal value if have not already done so:
        if y is None:
            y, ctx = self.solve(*xs)
            y.requires_grad = True
        # Set incoming gradient v = J_Y(x,y) to one if not specified:
        if v is None:
            v = torch.ones_like(y)

        b = y.size(0)
        m = y.view(b, -1).size(-1)

        # Get constraint parameters and form batch:
        A, d = self.linear_constraint_parameters(y)
        A = self._expand_as_batch(A, b)
        d = self._expand_as_batch(d, b)

        # Check linear equality constraints are satisfied:
        h = torch.einsum('bpm,bm->bp', (A, y)) - d
        if not self._check_equality_constraints(h):
            warnings.warn("Constraints not satisfied {}".format(
                h.detach().squeeze().cpu().numpy()))
        
        # Compute relevant derivatives with autograd:        
        with torch.enable_grad():
            # Split each input x into a tuple of n tensors of size bx1:
            # Required since gradients can only be computed wrt individual
            # tensors, not slices of a tensors. See:
            # https://discuss.pytorch.org/t/how-to-calculate-gradients-wrt-one-of-inputs/24407
            xs_split, xs_sizes = self._split_inputs(xs)
            xs = self._cat_inputs(xs_split, xs_sizes)

            # Evaluate objective function at (xs,y):
            f = self.objective(*xs, y) # b

        # Compute partial derivative of f wrt y at (xs,y):
        grad_outputs = torch.ones_like(f) # b
        fY = grad(f, y, grad_outputs=grad_outputs, create_graph=True
            )[0].view(b, -1) # bxm
        if not fY.requires_grad: # if fY is independent of y
            fY.requires_grad = True

        # Compute second-order partial derivative of f wrt y at (xs,y):
        fYY = self._batch_jacobian(fY, y)
        assert fYY is not None

        # Compute 2nd-order partial derivative of h wrt y at (xs,y) and form H:
        H = fYY.detach()

        # Solve u = -H^-1 v (bxm) and t = H^-1 A^T (bxmxp):
        H = 0.5 * (H + H.transpose(1, 2)) # Ensure that H is symmetric
        v = v.view(b, -1, 1) # bxmx1
        u, t = self._solve_linear_system(H, (-1.0 * v, A.transpose(-2, -1)))
        u = u.squeeze(-1) # bxm

        # ToDo: check for NaN values in u and t

        # Solve s = (A H^-1 A^T)^-1 A H^-1 v = -(A t)^-1 A u:
        s = self._solve_linear_system(torch.einsum('bpm,bmq->bpq', (A, t)),
            torch.einsum('bpm,bm->bp', (A, -1.0 * u))) # bxpx1
        s = s.squeeze(-1) # bxp
        
        # ToDo: check for NaN values in s

        # Compute u + ts = -H^-1 v + H^-1 A^T (A H^-1 A^T)^-1 A H^-1 v:
        uts = u + torch.einsum('bmp,bp->bm', (t, s)) # bxm

        # Compute bi^T (u + ts) for all i:
        gradients = []
        for x_split, x_size in zip(xs_split, xs_sizes): # Loop over input tuple
            if isinstance(x_split[0], torch.Tensor) and x_split[0].requires_grad:
                n = len(x_split)
                gradient = x_split[0].new_zeros(b, n) # bxn
                for i in range(n):
                    # 2nd-order partial derivative of f wrt y and xi at (xs,y):
                    fXiY = self._batch_jacobian(fY, x_split[i]) # bxmx1
                    bi = fXiY.detach().squeeze(-1) if (
                        fXiY is not None) else (
                        torch.zeros_like(fY)) # Shares storage with fXiY
                    gradient[:, i] = torch.einsum('bm,bm->b', (bi, uts))
                # Reshape gradient to size(x):
                gradients.append(gradient.view(x_size))
            else:
                gradients.append(None)
        return tuple(gradients)

class DeclarativeFunction(torch.autograd.Function):
    """Generic declarative autograd function.
    Defines the forward and backward functions. Saves all inputs and outputs,
    which may be memory-inefficient for the specific problem.
    
    Assumptions:
    * All inputs are PyTorch tensors
    * All inputs have a single batch dimension (b, ...)
    """
    @staticmethod
    def forward(ctx, problem, *inputs):
        output, solve_ctx = problem.solve(*inputs)
        ctx.save_for_backward(output, *inputs)
        ctx.problem = problem
        ctx.solve_ctx = solve_ctx
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        output, *inputs = ctx.saved_tensors
        problem = ctx.problem
        solve_ctx = ctx.solve_ctx
        output.requires_grad = True
        inputs = tuple(inputs)
        grad_inputs = problem.gradient(*inputs, y=output, v=grad_output,
            ctx=solve_ctx)
        return (None, *grad_inputs)

class DeclarativeLayer(torch.nn.Module):
    """Generic declarative layer.
    
    Assumptions:
    * All inputs are PyTorch tensors
    * All inputs have a single batch dimension (b, ...)

    Usage:
        problem = <derived class of *DeclarativeNode>
        declarative_layer = DeclarativeLayer(problem)
        y = declarative_layer(x1, x2, ...)
    """
    def __init__(self, problem):
        super(DeclarativeLayer, self).__init__()
        self.problem = problem
        
    def forward(self, *inputs):
        return DeclarativeFunction.apply(self.problem, *inputs)

