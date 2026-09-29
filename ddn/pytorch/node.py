# DEEP DECLARATIVE NODES
# Defines the PyTorch interface for prototyping declarative nodes. Choose the appropriate base class for your
# problem (unconstrained, equality constrained, etc). In most cases you will only need to implement the solver,
# objective and constraint methods. Gradients in the backward pass will then be automatically calculated. This
# will be most efficient if the code in your objective and constraint methods avoids looping over individual
# batch elements, i.e., use vectorized implementation wherever possible.
#
# When incorporating into a larger network, the typical pattern is wrap a node into a `DeclarativeLayer`,
#   class MyDeclarativeNode(AbstractDeclarativeNode):
#      def __init__(self, ...)
#      def solve(self, ...)
#      def objective(self, ...)
#
#   node = MyDeclarativeNode()
#   layer = DeclarativeLayer(node)
# The module `layer` can now be used as you would any other module in PyTorch (e.g., torch.nn.Linear).
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
        self.b = None
        self.m = None
        self.n = None

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

    def _expand_as_batch(self, x):
        """Helper function to replicate tensor along a new batch dimension
        without allocating new memory.

        Arguments:
            x: (...) Torch tensor,
                input tensor

        Return Values:
            batched tensor: (b, ...) Torch tensor
        """
        return x.expand(self.b, *x.size())

class AbstractDeclarativeNode(AbstractNode):
    """A general deep declarative node defined by unconstrained parameterized
    optimization problems of the form
        minimize (over y) f(x, y)
    where x is given (as a vector) and f is a scalar-valued function.
    Derived classes must implement the `objective` and `solve` functions.
    """
    def __init__(self, eps=1e-12, gamma=None, chunk_size=None):
        """Create a declarative node
        """
        super().__init__()
        self.eps = eps # tolerance to check if optimality conditions satisfied
        self.gamma = gamma # damping factor: H <-- H + gamma * I
        self.chunk_size = chunk_size # max. number of vector-Jacobian products computed at once (None = all)

    def objective(self, *xs, y):
        """Evaluates the objective function on a given input-output pair.
        Multiple input tensors can be passed as arguments, but the final
        argument must be the output tensor. For good performance during the
        backward pass it's best to vectorize calculations over the batch rather
        than looping over individual elements.
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
        xs, y, v, ctx = self._gradient_init(xs, y, v, ctx)

        fY, fYY = self._get_objective_derivatives(xs, y)

        if not self._check_optimality_cond(fY):
            warnings.warn(
                "Non-zero objective function gradient at y:\n{}".format(
                    fY.detach().squeeze().cpu().numpy()))

        # Form H:
        H = fYY
        H = 0.5 * (H + H.transpose(1, 2)) # Ensure that H is symmetric
        if self.gamma is not None:
            H += self.gamma * torch.eye(
                self.m, dtype=H.dtype, device=H.device).unsqueeze(0)

        # Solve u = -H^-1 v:
        v = v.reshape(self.b, -1, 1)
        u = self._solve_linear_system(H, -1.0 * v) # bxmx1
        u = u.squeeze(-1) # bxm

        # ToDo: check for NaN values in u

        # Compute -B^T H^-1 v (== B^T u) for all inputs as a single vector-Jacobian product,
        # B^T u = d/dx (fY^T u) with u held constant, without forming B = fXY:
        return self._input_vjp(xs, (fY, u))

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
        v = v.reshape(b, -1) # v: bxm
        m = v.size(-1)
        jacobians = [[] for x in xs]
        for i in range(m):
            v[:, i] = 1.0
            gradients = self.gradient(*xs, y=y, v=v.reshape_as(y), ctx=ctx)
            v[:, i] = 0.0
            for j in range(len(xs)):
                jacobians[j].append(gradients[j])
        jacobians = [torch.stack(jacobian, dim=1).reshape(
            y.shape + xs[i].shape[1:]) if (jacobian[0] is not None
            ) else None for i, jacobian in enumerate(jacobians)]
        return tuple(jacobians)

    def _gradient_init(self, xs, y, v, ctx):
        # Compute optimal value if have not already done so:
        if y is None:
            y, ctx = torch.no_grad()(self.solve)(*xs)
            y.requires_grad = True

        # Set incoming gradient v = J_Y(x,y) to one if not specified:
        if v is None:
            v = torch.ones_like(y)

        self.b = y.size(0)
        self.m = y.reshape(self.b, -1).size(-1)

        # Give each input its own node in the graph so that gradients are computed
        # per argument (even if the same tensor is passed as multiple arguments):
        xs = torch.enable_grad()(lambda: tuple(x.view_as(x)
            if isinstance(x, torch.Tensor) and x.requires_grad else x for x in xs))()

        return xs, y, v, ctx

    def _get_objective_derivatives(self, xs, y, hessian=True):
        # Evaluate objective function at (xs,y):
        f = torch.enable_grad()(self.objective)(*xs, y=y) # b

        # Compute partial derivative of f wrt y at (xs,y):
        fY = grad(f, y, grad_outputs=torch.ones_like(f), create_graph=True)[0]
        fY = torch.enable_grad()(fY.reshape)(self.b, -1) # bxm
        if not fY.requires_grad: # if fY is independent of y
            fY.requires_grad = True

        # Compute second-order partial derivative of f wrt y at (xs,y):
        fYY = self._hessian(fY, y) if hessian else None # bxmxm

        return fY, fYY

    def _hessian(self, gY, y):
        """Computes the (detached) Jacobian of the gradient gY (bxm) with respect to y,
        e.g., the Hessian of the objective or Lagrangian, returning zeros if gY is
        independent of y.
        """
        H = self._batch_jacobian(gY, y) # bxmxm
        return H.detach() if H is not None else y.new_zeros(
            self.b, self.m, self.m)

    @torch.enable_grad()
    def _input_vjp(self, xs, *terms):
        """Computes the gradient of z = sum_k sum(a_k * w_k) with respect to each
        input tensor in xs, where terms = ((a_1, w_1), (a_2, w_2), ...) and the
        weights w_k are held constant. Returns None for inputs that are not tensors
        or do not require grad. Used to evaluate the vector--Jacobian products
        B^T u and C^T s with a single backward pass rather than forming the
        Jacobians B and C explicitly.
        """
        indices = [i for i, x in enumerate(xs)
            if isinstance(x, torch.Tensor) and x.requires_grad]
        gradients = [None] * len(xs)
        if indices:
            z = sum((a * w.detach()).sum() for a, w in terms)
            vjps = grad(z, [xs[i] for i in indices], allow_unused=True)
            for i, vjp in zip(indices, vjps):
                # grad returns None instead of zero if z is independent of xs[i]
                gradients[i] = torch.zeros_like(xs[i]) if vjp is None else vjp
        return tuple(gradients)

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
        # Batchwise Cholesky solve (info > 0 flags batch elements where A is not positive definite)
        A_decomp, info = torch.linalg.cholesky_ex(A, upper=False)
        X = torch.cholesky_solve(B, A_decomp, upper=False) # bxmxn
        failed = info > 0 # b
        if failed.any(): # Revert to LU solve for failed batch elements only
            X[failed] = torch.linalg.solve(A[failed], B[failed]) # kxmxn
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
        y = y.reshape(self.b, -1) # bxm
        m = y.size(-1)
        n = x.reshape(self.b, -1).size(-1)
        k = m if self.chunk_size is None else min(max(1, self.chunk_size), m)
        if k > 1:
            # Compute the m vector-Jacobian products in vectorized (vmap) backward passes of
            # (at most) chunk_size products each, trading memory for speed:
            jacobian = []
            try:
                for rows in torch.eye(m, dtype=y.dtype, device=y.device).split(k):
                    grad_outputs = rows.unsqueeze(1).expand(-1, self.b, m) # kxbxm
                    yX, = grad(y, x, grad_outputs=grad_outputs, retain_graph=True,
                        create_graph=create_graph, allow_unused=True, is_grads_batched=True) # kxbxn1xn2x...
                    if yX is None: # grad returns None instead of zero
                        return None
                    jacobian.append(yX.reshape(-1, self.b, n))
                return torch.cat(jacobian).transpose(0, 1) # bxmxn
            except RuntimeError: # Revert to loop if graph contains operations unsupported by vmap
                pass
        jacobian = y.new_zeros(self.b, m, n) # bxmxn
        for i in range(m):
            grad_outputs = torch.zeros_like(y, requires_grad=False) # bxm
            grad_outputs[:, i] = 1.0
            yiX, = grad(y, x, grad_outputs=grad_outputs, retain_graph=True,
                create_graph=create_graph, allow_unused=True) # bxn1xn2x...
            if yiX is None: # grad returns None instead of zero
                return None # If any are None, all are None
            jacobian[:, i:(i+1), :] = yiX.reshape(self.b, -1).unsqueeze(1) # bx1xn
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

    def __init__(self, eps=1e-12, gamma=None, chunk_size=None):
        """Create an equality constrained declarative node
        """
        super().__init__(eps=eps, gamma=gamma, chunk_size=chunk_size)

    def equality_constraints(self, *xs, y):
        """Evaluates the equality constraint functions on a given input-output
        pair. Multiple input tensors can be passed as arguments, but the final
        argument must be the output tensor. For good performance during the
        backward pass it's best to vectorize calculations over the batch rather
        than looping over individual elements.
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
        xs, y, v, ctx = self._gradient_init(xs, y, v, ctx)

        fY, _ = self._get_objective_derivatives(xs, y, hessian=False)

        h, hY = self._get_constraint_derivatives(xs, y)

        nu = self._get_nu(fY, hY) if (ctx is None or 'nu' not in ctx
            ) else self._ensure2d(ctx['nu'])
        nu = nu.detach()

        if not self._check_optimality_cond(fY, hY, nu):
            warnings.warn("Non-zero Lagrangian gradient at y:\n{}\n"
                "fY: {}, hY: {}, nu: {}".format((fY - torch.einsum('ab,abc->ac',
                    (nu, hY))).detach().squeeze().cpu().numpy(),
                    fY.detach().squeeze().cpu().numpy(),
                    hY.detach().squeeze().cpu().numpy(),
                    nu.detach().squeeze().cpu().numpy()))

        # Form H as the Jacobian of the Lagrangian gradient LY = fY - nu^T hY wrt y,
        # H = fYY - sum_i nu_i hiYY, in a single pass:
        LY = torch.enable_grad()(lambda: fY - torch.einsum('bp,bpm->bm', (nu, hY)))() # bxm
        H = self._hessian(LY, y)
        H = 0.5 * (H + H.transpose(1, 2)) # Ensure that H is symmetric
        if self.gamma is not None:
            H += self.gamma * torch.eye(
                self.m, dtype=H.dtype, device=H.device).unsqueeze(0)

        # Solve u = -H^-1 v (bxm) and t = H^-1 A^T (bxmxp):
        A = hY.detach() # Shares storage with hY
        v = v.reshape(self.b, -1, 1) # bxmx1
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

        # Compute B^T (u + ts) - C^T s for all inputs as a single vector-Jacobian
        # product, where B = d/dx LY and C = d/dx h, without forming B or C:
        return self._input_vjp(xs, (LY, uts), (h, -1.0 * s))

    def _get_constraint_derivatives(self, xs, y):
        # Evaluate constraint function(s) at (xs,y):
        h = torch.enable_grad()(self._get_constraint_set)(xs, y) # bxp

        # Compute partial derivative of h wrt y at (xs,y):
        hY = self._batch_jacobian(h, y, create_graph=True) # bxpxm
        if not hY.requires_grad: # if hY is independent of y
            hY.requires_grad = True

        return h, hY

    def _get_constraint_set(self, xs, y):
        """Filters constraints.
        """
        # ToDo: remove duplicate constraints (first-order identical)
        h = self.equality_constraints(*xs, y=y)
        if h is not None:
            h = self._ensure2d(h)
            if not self._check_equality_constraints(h):
                warnings.warn("Constraints not satisfied exactly:\n{}".format(
                    h.detach().squeeze().cpu().numpy()))
        return h

    def _get_nu(self, fY, hY):
        """Compute nu (i.e., lambda) if not provided by the problem's solver.
        That is, solve: hY^T nu = fY^T.
        """
        p = hY.size(1)
        nu = torch.linalg.lstsq(hY.transpose(1, 2), fY)[0][:, :p]
        return nu

    def _check_equality_constraints(self, h):
        """Check that the problem's constraints are satisfied.
        """
        return torch.allclose(h, torch.zeros_like(h), rtol=0.0, atol=self.eps)

    def _check_optimality_cond(self, fY, hY=None, nu=None):
        """Checks that the problem's first-order optimality condition is
        satisfied.
        """
        if hY is None:
            return super()._check_optimality_cond(fY)

        nu = self._get_nu(fY, hY) if (nu is None) else nu
        # Check for invalid Lagrangian (gradient of constraint zero at optimum)
        if torch.allclose(hY, torch.zeros_like(hY), rtol=0.0, atol=self.eps):
            warnings.warn(
                "Gradient of constraint function vanishes at the optimum.")
            return True
        LY = fY - torch.einsum('ab,abc->ac', (nu, hY)) # bxm - bxp * bxpxm
        return torch.allclose(LY, torch.zeros_like(fY), rtol=0.0, atol=self.eps)

    def _ensure2d(self, x):
        return x.unsqueeze(-1) if len(x.size()) == 1 else x

class IneqConstDeclarativeNode(EqConstDeclarativeNode):
    """A general deep declarative node defined by a parameterized optimization
    problem with at least one (non-linear) inequality constraint of the form
        minimize (over y) f(x, y)
        subject to        h_i(x, y) == 0
                          g_i(x, y) <= 0
    where x is given (as a vector) and f, h_i and g_i are scalar-valued
    functions. Derived classes must implement the `objective`,
    `inequality_constraints` and `solve` functions.
    """

    def __init__(self, eps=1e-12, gamma=None, chunk_size=None):
        """Create an inequality constrained declarative node
        """
        super().__init__(eps=eps, gamma=gamma, chunk_size=chunk_size)

    def equality_constraints(self, *xs, y):
        """Evaluates the equality constraint functions on a given input-output
        pair. Multiple input tensors can be passed as arguments, but the final
        argument must be the output tensor. For good performance during the
        backward pass it's best to vectorize calculations over the batch rather
        than looping over individual elements.
        """
        return None

    def inequality_constraints(self, *xs, y):
        """Evaluates the inequality constraint functions on a given input-output
        pair. Multiple input tensors can be passed as arguments, but the final
        argument must be the output tensor. For good performance during the
        backward pass it's best to vectorize calculations over the batch rather
        than looping over individual elements.
        """
        warnings.warn("inequality constraint function not implemented")
        return None

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
        xs, y, v, ctx = self._gradient_init(xs, y, v, ctx)

        # Collect batch indices such that each sub-batch will have the same
        # number of active constraints:
        indices_list, unconstrained = self._get_uniform_indices(xs, y)

        # If all batch elements have same number of active constraints:
        if indices_list is None:
            if unconstrained:
                gradients = AbstractDeclarativeNode.gradient(self,
                    *xs, y=y, v=v, ctx=ctx)
            else:
                gradients = EqConstDeclarativeNode.gradient(self,
                    *xs, y=y, v=v, ctx=ctx)
        else: # Otherwise, loop over uniform batch subsets:
            gradients = [torch.zeros_like(x)
                if x.requires_grad else None for x in xs]
            for indices in indices_list:
                xs_subset = tuple([x.index_select(0, indices).requires_grad_()
                    for x in xs])
                y_subset = y.index_select(0, indices).requires_grad_()
                v_subset = v.index_select(0, indices)
                ctx_subset = None if ctx is None else {
                    key : value.index_select(0, indices)
                    if isinstance(value, torch.Tensor) else value
                    for key, value in ctx.items()}
                if unconstrained:
                    gradients_subset = AbstractDeclarativeNode.gradient(self,
                        *xs_subset, y=y_subset, v=v_subset, ctx=ctx_subset)
                    unconstrained = False # Only first subset is uncontrained
                else:
                    gradients_subset = EqConstDeclarativeNode.gradient(self,
                        *xs_subset, y=y_subset, v=v_subset, ctx=ctx_subset)
                # Insert gradients into correct locations:
                for i in range(len(gradients)):
                    if gradients[i] is not None:
                        gradients[i][indices, ...] = gradients_subset[i]
            gradients = tuple(gradients)
        return gradients

    def _get_uniform_indices(self, xs, y):
        """Collects batch indices such that each subset will have the same
        number of active constraints.

        Arguments:
            xs: ((b, ...), ...) tuple of Torch tensors,
                tuple of batches of input tensors

            y: (b, ...) Torch tensor,
                batch of minima of the objective function

        Return values:
            indices_list: [(k1), (k2), ...] list of Torch tensors or None,
                list of variable-length index tensors

            unconstrained: bool,
                true if first subset has no active constraints
        """
        h = self.equality_constraints(*xs, y=y) # bxp or None
        p = 0 if h is None else self._ensure2d(h).size(-1)
        g = self.inequality_constraints(*xs, y=y) # bxq
        if g is None:
            indices_list = None
            unconstrained = True if p == 0 else False
        else:
            g = self._ensure2d(g)
            q = torch.stack([gi.isclose(torch.zeros_like(gi),
                rtol=0.0, atol=self.eps).long().sum() for gi in g])
            q_sorted, indices = q.sort()
            q_unique, counts = q_sorted.unique_consecutive(return_counts=True)
            indices_list = indices.split(counts.split(1)) if (
                q_unique.size(-1) > 1) else None
            unconstrained = True if (p + q_unique[0] == 0) else False
        return indices_list, unconstrained

    def _get_constraint_set(self, xs, y):
        """Filters constraints.
        
        Arguments:
            xs: ((b, ...), ...) tuple of Torch tensors,
                tuple of batches of input tensors

            y: (b, ...) Torch tensor,
                batch of minima of the objective function

        Return values:
            constraint_set: (b, p) Torch tensor,
                tensor of active constraints

        Assumptions:
            batch has a uniform number of active constraints
        """
        # ToDo: remove duplicate constraints (first-order identical)
        constraint_set = None
        h = self.equality_constraints(*xs, y=y) # bxp or None
        if h is not None:
            h = self._ensure2d(h)
            if not self._check_equality_constraints(h):
                warnings.warn(
                    "Equality constraints not satisfied exactly:\n{}".format(
                    h.detach().squeeze().cpu().numpy()))
            constraint_set = h # bxp

        g = self.inequality_constraints(*xs, y=y) # bxq
        if g is not None:
            g = self._ensure2d(g)
            if not self._check_inequality_constraints(g):
                warnings.warn(
                    "Inequality constraints not satisfied exactly:\n{}".format(
                    g.detach().squeeze().cpu().numpy()))
            # Identify active constraints:
            mask = g.isclose(torch.zeros_like(g), rtol=0.0, atol=self.eps)
            g = g.masked_select(mask).reshape(self.b, -1) if mask.any() else None

            if h is None:
                constraint_set = g # bxq
            elif g is not None:
                constraint_set = torch.cat((h, g), dim=-1) # bx(p+q)
        return constraint_set

    def _check_inequality_constraints(self, g):
        """Check that the problem's constraints are satisfied."""
        return torch.all(g <= self.eps)

class LinEqConstDeclarativeNode(EqConstDeclarativeNode):
    """A deep declarative node defined by a linear equality constrained
    parameterized optimization problem of the form:
        minimize (over y) f(x, y)
        subject to        A y = d
    where x is given, and A and d are independent of x. Derived classes must
    implement the objective and solve functions.
    """
    def __init__(self, eps=1e-12, gamma=None, chunk_size=None):
        """Create a linear equality constrained declarative node
        """
        super().__init__(eps=eps, gamma=gamma, chunk_size=chunk_size)

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
        xs, y, v, ctx = self._gradient_init(xs, y, v, ctx)

        fY, fYY = self._get_objective_derivatives(xs, y)

        # Get constraint parameters and form batch:
        A, d = self.linear_constraint_parameters(y)
        A = self._expand_as_batch(A)
        d = self._expand_as_batch(d)

        # Check linear equality constraints are satisfied:
        h = torch.einsum('bpm,bm->bp', (A, y)) - d
        if not self._check_equality_constraints(h):
            warnings.warn("Constraints not satisfied exactly:\n{}".format(
                h.detach().squeeze().cpu().numpy()))

        # Form H:
        H = fYY
        H = 0.5 * (H + H.transpose(1, 2)) # Ensure that H is symmetric
        if self.gamma is not None:
            H += self.gamma * torch.eye(
                self.m, dtype=H.dtype, device=H.device).unsqueeze(0)

        # Solve u = -H^-1 v (bxm) and t = H^-1 A^T (bxmxp):    
        v = v.reshape(self.b, -1, 1) # bxmx1
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

        # Compute B^T (u + ts) for all inputs as a single vector-Jacobian product:
        return self._input_vjp(xs, (fY, uts))

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
        output, solve_ctx = torch.no_grad()(problem.solve)(*inputs)
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

