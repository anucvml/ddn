# SAMPLE DEEP DECLARATIVE NODES
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>

import torch
import math
from ddn.pytorch.node import *

class SquaredErrorNode(AbstractNode):
    """Computes the squared difference between the input and a given target vector."""
    def __init__(self, x_target):
        super().__init__()
        self.x_target = x_target

    def solve(self, x):
        return 0.5 * ((x - self.x_target) ** 2).sum(dim=-1), None

    def gradient(self, x, y=None, v=None, ctx=None):
        if v is None:
            v = x.new_ones(x.size(0)) # b
        return torch.einsum('b,bm->bm', (v, x - self.x_target)),


class UnconstPolynomial(AbstractDeclarativeNode):
    """Solves min. f(x, y) = xy^4 + 2x^2y^3 - 12y^2  from Gould et al., 2016. Takes smallest x over the three
    stationary points."""
    def __init__(self):
        super().__init__()
        
    def objective(self, x, y):
        return (x * y ** 2.0 + 2 * x ** 2.0 * y - 12) * y ** 2.0

    def solve(self, x):
        delta = (9.0 * x ** 4.0 + 96.0 * x).sqrt()
        y_stationary = torch.cat((torch.zeros_like(x), (-3.0 * x ** 2.0 - delta) / (4.0 * x), (-3.0 * x ** 2.0 + delta) / (4.0 * x)), dim=-1)
        y_min_indx = self.objective(x, y_stationary).argmin(dim=-1)
        y = torch.cat([torch.index_select(a, dim=0, index=i).unsqueeze(0) for a, i in zip(y_stationary, y_min_indx)])
        return y, None

    def gradient(self, x, y=None, v=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        x = x.detach()
        if y is None:
            y, ctx = self.solve(x)
        y = y.detach()
        if v is None:
            v = torch.ones_like(y)
        return torch.einsum('bm,bm->bm', (v, -1.0 * (y ** 3 + 3.0 * x * y ** 2) / (3.0 * x * y ** 2 + 3.0 * x ** 2 * y - 6.0))),

class GlobalPseudoHuberPool2d(AbstractDeclarativeNode):
    """"""
    def __init__(self, chunk_size=None):
        super().__init__(chunk_size=chunk_size)
        
    def objective(self, x, alpha, y):
        alpha2 = (alpha * alpha).unsqueeze(-1).expand_as(x)
        z = y.unsqueeze(-1).unsqueeze(-1) - x
        phi = alpha2 * (torch.sqrt(1.0 + torch.pow(z, 2) / alpha2) - 1.0)
        return phi.sum(dim=(-2,-1)) # b

    def runOptimisation(self, x, alpha, y):
        with torch.enable_grad():
            opt = torch.optim.LBFGS([y],
                                    lr=1, # Default: 1
                                    max_iter=100, # Default: 20
                                    max_eval=None, # Default: None
                                    tolerance_grad=1e-05, # Default: 1e-05
                                    tolerance_change=1e-09, # Default: 1e-09
                                    history_size=100, # Default: 100
                                    line_search_fn=None # Default: None, Alternative: "strong_wolfe"
                                    )
            def reevaluate():
                opt.zero_grad()
                f = self.objective(x, alpha, y).sum() # sum over batch elements
                f.backward()
                return f
            opt.step(reevaluate)
        return y

    def solve(self, x, alpha):
        x = x.detach()
        y = x.mean([-2, -1]).clone().requires_grad_()
        y = self.runOptimisation(x, alpha, y)
        y = y.detach()
        z = (y.unsqueeze(-1).unsqueeze(-1) - x).clone()
        ctx = {'z': z}
        return y, ctx

    def gradient(self, x, alpha, y=None, v=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        if y is None:
            y, ctx = self.solve(x, alpha)
        if v is None:
            v = torch.ones_like(y)
        z = ctx['z'] # b x n1 x n2
        alpha2 = (alpha * alpha).unsqueeze(-1).expand_as(z)
        w = torch.pow(1.0 + torch.pow(z, 2) / alpha2, -1.5)
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).expand_as(w)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum))  # b x n1 x n2
        return torch.einsum('b,bmn->bmn', (v, Dy_at_x)), None

class LinFcnOnUnitCircle(EqConstDeclarativeNode):
    """
    Solves the problem
        minimize   f(x, y) = (1, x)^Ty
        subject to h(y) = ||y||^2 = 1
    for 1d input (x) and 2d output (y).
    """
    def __init__(self):
        super().__init__()

    def objective(self, x, y):
        return y[:, 0] + y[:, 1] * x[:, 0]

    def equality_constraints(self, x, y):
        return torch.einsum('bm,bm->b', (y, y)) - 1.0

    def solve(self, x):
        x_aug = torch.cat((torch.ones_like(x), x), dim=-1) # bx2
        t = torch.sqrt(1.0 + torch.pow(x, 2.0))
        y = -1.0 * x_aug / t # bx2
        ctx = {'nu': -0.5 * t} # b
        return y, ctx

    def gradient(self, x, y=None, v=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        x = x.detach()
        if v is None:
            v = x.new_ones(x.size(0), 2) # bx2
        x_aug = torch.cat((x, -1.0 * torch.ones_like(x)), dim=-1) # bx2
        t = torch.pow(1.0 + torch.pow(x, 2.0), 1.5)
        return torch.einsum('bm,bmn->bn', (v, (x_aug / t).unsqueeze(-1))),

class ConstLinFcnOnParameterizedCircle(EqConstDeclarativeNode):
    """
    Solves the problem
        minimize   f(x, y) = (1, 1)^Ty
        subject to h(y) = ||y||^2 = x^2
    for 1d input (x) and 2d output (y).
    """
    def __init__(self):
        super().__init__()

    def objective(self, x, y):
        return y[:, 0] + y[:, 1]

    def equality_constraints(self, x, y):
        return torch.einsum('bm,bm->b', (y, y)) - torch.einsum('b,b->b', (x[:, 0], x[:, 0]))

    def solve(self, x):
        y = -1.0 * torch.abs(x[:, 0]).unsqueeze(-1) * x.new_ones(x.size(0), 2) / math.sqrt(2.0) # bx2
        nu = torch.where(x[:, 0] == 0.0, torch.zeros_like(x[:, 0]), 0.5 / y[:, 0])
        ctx = {'nu': nu}
        return y, ctx

    def gradient(self, x, y=None, v=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        x = x.detach()
        if v is None:
            v = x.new_ones(x.size(0), 2) # bx2
        Dy_at_x = -1.0 * torch.sign(x[:, 0]).unsqueeze(-1) * x.new_ones(x.size(0), 2) / math.sqrt(2.0) # bx2
        return torch.einsum('bm,bmn->bn', (v, Dy_at_x.unsqueeze(-1))), #

class LinFcnOnParameterizedCircle(EqConstDeclarativeNode):
    """
    Solves the problem
        minimize   f(x, y) = (1, x_1)^Ty
        subject to h(y) = \|y\|^2 = x_2^2
    for 2d input (x) and 2d output (y).
    """
    def __init__(self):
        super().__init__()

    def objective(self, x, y):
        return y[:, 0] + torch.einsum('b,b->b', (x[:, 0], y[:, 1]))

    def equality_constraints(self, x, y):
        return torch.einsum('bm,bm->b', (y, y)) - torch.einsum('b,b->b', (x[:, 1], x[:, 1]))

    def solve(self, x):
        y = -1.0 * torch.abs(x[:, 1]).unsqueeze(-1) * torch.cat((torch.ones_like(x[:, 0:1]), x[:, 0:1]), dim=-1) / torch.sqrt(1.0 + torch.pow(x[:, 0], 2.0)).unsqueeze(-1) # bx2
        nu = torch.where(x[:, 1] == 0.0, torch.zeros_like(x[:, 0]), 0.5 / y[:, 0])
        ctx = {'nu': nu}
        return y, ctx

    def gradient(self, x, y=None, v=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        x = x.detach()
        if v is None:
            v = x.new_ones(x.size(0), 2) # bx2
        a = torch.abs(x[:, 1]).unsqueeze(-1) * torch.cat((x[:, 0:1], -1.0 * torch.ones_like(x[:, 0:1])), dim=-1) / torch.pow(1.0 + torch.pow(x[:, 0], 2.0), 1.5).unsqueeze(-1) # bx2
        b = -1.0 * torch.sign(x[:, 1]).unsqueeze(-1) * torch.cat((torch.ones_like(x[:, 0:1]), x[:, 0:1]), dim=-1) / torch.sqrt(1.0 + torch.pow(x[:, 0], 2.0)).unsqueeze(-1) # bx2
        Dy_at_x = torch.stack((a, b), dim=-1)
        return torch.einsum('bm,bmn->bn', (v, Dy_at_x)),

class QuadFcnOnSphere(EqConstDeclarativeNode):
    """
    Solves the problem
        minimize   f(x, y) = 0.5 * y^Ty - x^T y
        subject to h(y) = \|y\|^2 = 1
    """

    def __init__(self):
        super().__init__()

    def objective(self, x, y):
        return 0.5 * torch.einsum('bm,bm->b', (y, y)) - torch.einsum('bm,bm->b', (y, x))

    def equality_constraints(self, x, y):
        return torch.einsum('bm,bm->b', (y, y)) - 1.0

    def solve(self, x):
        y = x / torch.sqrt(torch.einsum('bm,bm->b', (x, x))).unsqueeze(-1)
        return y, None

    def gradient(self, x, y=None, v=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        x = x.detach()
        if v is None:
            v = torch.ones_like(x) # bxm
        x_inner = torch.einsum('bm,bm->b', (x, x))
        x_outer = torch.einsum('bm,bn->bmn', (x, x))
        eye_batch = torch.eye(x.size(1), dtype=x.dtype, device=x.device).expand_as(x_outer)
        Dy_at_x = (torch.einsum('b,bmn->bmn', (x_inner, eye_batch)) - x_outer) / torch.pow(torch.einsum('bm,bm->b', (x, x)), 1.5).unsqueeze(-1).unsqueeze(-1)
        return torch.einsum('bm,bmn->bn', (v, Dy_at_x)),

class QuadFcnOnBall(IneqConstDeclarativeNode):
    """
    Solves the (inequality constrained) problem
        minimize   f(x, y) = 0.5 * y^Ty - x^T y
        subject to h(y) = \|y\|^2 <= 1
    """

    def __init__(self):
        super().__init__()

    def objective(self, x, y):
        return 0.5 * torch.einsum('bm,bm->b', (y, y)) - torch.einsum('bm,bm->b', (y, x))

    def inequality_constraints(self, x, y):
        return torch.einsum('bm,bm->b', (y, y)) - 1.0

    def solve(self, x):
        x_norm_sq = torch.einsum('bm,bm->b', (x, x)).unsqueeze(-1) # bx1
        y = torch.where(x_norm_sq <= 1.0, x.clone(), x / torch.sqrt(x_norm_sq))
        return y, None

    def gradient(self, x, y=None, v=None, ctx=None):
        """Override base class to compute the analytic gradient of the optimal solution."""
        x = x.detach()
        if v is None:
            v = torch.ones_like(x) # bxm
        x_inner = torch.einsum('bm,bm->b', (x, x))
        x_outer = torch.einsum('bm,bn->bmn', (x, x))
        eye_batch = torch.eye(x.size(1), dtype=x.dtype, device=x.device).expand_as(x_outer)
        Dy_at_x = torch.where(x_inner.unsqueeze(-1).unsqueeze(-1) <= 1.0, eye_batch,
            (torch.einsum('b,bmn->bmn', (x_inner, eye_batch)) - x_outer) / torch.pow(torch.einsum('bm,bm->b', (x, x)), 1.5).unsqueeze(-1).unsqueeze(-1))
        return torch.einsum('bm,bmn->bn', (v, Dy_at_x)),
