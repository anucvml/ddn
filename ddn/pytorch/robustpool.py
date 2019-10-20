#
# Robust pooling
#
# y(x) = argmin_u f(x, u)
#
# where f(x, u) = sum_{i=1}^n phi(u - x_i; alpha)
# with penalty function phi in
# {quadratic, pseudo-huber, huber, welsch, truncated quadratic}
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>
#

import torch

class Quadratic():
    is_convex = True

    @staticmethod
    def phi(z, alpha = 1.0):
        """ Quadratic penalty function

        phi(z; alpha) = 0.5 * z^2

        Arguments:
            z: (b, ...) Torch tensor,
                batch of residuals

            alpha: float, optional, default: 1.0,
                ignored

        Return Values:
            phi_z: (b, ...) Torch tensor,
                Quadratic penalty associated with each residual

        Complexity:
            O(1)
        """
        phi_at_z = 0.5 * torch.pow(z, 2)
        return phi_at_z

    @staticmethod
    def Dy(z, alpha = 1.0):
        # Derivative of y(x) for the quadratic penalty function
        Dy_at_x = torch.ones_like(z) / (z.size(-1) * z.size(-2))
        return Dy_at_x

class PseudoHuber():
    is_convex = True

    @staticmethod
    def phi(z, alpha = 1.0):
        """ Pseudo-Huber penalty function

        phi(z; alpha) = alpha^2 (sqrt{1 + (z / alpha)^2} - 1)

        Arguments:
            z: (b, ...) Torch tensor,
                batch of residuals

            alpha: float, optional, default: 1.0,
                ~slope of the linear region
                ~maximum residual in the quadratic region

        Return Values:
            phi_z: (b, ...) Torch tensor,
                Pseudo-Huber penalty associated with each residual

        Complexity:
            O(1)
        """
        assert alpha > 0.0, "alpha must be strictly positive (%f <= 0)" % alpha
        alpha2 = alpha * alpha
        phi_at_z = alpha2 * (torch.sqrt(1.0 + torch.pow(z, 2) / alpha2) - 1.0)
        return phi_at_z

    @staticmethod
    def Dy(z, alpha = 1.0):
        # Derivative of y(x) for the pseudo-Huber penalty function
        w = torch.pow(1.0 + torch.pow(z, 2) / (alpha * alpha), -1.5)
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).expand_as(w)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum))
        return Dy_at_x

class Huber():
    is_convex = True

    @staticmethod
    def phi(z, alpha = 1.0):
        """ Huber penalty function

                        / 0.5 z^2 for |z| <= alpha
        phi(z; alpha) = |
                        \ alpha (|z| - 0.5 alpha) else

        Arguments:
            z: (b, ...) Torch tensor,
                batch of residuals

            alpha: float, optional, default: 1.0,
                slope of the linear region
                maximum residual in the quadratic region

        Return Values:
            phi_z: (b, ...) Torch tensor,
                Huber penalty associated with each residual

        Complexity:
            O(1)
        """
        assert alpha > 0.0, "alpha must be strictly positive (%f <= 0)" % alpha
        z = z.abs()
        phi_at_z = torch.where(z <= alpha, 0.5 * torch.pow(z, 2), alpha * (z - 0.5 * alpha))
        return phi_at_z

    @staticmethod
    def Dy(z, alpha = 1.0):
        # Derivative of y(x) for the Huber penalty function
        w = torch.where(z.abs() <= alpha, torch.ones_like(z), torch.zeros_like(z))
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).expand_as(w)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum))
        return Dy_at_x

class Welsch():
    is_convex = False

    @staticmethod
    def phi(z, alpha = 1.0):
        """ Welsch penalty function

        phi(z; alpha) = 1 - exp(-0.5 * z^2 / alpha^2)

        Arguments:
            z: (b, ...) Torch tensor,
                batch of residuals

            alpha: float, optional, default: 1.0,
                ~maximum residual in the quadratic region

        Return Values:
            phi_z: (b, ...) Torch tensor,
                Welsch penalty associated with each residual

        Complexity:
            O(1)
        """
        assert alpha > 0.0, "alpha must be strictly positive (%f <= 0)" % alpha
        phi_at_z = 1.0 - torch.exp(-torch.pow(z, 2) / (2.0 * alpha * alpha))
        return phi_at_z

    @staticmethod
    def Dy(z, alpha = 1.0):
        # Derivative of y(x) for the Welsch penalty function
        alpha2 = alpha * alpha
        z2_on_alpha2 = torch.pow(z, 2) / alpha2
        w = (1.0 - z2_on_alpha2) * torch.exp(-0.5 * z2_on_alpha2) / alpha2
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).expand_as(w)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum))
        Dy_at_x = torch.clamp(Dy_at_x, -1.0, 1.0) # Clip gradients to +/- 1
        return Dy_at_x

class TruncatedQuadratic():
    is_convex = False

    @staticmethod
    def phi(z, alpha = 1.0):
        """ Truncated quadratic penalty function

                        / 0.5 z^2 for |z| <= alpha
        phi(z; alpha) = |
                        \ 0.5 alpha^2 else

        Arguments:
            z: (b, ...) Torch tensor,
                batch of residuals

            alpha: float, optional, default: 1.0,
                maximum residual in the quadratic region

        Return Values:
            phi_z: (b, ...) Torch tensor,
                Truncated quadratic penalty associated with each residual

        Complexity:
            O(1)
        """
        assert alpha > 0.0, "alpha must be strictly positive (%f <= 0)" % alpha
        z = z.abs()
        phi_at_z = torch.where(z <= alpha, 0.5 * torch.pow(z, 2), 0.5 * alpha * alpha * torch.ones_like(z))
        return phi_at_z

    @staticmethod
    def Dy(z, alpha = 1.0):
        # Derivative of y(x) for the truncated quadratic penalty function
        w = torch.where(z.abs() <= alpha, torch.ones_like(z), torch.zeros_like(z))
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).expand_as(w)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum))
        return Dy_at_x

class RobustGlobalPool2dFn(torch.autograd.Function):
    """
    A function to globally pool a 2D response matrix using a robust penalty function
    """
    @staticmethod
    def runOptimisation(x, y, method, alpha_scalar):
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
                # Sum cost function across residuals and batch (all fi are positive)
                f = method.phi(y.unsqueeze(-1).unsqueeze(-1) - x, alpha=alpha_scalar).sum()
                f.backward()
                return f
            opt.step(reevaluate)
        return y

    @staticmethod
    def forward(ctx, x, method, alpha):
        input_size = x.size()
        assert len(input_size) >= 2, "input must at least 2D (%d < 2)" % len(input_size)
        alpha_scalar = alpha.item()
        assert alpha.item() > 0.0, "alpha must be strictly positive (%f <= 0)" % alpha.item()
        x = x.detach()
        x = x.flatten(end_dim=-3) if len(input_size) > 2 else x
        # Handle non-convex functions separately
        if method.is_convex:
            # Use mean as initial guess
            y = x.mean([-2, -1]).clone().requires_grad_()
            y = RobustGlobalPool2dFn.runOptimisation(x, y, method, alpha_scalar)
        else:
            # Use mean and median as initial guesses and choose the best
            # ToDo: multiple random starts
            y_mean = x.mean([-2, -1]).clone().requires_grad_()
            y_mean = RobustGlobalPool2dFn.runOptimisation(x, y_mean, method, alpha_scalar)
            y_median = x.flatten(start_dim=-2).median(dim=-1)[0].clone().requires_grad_()
            y_median = RobustGlobalPool2dFn.runOptimisation(x, y_median, method, alpha_scalar)
            f_mean = method.phi(y_mean.unsqueeze(-1).unsqueeze(-1) - x, alpha=alpha_scalar).sum(-1).sum(-1)
            f_median = method.phi(y_median.unsqueeze(-1).unsqueeze(-1) - x, alpha=alpha_scalar).sum(-1).sum(-1)
            y = torch.where(f_mean <= f_median, y_mean, y_median)
        y = y.detach()
        z = (y.unsqueeze(-1).unsqueeze(-1) - x).clone()
        ctx.method = method
        ctx.input_size = input_size
        ctx.save_for_backward(z, alpha)
        return y.reshape(input_size[:-2]).clone()

    @staticmethod
    def backward(ctx, grad_output):
        z, alpha = ctx.saved_tensors
        input_size = ctx.input_size
        method = ctx.method
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Flatten:
            grad_output = grad_output.detach().flatten(end_dim=-1)
            # Use implicit differentiation to compute derivative:
            grad_input = method.Dy(z, alpha) * grad_output.unsqueeze(-1).unsqueeze(-1)
            # Unflatten:
            grad_input = grad_input.reshape(input_size)
        return grad_input, None, None

class RobustGlobalPool2d(torch.nn.Module):
    def __init__(self, method, alpha=1.0):
        super(RobustGlobalPool2d, self).__init__()
        self.method = method
        self.register_buffer('alpha', torch.tensor([alpha]))

    def forward(self, input):
        return RobustGlobalPool2dFn.apply(input,
                                          self.method,
                                          self.alpha
                                          )

    def extra_repr(self):
        return 'method={}, alpha={}'.format(
            self.method, self.alpha
        )

""" Check gradients
from torch.autograd import gradcheck

alpha = 1.0
# alpha = 0.2
# alpha = 5.0

method = Quadratic
# method = PseudoHuber
# method = Huber
# method = Welsch # Can fail gradcheck due to numerically-necessary gradient clipping
# method = TruncatedQuadratic

robustPool = RobustGlobalPool2dFn.apply
alpha_tensor = torch.tensor([alpha], dtype=torch.double, requires_grad=False)
input = (torch.randn(2, 3, 7, 7, dtype=torch.double, requires_grad=True), method, alpha_tensor)
test = gradcheck(robustPool, input, eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=True)
print("{}: {}".format(method.__name__, test))
"""
