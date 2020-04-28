#
# Euclidean projection onto the Lp-sphere
#
# y(x) = argmin_u f(x, u)
# subject to h(u) = 0
#
# where f(x, u) = 0.5 ||u - x||_2^2
#       h(u) = ||u||_p = 1
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>
#

import torch
import torch.nn.functional as F

class Simplex():
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto a positive simplex

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to sum_i w_i = z, w_i >= 0 

        using the algorithm (Figure 1) from:
        [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions,
            John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra,
            International Conference on Machine Learning (ICML 2008),
            http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the simplex

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the simplex

        Complexity:
            O(n log(n))
            A linear time alternative is proposed in [1], similar to using a
            selection algorithm instead of sorting.
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # 1. Sort v into mu (decreasing)
        mu, _ = v.sort(dim = -1, descending = True)
        # 2. Find rho (number of strictly positive elements of optimal solution w)
        mu_cumulative_sum = mu.cumsum(dim = -1)
        rho = torch.sum(mu * torch.arange(1, v.size()[-1] + 1, dtype=v.dtype, device=v.device) > (mu_cumulative_sum - z), dim = -1, keepdim=True)
        # 3. Compute the Lagrange multiplier theta associated with the simplex constraint
        theta = (torch.gather(mu_cumulative_sum, -1, (rho - 1)) - z) / rho.type(v.dtype)
        # 4. Compute projection
        w = (v - theta).clamp(min = 0.0)
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        input = input.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        # Select active positivity constraints
        mask = torch.where(output > 0.0, torch.ones_like(input), torch.zeros_like(input))
        masked_output = mask * grad_output
        grad_input = masked_output - mask * (
            masked_output.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True))
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input

class L1Sphere(Simplex):
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an L1-sphere

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_1 = z

        using the algorithm (Figure 1) from:
        [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions,
            John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra,
            International Conference on Machine Learning (ICML 2008),
            http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the L1-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the L1-sphere

        Complexity:
            O(n log(n))
            A linear time alternative is proposed in [1], similar to using a
            selection algorithm instead of sorting.
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # # 1. Replace v = 0 with v = [1, 0, ..., 0]
        # mask = torch.isclose(v, torch.zeros_like(v), rtol=0.0, atol=1e-12).sum(dim=-1, keepdim=True) == v.size(-1)
        # unit_vector = F.one_hot(v.new_zeros(1, dtype=torch.long), num_classes=v.size(-1)).type(v.dtype)
        # v = torch.where(mask, unit_vector, v)
        # 1. Take the absolute value of v
        u = v.abs()
        # 2. Project u onto the positive simplex
        beta, _ = Simplex.project(u, z=z)
        # 3. Correct the element signs
        w = beta * torch.where(v < 0, -torch.ones_like(v), torch.ones_like(v))
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        DYh = output.sign()
        grad_input = DYh.abs() * grad_output - DYh * (
            (DYh * grad_output).sum(-1, keepdim=True) / (DYh * DYh).sum(-1, keepdim=True))
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input

class L1Ball(L1Sphere):
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an L1-ball

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_1 <= z

        using the algorithm (Figure 1) from:
        [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions,
            John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra,
            International Conference on Machine Learning (ICML 2008),
            http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the L1-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the L1-ball

        Complexity:
            O(n log(n))
            A linear time alternative is proposed in [1], similar to using a
            selection algorithm instead of sorting.
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # 1. Project onto L1 sphere
        w, _ = L1Sphere.project(v, z=z)
        # 2. Select v if already inside ball, otherwise select w
        is_outside = v.abs().sum(dim=-1, keepdim=True).gt(z)
        w = torch.where(is_outside, w, v)
        return w, is_outside

    @staticmethod
    def gradient(grad_output, output, input, is_outside):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Compute constrained gradient
        grad_input = L1Sphere.gradient(grad_output, output, input, is_outside)
        # 2. If input was already inside ball (or on surface), use unconstrained gradient instead
        grad_input = torch.where(is_outside, grad_input, grad_output)
        return grad_input

class L2Sphere():
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an L2-sphere

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_2 = z

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the L2-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the L2-sphere

        Complexity:
            O(n)
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # Replace v = 0 with unit vector:
        mask = torch.isclose(v, torch.zeros_like(v), rtol=0.0, atol=1e-12).sum(dim=-1, keepdim=True) == v.size(-1)
        unit_vector = torch.ones_like(v).div(torch.ones_like(v).norm(p=2, dim=-1, keepdim=True))
        v = torch.where(mask, unit_vector, v)
        # Compute projection:
        w = z * v.div(v.norm(p=2, dim=-1, keepdim=True))
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # ToDo: Check for div by zero
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        input = input.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        output_norm = output.norm(p=2, dim=-1, keepdim=True)
        input_norm = input.norm(p=2, dim=-1, keepdim=True)
        ratio = output_norm.div(input_norm)
        grad_input = ratio * (grad_output - output * (
            output * grad_output).sum(-1, keepdim=True).div(output_norm.pow(2)))
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input

class L2Ball(L2Sphere):
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an L2-ball

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_2 <= z

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the L2-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the L2-ball

        Complexity:
            O(n)
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # 1. Project onto L2 sphere
        w, _ = L2Sphere.project(v, z=z)
        # 2. Select v if already inside ball, otherwise select w
        is_outside = v.norm(p=2, dim=-1, keepdim=True).gt(z)
        w = torch.where(is_outside, w, v)
        return w, is_outside

    @staticmethod
    def gradient(grad_output, output, input, is_outside):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Compute constrained gradient
        grad_input = L2Sphere.gradient(grad_output, output, input, is_outside)
        # 2. If input was already inside ball (or on surface), use unconstrained gradient instead
        grad_input = torch.where(is_outside, grad_input, grad_output)
        return grad_input

class LInfSphere():
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an LInf-sphere

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_infinity = z

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the LInf-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the LInf-sphere

        Complexity:
            O(n)
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # 1. Take the absolute value of v
        u = v.abs()
        # 2. Project u onto the (non-negative) LInf-sphere
        # If u_i >= z, u_i = z
        # If u_i < z forall i, find max and set to z
        z = torch.tensor(z, dtype=v.dtype, device=v.device)
        u = torch.where(u.gt(z), z, u)
        u = torch.where(u.ge(u.max(dim=-1, keepdim=True)[0]), z, u)
        # 3. Correct the element signs
        w = u * torch.where(v < 0, -torch.ones_like(v), torch.ones_like(v))
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        mask = output.abs().ge(output.abs().max(dim=-1, keepdim=True)[0])
        hY = output.sign() * mask.type(output.dtype)
        grad_input = grad_output - hY.abs() * grad_output
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input

class LInfBall(LInfSphere):
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an LInf-ball

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_infinity <= z

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the LInf-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the LInf-ball

        Complexity:
            O(n)
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # Using LInfSphere.project is more expensive here
        # 1. Take the absolute value of v
        u = v.abs()
        is_outside = u.max(dim=-1, keepdim=True)[0].gt(z) # Store for backward pass
        # 2. Project u onto the (non-negative) LInf-sphere if outside
        # If u_i >= z, u_i = z
        z = torch.tensor(z, dtype=v.dtype, device=v.device)
        u = torch.where(u.gt(z), z, u)
        # 3. Correct the element signs
        w = u * torch.where(v < 0, -torch.ones_like(v), torch.ones_like(v))
        return w, is_outside

    @staticmethod
    def gradient(grad_output, output, input, is_outside):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Compute constrained gradient
        grad_input = LInfSphere.gradient(grad_output, output, input, is_outside)
        # 2. If input was already inside ball (or on surface), use unconstrained gradient instead
        grad_input = torch.where(is_outside, grad_input, grad_output)
        return grad_input

class EuclideanProjectionFn(torch.autograd.Function):
    """
    A function to project a set of features to an Lp-sphere or Lp-ball
    """
    @staticmethod
    def forward(ctx, input, method, radius):
        output, is_outside = method.project(input, radius.item())
        ctx.method = method
        ctx.save_for_backward(output.clone(), input.clone(), is_outside)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input, is_outside = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.method.gradient(grad_output, output, input, is_outside)
        return grad_input, None, None

class EuclideanProjection(torch.nn.Module):
    def __init__(self, method, radius = 1.0):
        super(EuclideanProjection, self).__init__()
        self.method = method
        self.register_buffer('radius', torch.tensor([radius]))

    def forward(self, input):
        return EuclideanProjectionFn.apply(input,
                                           self.method,
                                           self.radius
                                           )

    def extra_repr(self):
        return 'method={}, radius={}'.format(
            self.method.__name__, self.radius
        )

""" Check gradients
from torch.autograd import gradcheck

# method = Simplex
method = L1Sphere
# method = L1Ball
# method = L2Sphere
# method = L2Ball
# method = LInfSphere
# method = LInfBall

radius = 100.0
radius = 1.0
# radius = 0.5

projection = EuclideanProjectionFn.apply
radius_tensor = torch.tensor([radius], requires_grad=False)
features = torch.randn(4, 2, 2, 100, dtype=torch.double, requires_grad=True)
input = (features, method, radius_tensor)
test = gradcheck(projection, input, eps=1e-6, atol=1e-4)
print("{}: {}".format(method.__name__, test))

# Check projections
features = torch.randn(1, 1, 1, 10, dtype=torch.double, requires_grad=True)
input = (features, method, radius_tensor)
print(features.sum(dim=-1))
print(features.abs().sum(dim=-1))
print(features.norm(p=2, dim=-1))
print(features.abs().max(dim=-1)[0])
print(features)
output = projection(*input)
print(output.sum(dim=-1))
print(output.abs().sum(dim=-1))
print(output.norm(p=2, dim=-1))
print(output.abs().max(dim=-1)[0])
print(output)
"""