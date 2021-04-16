#!/usr/bin/env python
#
# LEAST SQUARES NODES
# Implementation of differentiable weighted and unweighted least squares nodes. Can be used for rank pooling operations
# as well as many other tasks. See accompanying Jupyter Notebook tutorial at https://deepdeclarativenetworks.com.
#
# Stephen Gould <stephen.gould@anu.edu.au>
#

import torch
import torch.nn as nn

#
# --- PyTorch Functions ---
#

class WeightedLeastSquaresFcn(torch.autograd.Function):
    """
    PyTorch autograd function for weighted least squares,

        y, y_0 = argmin_{u, u_0} 1/2 \sum_i w_i (x_i^T u + u_0 - t_i)^2 + beta/2 \|u\|^2,

    returning y and y_0. Features x, target t and weights w are all provided as input.

    Assumes the input data (features) is provided as a (B,C,T) tensor, which is interpreted
    as B sequences of C-dimensional features, each sequence of length T. Weights and target
    are provided as (B,1,T) or (1,1,T) tensors. In the case of the latter, the values are
    replicated across batches, which may be useful if learning them as shared parameters
    in the model. Outputs are (B,C) and (B,1) tensors for y and y_0, respectively. Weights
    can also be None, indicating uniform weighting.

    Works well for feature sizes (C) up to 1024 dimensions.
    """

    @staticmethod
    def forward(ctx, input, target, weights=None, beta=1.0e-3, cache_decomposition=False):
        # allocate output tensors
        B, C, T = input.shape
        assert target.shape == (B, 1, T) or target.shape == (1, 1, T), "{} vs {}".format(input.shape, target.shape)
        assert weights is None or weights.shape == (B, 1, T) or weights.shape == (1, 1, T), "{} vs {}".format(input.shape, weights.shape)
        if cache_decomposition:
            L = torch.empty((B, C + 1, C + 1), device=input.device, dtype=input.dtype, requires_grad=False)
        else:
            L = None

        # replicate across batch if sharing weights or target
        with torch.no_grad():
            if target.shape[0] != B:
                target = target.repeat(B, 1, 1)
                ctx.collapse_target = True
            else:
                ctx.collapse_target = False
            if weights is not None and weights.shape[0] != B:
                weights = weights.repeat(B, 1, 1)
                ctx.collapse_weights = True
            else:
                ctx.collapse_weights = False

            # compute solution y and pack into output
            # Warning: if beta is zero or too small then the problem may not be strongly convex
            weightedX = input if weights is None else torch.einsum("bnm,bm->bnm", input, weights.view(B, -1))
            weightedTSum = target.sum(2).view(B, 1) if weights is None else torch.einsum("bm,bm->b", target.view(B, -1), weights.view(B, -1)).view(B, 1)
            weightedXdotT = torch.einsum("bnm,bm->bn", weightedX, target.view(B, -1))

            AtA = torch.empty((B, C + 1, C + 1), device=input.device, dtype=input.dtype)
            AtA[0:B, -1, -1] = T if weights is None else torch.sum(weights.view(B, -1), 1)
            AtA[:, 0:C, 0:C] = torch.einsum("bik,bjk->bij", weightedX, input) + \
                (beta * torch.eye(C, device=input.device, dtype=input.dtype)).view(1, C, C)
            AtA[:, 0:C, -1] = AtA[:, -1, 0:C] = torch.sum(weightedX, 2)

            if cache_decomposition:
                torch.cholesky(AtA, out=L)
                y = torch.cholesky_solve(torch.cat((weightedXdotT, weightedTSum), 1).view(B, C + 1, 1), L)
            else:
                y, _ = torch.solve(torch.cat((weightedXdotT, weightedTSum), 1).view(B, C + 1, 1), AtA)

            # assign to output
            output = y[:, 0:C, 0].squeeze(-1)
            bias = y[:, C, 0].view(B, 1)

        # save state for backward pass
        ctx.save_for_backward(input, target, weights, output, bias, L)
        ctx.beta = beta

        # return rank pool vector and bias
        return output, bias

    @staticmethod
    def backward(ctx, grad_output, grad_bias):
        # check for None tensors
        if grad_output is None and grad_bias is None:
            return None, None

        # unpack cached tensors
        input, target, weights, output, bias, L = ctx.saved_tensors
        B, C, T = input.shape
        weightedX = input if weights is None else torch.einsum("bnm,bm->bnm", input, weights.view(B, -1))

        # solve for w
        if L is None:
            AtA = torch.empty((B, C + 1, C + 1), device=input.device, dtype=input.dtype)
            AtA[0:B, -1, -1] = T if weights is None else torch.sum(weights.view(B, -1), 1)
            AtA[:, 0:C, 0:C] = torch.einsum("bik,bjk->bij", weightedX, input) + \
                (ctx.beta * torch.eye(C, device=input.device, dtype=input.dtype)).view(1, C, C)
            AtA[:, 0:C, -1] = AtA[:, -1, 0:C] = torch.sum(weightedX, 2)

            w, _ = torch.solve(torch.cat((grad_output, grad_bias), 1).view(B, C + 1, 1), AtA)
        else:
            w = torch.cholesky_solve(torch.cat((grad_output, grad_bias), 1).view(B, C + 1, 1), L)

        # compute w^T B
        grad_weights = None
        if weights is not None:
            grad_input = w[:, 0:-1].view(B, C, 1) * torch.mul(weights,
                (target.view(B, T) - torch.einsum("bn,bnm->bm", output.view(B, C), input) - bias.view(B, 1)).view(B, 1, T)) - \
                torch.mul(weights, (torch.einsum("bn,bnm->bm", w[:, 0:C].view(B, C), input) + w[:,C].view(B, 1)).view(B, 1, T)) * \
                output.view(B, C, 1)

            grad_target = (torch.einsum("bn,bnm->bm", w[:, 0:C].view(B, C), weightedX) +
                w[:, C].view(B, 1) * weights.view(B, T)).view(B, 1, T)

            grad_weights = ((target.view(B, T) - torch.einsum("bn,bnm->bm", output.view(B, C), input) - bias.view(B, 1)) *
                (torch.einsum("bn,bnm->bm", w[:, 0:C].view(B, C), input) + w[:, C].view(B, 1))).view(B, 1, T)

        else:
            grad_input = w[:, 0:-1].view(B, C, 1) * \
                (target.view(B, T) - torch.einsum("bn,bnm->bm", output.view(B, C), input) - bias.view(B, 1)).view(B, 1, T) - \
                (torch.einsum("bn,bnm->bm", w[:, 0:C].view(B, C), input) + w[:, C].view(B, 1)).view(B, 1, T) * \
                output.view(B, C, 1)

            grad_target = (torch.einsum("bn,bnm->bm", w[:, 0:C].view(B, C), weightedX) + w[:, C].view(B, 1)).view(B, 1, T)

        if ctx.collapse_target:
            grad_target = torch.sum(grad_target, 0, keepdim=True)
        if ctx.collapse_weights:
            grad_weights = torch.sum(grad_weights, 0, keepdim=True)

        # return gradients (None for `beta` and `cache_decomposition`)
        return grad_input, grad_target, grad_weights, None, None


#
# --- PyTorch Modules ---
#

class LeastSquaresLayer(nn.Module):
    """Neural network layer to implement (unweighted) least squares fitting."""

    def __init__(self, beta=1.0e-3, cache_decomposition=False):
        super(LeastSquaresLayer, self).__init__()
        self.beta = beta
        self.cache_decomposition = cache_decomposition

    def forward(self, input, target):
        return WeightedLeastSquaresFcn.apply(input, target, None, self.beta, self.cache_decomposition)


class WeightedLeastSquaresLayer(nn.Module):
    """Neural network layer to implement weighted least squares fitting."""

    def __init__(self, beta=1.0e-3, cache_decomposition=False):
        super(WeightedLeastSquaresLayer, self).__init__()
        self.beta = beta
        self.cache_decomposition = cache_decomposition

    def forward(self, input, target, weights):
        return WeightedLeastSquaresFcn.apply(input, target, weights, self.beta, self.cache_decomposition)


#
# --- Test Gradient ---
#

if __name__ == '__main__':
    from torch.autograd import gradcheck

    B, C, T = 2, 64, 12
    # device = torch.device("cuda")
    device = torch.device("cpu")
    X = torch.randn((B, C, T), dtype=torch.double, device=device, requires_grad=True)
    W1 = torch.rand((B, 1, T), dtype=torch.double, device=device, requires_grad=True)
    T1 = torch.rand((B, 1, T), dtype=torch.double, device=device, requires_grad=True)
    W2 = torch.rand((1, 1, T), dtype=torch.double, device=device, requires_grad=True)
    T2 = torch.rand((1, 1, T), dtype=torch.double, device=device, requires_grad=True)

    print("Foward test of WeightedLeastSquaresFcn...")
    f = WeightedLeastSquaresFcn(beta=0.0).apply
    y, y0 = f(X, T1, torch.ones_like(W1))
    if torch.allclose(torch.einsum("bnm,bn->bm", X, y) + y0, T1.view(B, T), atol=1.0e-5, rtol=1.0e-3):
        print("Passed")
    else:
        print("Failed")
        print(torch.einsum("bnm,bn->bm", X, y) + y0 - T1.view(B, T))

    ytilde, y0tilde = f(X, T1)
    if torch.allclose(ytilde, y) and torch.allclose(y0tilde, y0):
        print("Passed")
    else:
        print("Failed")
        print(y - ytilde)
        print(y0 - y0tilde)

    print("Gradient test on WeightedLeastSquaresFcn...")
    f = WeightedLeastSquaresFcn(cache_decomposition=False).apply
    test = gradcheck(f, (X, T1, W1), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)
    test = gradcheck(f, (X, T2, W2), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    f = WeightedLeastSquaresFcn(cache_decomposition=True).apply
    test = gradcheck(f, (X, T1, W1), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)
    test = gradcheck(f, (X, T2, W2), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    print("Gradient test on (unweighted) WeightedLeastSquaresFcn...")
    f = WeightedLeastSquaresFcn(cache_decomposition=False).apply
    test = gradcheck(f, (X, T1), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)
    test = gradcheck(f, (X, T2), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    f = WeightedLeastSquaresFcn(cache_decomposition=True).apply
    test = gradcheck(f, (X, T1), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)
    test = gradcheck(f, (X, T2), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)
