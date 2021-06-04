#!/usr/bin/env python
#
# LEAST SQUARES NODES
# Implementation of differentiable weighted and unweighted least squares nodes. Can be used for rank pooling operations
# as well as many other tasks. See accompanying Jupyter Notebook tutorial at https://deepdeclarativenetworks.com.
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import torch
import torch.nn as nn
import math

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
    def forward(ctx, input, target, weights=None, beta=1.0e-3, cache_decomposition=False, enable_bias=True, inverse_mode='cholesky'):
        # allocate output tensors
        B, C, T = input.shape
        assert target.shape == (B, 1, T) or target.shape == (1, 1, T), "{} vs {}".format(input.shape, target.shape)
        assert weights is None or weights.shape == (B, 1, T) or weights.shape == (1, 1, T), "{} vs {}".format(input.shape, weights.shape)

        inverse_mode = inverse_mode.lower()
        U_sz = C + 1 if (enable_bias) else C  # H = DDf/DYDY is in R^{(n+1)*(n+1)} if enable_bias; otherwise, in R^{n*n}, where n=C.

        with torch.no_grad():
            # replicate across batch if sharing weights or target
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
            L, R = None, None
            if inverse_mode == 'qr':  # need to get A for A=QR
                weightedsqrt = torch.ones_like(target).view(B, -1) if (weights is None) else torch.sqrt(weights).view(B, -1)
                weightedsqrtX = input if (weights is None) else torch.einsum("bnm,bm->bnm", input, weightedsqrt)
                weightedsqrtT = target.view(B, -1) if (weights is None) else torch.einsum("bm,bm->bm", target.view(B, -1), weightedsqrt).view(B, -1)
                A = torch.empty((B, U_sz, C + T), device=input.device, dtype=input.dtype)
                b = torch.cat((weightedsqrtT, torch.zeros(B, C)), 1).view(B, C + T)

                # solve x = (R)^{-1} Q^T b
                if enable_bias:
                    A[:, :C, :T] = weightedsqrtX
                    A[:, -1, :T] = weightedsqrt
                    A[:, :C, T:] = math.sqrt(beta) * torch.eye(C, device=input.device, dtype=input.dtype)
                    A[:, -1, T:] = torch.zeros((B, C), device=input.device, dtype=input.dtype)
                else:
                    A[:, :C, :T] = weightedsqrtX
                    A[:, :C, T:] = math.sqrt(beta) * torch.eye(C, device=input.device, dtype=input.dtype)

                Q, R = torch.qr(A.permute(0, 2, 1), some=True)
                Qtb = torch.einsum("bij,bi->bj", Q, b).view(B, -1, 1)
                y, _ = torch.solve(Qtb, R)

                R = R if cache_decomposition else None
            else:  # need to get AtA
                weightedX = input if weights is None else torch.einsum("bnm,bm->bnm", input, weights.view(B, -1))
                weightedTSum = target.sum(2).view(B, 1) if weights is None else torch.einsum("bm,bm->b", target.view(B, -1), weights.view(B, -1)).view(B, 1)
                weightedXdotT = torch.einsum("bnm,bm->bn", weightedX, target.view(B, -1))

                # solve x = (A^TA)^{-1} A^T b
                if enable_bias:
                    AtA = torch.empty((B, U_sz, U_sz), device=input.device, dtype=input.dtype)
                    AtA[:, -1, -1] = T if weights is None else torch.sum(weights.view(B, -1), 1)
                    AtA[:, :C, :C] = torch.einsum("bik,bjk->bij", weightedX, input) + \
                                     (beta * torch.eye(C, device=input.device, dtype=input.dtype)).view(1, C, C)
                    AtA[:, :C, -1] = AtA[:, -1, :C] = torch.sum(weightedX, 2)
                    Atb = torch.cat((weightedXdotT, weightedTSum), 1).view(B, U_sz, 1)
                else:
                    AtA = torch.einsum("bik,bjk->bij", weightedX, input) + \
                          (beta * torch.eye(C, device=input.device, dtype=input.dtype)).view(1, C, C)
                    Atb = weightedXdotT.view(B, U_sz, 1)

                if cache_decomposition:
                    L = torch.cholesky(AtA)
                    y = torch.cholesky_solve(Atb, L)
                else:
                    y, _ = torch.solve(Atb, AtA)

            # assign to output
            output = y[:, :C, 0].squeeze(-1)
            bias = y[:, C, 0].view(B, 1) if enable_bias else torch.zeros((B, 1), device=input.device, dtype=input.dtype)

        # save state for backward pass
        ctx.save_for_backward(input, target, weights, output, bias, L, R)
        ctx.beta = beta
        ctx.enable_bias = enable_bias
        ctx.inverse_mode = inverse_mode

        # return rank pool vector and bias
        return output, bias

    @staticmethod
    def backward(ctx, grad_output, grad_bias):
        # check for None tensors
        if grad_output is None and grad_bias is None:
            return None, None

        # unpack cached tensors
        input, target, weights, output, bias, L, R = ctx.saved_tensors
        enable_bias = ctx.enable_bias
        inverse_mode = ctx.inverse_mode
        B, C, T = input.shape
        U_sz = C + 1 if enable_bias else C
        weightedX = input if weights is None else torch.einsum("bnm,bm->bnm", input, weights.view(B, -1))

        # solve for w = (R^TR)^{-1} v for QR; w = (A^TA)^{-1} v for others
        if enable_bias:
            v = torch.cat((grad_output, grad_bias), 1).view(B, U_sz, 1)
        else:
            v = grad_output.view(B, U_sz, 1)

        if inverse_mode == 'qr':
            if R is None:
                weightedsqrt = torch.ones_like(target).view(B, -1) if (weights is None) else torch.sqrt(weights).view(B, -1)
                weightedsqrtX = input if weights is None else torch.einsum("bnm,bm->bnm", input, weightedsqrt)
                A = torch.empty((B, U_sz, C + T), device=input.device, dtype=input.dtype)

                if enable_bias:
                    A[:, :C, :T] = weightedsqrtX
                    A[:, -1, :T] = weightedsqrt
                    A[:, :C, T:] = math.sqrt(ctx.beta) * torch.eye(C, device=input.device, dtype=input.dtype)
                    A[:, -1, T:] = torch.zeros((B, C), device=input.device, dtype=input.dtype)
                else:
                    A[:, :C, :T] = weightedsqrtX
                    A[:, :C, T:] = math.sqrt(ctx.beta) * torch.eye(C, device=input.device, dtype=input.dtype)

                _, R = torch.qr(A.permute(0, 2, 1), some=True)

            w, _ = torch.solve(v, torch.einsum("bij,bik->bjk", R, R))
        else:
            if L is None:
                if enable_bias:
                    AtA = torch.empty((B, U_sz, U_sz), device=input.device, dtype=input.dtype)
                    AtA[:, -1, -1] = T if weights is None else torch.sum(weights.view(B, -1), 1)
                    AtA[:, :C, :C] = torch.einsum("bik,bjk->bij", weightedX, input) + \
                        (ctx.beta * torch.eye(C, device=input.device, dtype=input.dtype)).view(1, C, C)
                    AtA[:, :C, -1] = AtA[:, -1, :C] = torch.sum(weightedX, 2)
                else:
                    AtA= torch.einsum("bik,bjk->bij", weightedX, input) + \
                         (ctx.beta * torch.eye(C, device=input.device, dtype=input.dtype)).view(1, C, C)

                w, _ = torch.solve(v, AtA)
            else:
                w = torch.cholesky_solve(v, L)

        # compute w^T B
        grad_weights = None
        if enable_bias:
            bias = bias.view(B, 1)
            w_bias = w[:, C].view(B, 1)
        else:
            bias, w_bias = 0.0, 0.0

        if weights is not None:
            grad_input = w[:, :C].view(B, C, 1) * torch.mul(weights,
                (target.view(B, T) - torch.einsum("bn,bnm->bm", output.view(B, C), input) - bias).view(B, 1, T)) - \
                torch.mul(weights, (torch.einsum("bn,bnm->bm", w[:, :C].view(B, C), input) + w_bias).view(B, 1, T)) * \
                output.view(B, C, 1)

            grad_target = (torch.einsum("bn,bnm->bm", w[:, :C].view(B, C), weightedX) +
                w_bias * weights.view(B, T)).view(B, 1, T)

            grad_weights = ((target.view(B, T) - torch.einsum("bn,bnm->bm", output.view(B, C), input) - bias) *
                (torch.einsum("bn,bnm->bm", w[:, :C].view(B, C), input) + w_bias)).view(B, 1, T)
        else:
            grad_input = w[:, :C].view(B, C, 1) * \
                (target.view(B, T) - torch.einsum("bn,bnm->bm", output.view(B, C), input) - bias).view(B, 1, T) - \
                (torch.einsum("bn,bnm->bm", w[:, :C].view(B, C), input) + w_bias).view(B, 1, T) * \
                output.view(B, C, 1)

            grad_target = (torch.einsum("bn,bnm->bm", w[:, :C].view(B, C), weightedX) + w_bias).view(B, 1, T)

        if ctx.collapse_target:
            grad_target = torch.sum(grad_target, 0, keepdim=True)
        if ctx.collapse_weights:
            grad_weights = torch.sum(grad_weights, 0, keepdim=True)

        # return gradients (None for `beta`, `cache_decomposition`, 'enable_bias', 'inverse_mode')
        return grad_input, grad_target, grad_weights, None, None, None, None


#
# --- PyTorch Modules ---
#

class LeastSquaresLayer(nn.Module):
    """Neural network layer to implement (unweighted) least squares fitting."""

    def __init__(self, beta=1.0e-3, cache_decomposition=False, enable_bias=True, inverse_mode='cholesky'):
        super(LeastSquaresLayer, self).__init__()
        self.beta = beta
        self.cache_decomposition = cache_decomposition
        self.enable_bias = enable_bias
        self.inverse_mode = inverse_mode

    def forward(self, input, target):
        return WeightedLeastSquaresFcn.apply(input, target, None, self.beta, self.cache_decomposition,
                                             self.enable_bias, self.inverse_mode)

class WeightedLeastSquaresLayer(nn.Module):
    """Neural network layer to implement weighted least squares fitting."""

    def __init__(self, beta=1.0e-3, cache_decomposition=False, enable_bias=True, inverse_mode='cholesky'):
        super(WeightedLeastSquaresLayer, self).__init__()
        self.beta = beta
        self.cache_decomposition = cache_decomposition
        self.enable_bias = enable_bias
        self.inverse_mode = inverse_mode

    def forward(self, input, target, weights):
        return WeightedLeastSquaresFcn.apply(input, target, weights, self.beta, self.cache_decomposition,
                                             self.enable_bias, self.inverse_mode)

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
    f = WeightedLeastSquaresFcn.apply

    for inverse_mode in ['cholesky', 'qr']:
        for enable_bias in [True, False]:
            # Forward check
            print("Foward test of WeightedLeastSquaresFcn, mode: {}, bias: {}...".format(inverse_mode, enable_bias))
            y, y0 = f(X, T1, torch.ones_like(W1), 1.0e-6, False, enable_bias, inverse_mode)
            if torch.allclose(torch.einsum("bnm,bn->bm", X, y) + y0, T1.view(B, T), atol=1.0e-5, rtol=1.0e-3):
                print("Passed")
            else:
                print("Failed")
                print(torch.einsum("bnm,bn->bm", X, y) + y0 - T1.view(B, T))

            ytilde, y0tilde = f(X, T1, None, 1.0e-6, False, enable_bias, inverse_mode)
            if torch.allclose(ytilde, y) and torch.allclose(y0tilde, y0):
                print("Passed")
            else:
                print("Failed")
                print(y - ytilde)
                print(y0 - y0tilde)

            # Backward check
            print("Gradient test on WeightedLeastSquaresFcn, mode: {}, bias: {}...".format(inverse_mode, enable_bias))
            test = gradcheck(f, (X, T1, W1, 1.0e-3, False, enable_bias, inverse_mode), eps=1e-6, atol=1e-3, rtol=1e-6)
            print(test)
            test = gradcheck(f, (X, T2, W2, 1.0e-3, False, enable_bias, inverse_mode), eps=1e-6, atol=1e-3, rtol=1e-6)
            print(test)

            f = WeightedLeastSquaresFcn.apply
            test = gradcheck(f, (X, T1, W1, 1.0e-3, True, enable_bias, inverse_mode), eps=1e-6, atol=1e-3, rtol=1e-6)
            print(test)
            test = gradcheck(f, (X, T2, W2, 1.0e-3, True, enable_bias, inverse_mode), eps=1e-6, atol=1e-3, rtol=1e-6)
            print(test)

            print("Gradient test on (unweighted) WeightedLeastSquaresFcn, mode: {}, bias: {}...".format(inverse_mode, enable_bias))
            f = WeightedLeastSquaresFcn.apply
            test = gradcheck(f, (X, T1, None, 1.0e-3, False, enable_bias, inverse_mode), eps=1e-6, atol=1e-3, rtol=1e-6)
            print(test)
            test = gradcheck(f, (X, T2, None, 1.0e-3, False, enable_bias, inverse_mode), eps=1e-6, atol=1e-3, rtol=1e-6)
            print(test)

            f = WeightedLeastSquaresFcn.apply
            test = gradcheck(f, (X, T1, None, 1.0e-3, True, enable_bias, inverse_mode), eps=1e-6, atol=1e-3, rtol=1e-6)
            print(test)
            test = gradcheck(f, (X, T2, None, 1.0e-3, True, enable_bias, inverse_mode), eps=1e-6, atol=1e-3, rtol=1e-6)
            print(test)

    print("Foward test of WeightedLeastSquaresFcn bias True vs False...")
    y_true, y0_true = f(X, T1, torch.ones_like(W1), 1.0e-6, False, True)
    y_false, y0_false = f(X, T1, None, 1.0e-6, False, False)
    if torch.allclose(y_true, y_false) and torch.allclose(y0_true, y0_false) and (y0_false != 0.0):
        print("Failed")
        print(y_true, y_false)
        print(y0_true, y0_false)
    else:
        print("Passed")

    print("Foward test of WeightedLeastSquaresFcn mode Cholesky vs QR...")
    y_chol, y0_chol = f(X, T1, None, 1.0e-6, False, True, 'cholesky')
    y_qr, y0_qr = f(X, T1, None, 1.0e-6, False, True, 'qr')
    if torch.allclose(y_chol, y_qr) and torch.allclose(y0_chol, y0_qr):
        print("Passed")
    else:
        print("Failed")
        print(y_chol, y_qr)
        print(y0_chol, y0_qr)
