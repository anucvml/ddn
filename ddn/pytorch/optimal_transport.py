#!/usr/bin/env python
#
# OPTIMAL TRANSPORT NODE
# Implementation of differentiable optimal transport using implicit differentiation. Makes use of Sinkhorn normalization
# to solve the entropy regularized problem (Cuturi, NeurIPS 2013) in the forward pass. The problem can be written as
# Let us write the entropy regularized optimal transport problem in the following form,
#
#    minimize (over P) <P, M> + 1/\gamma KL(P || rc^T)
#    subject to        P1 = r and P^T1 = c
#
# where r and c are m- and n-dimensional vectors, respectively, each summing to one. Here m-by-n matrix M is the input
# and positive m-by-n dimensional matrix P is the output. The above problem leads to a solution of the form
#
#   P_{ij} = \alpha_i \beta_j e^{-\gamma M_{ij}}
#
# where \alpha and \beta are found by iteratively applying row and column normalizations.
#
# See accompanying Jupyter Notebook at https://deepdeclarativenetworks.com.
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
#

import torch
import torch.nn as nn


def sinkhorn(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000):
    """
    PyTorch function for entropy regularized optimal transport. Assumes batched inputs as follows:
        M:  (B,H,W) tensor
        r:  (B,H) tensor, (1,H) tensor or None for constant uniform vector 1/H
        c:  (B,W) tensor, (1,W) tensor or None for constant uniform vector 1/W

    You can back propagate through this function in O(TBWH) time where T is the number of iterations taken to converge.
    """

    B, H, W = M.shape
    if r is None: r = torch.ones((1, H), device=M.device, dtype=M.dtype) / H
    if c is None: c = torch.ones((1, W), device=M.device, dtype=M.dtype) / W

    assert r.shape == (B, H) or r.shape == (1, H)
    assert c.shape == (B, W) or c.shape == (1, W)

    P = torch.exp(-1.0 * gamma * M)
    for i in range(maxiters):
        alpha = torch.sum(P, 2).reshape(B, H)
        if torch.all(torch.isclose(alpha, r, atol=eps, rtol=0.0)):
            break
        P = (r / alpha).view(B, H, 1) * P

        beta = torch.sum(P, 1).reshape(B, W)
        if torch.all(torch.isclose(beta, c, atol=eps, rtol=0.0)):
            break
        P = P * (c / beta).view(B, 1, W)

    return P


class OptimalTransportFcn(torch.autograd.Function):
    """
    PyTorch autograd function for entropy regularized optimal transport. Assumes batched inputs as follows:
        M:  (B,H,W) tensor
        r:  (B,H) tensor, (1,H) tensor or None for constant uniform vector
        c:  (B,W) tensor, (1,W) tensor or None for constant uniform vector

    Allows for approximate gradient calculations, which is faster and may be useful during early stages of learning,
    when exp(-\gamma M) is already nearly doubly stochastic, or when gradients are otherwise noisy.
    """

    @staticmethod
    def forward(ctx, M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000, approx_grad=False, block_inverse=True):
        """Solve optimal transport using skinhorn."""

        with torch.no_grad():
            P = sinkhorn(M, r, c, gamma, eps, maxiters)

        ctx.save_for_backward(M, r, c, P)
        ctx.gamma = gamma
        ctx.approx_grad = approx_grad
        ctx.block_inverse = block_inverse

        return P

    @staticmethod
    def backward(ctx, dJdP):
        """Implement backward pass using implicit differentiation."""

        M, r, c, P = ctx.saved_tensors
        B, H, W = M.shape

        if r is None: r = torch.ones((1, H), device=M.device, dtype=M.dtype) / H
        if c is None: c = torch.ones((1, W), device=M.device, dtype=M.dtype) / W

        # initialize backward gradients (-v^T H^{-1} B with v = dJdP and B = I)
        dJdM = -1.0 * ctx.gamma * P * dJdP

        # return approximate gradients
        if ctx.approx_grad:
            return dJdM, None, None, None, None, None, None, None

        # compute v^T H^{-1} A^T as two blocks
        vHAt1 = torch.sum(dJdM[:, 1:H, 0:W], dim=2)
        vHAt2 = torch.sum(dJdM, dim=1)

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1}
        if ctx.block_inverse:
            # by block inverse of (A H^{-1] A^T)
            block_11 = torch.cholesky(torch.diag_embed(r[:, 1:H]) -
                torch.einsum("bij,bkj->bik", P[:, 1:H, 0:W], P[:, 1:H, 0:W] / c.view(c.shape[0], 1, W)))
            block_12 = torch.cholesky_solve(P[:, 1:H, 0:W] / c.view(c.shape[0], 1, W), block_11)
            block_22 = torch.diag_embed(1.0 / c) + torch.einsum("bji,bjk->bik", block_12, P[:, 1:H, 0:W] / c.view(c.shape[0], 1, W))

            v1 = torch.cholesky_solve(vHAt1.view(B, H-1, 1), block_11).view(B, H-1) - torch.einsum("bi,bji->bj", vHAt2, block_12)
            v2 = torch.einsum("bi,bij->bj", vHAt2, block_22) - torch.einsum("bi,bij->bj", vHAt1, block_12)

        else:
            # by full inverse of (A H^{-1] A^T)
            AinvHAt = torch.empty((B, H + W - 1, H + W - 1), device=M.device, dtype=M.dtype)
            AinvHAt[:, 0:H - 1, 0:H - 1] = torch.diag_embed(r[:, 1:H])
            AinvHAt[:, H - 1:H + W - 1, H - 1:H + W - 1] = torch.diag_embed(c)
            AinvHAt[:, 0:H - 1, H - 1:H + W - 1] = P[:, 1:H, 0:W]
            AinvHAt[:, H - 1:H + W - 1, 0:H - 1] = P[:, 1:H, 0:W].transpose(1, 2)

            v = torch.einsum("bi,bij->bj", torch.cat((vHAt1, vHAt2), dim=1), torch.inverse(AinvHAt))
            v1 = v[:, 0:H-1]
            v2 = v[:, H-1:H+W-1]

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1} A H^{-1} - v^T H^{-1} B
        dJdM[:, 1:H, 0:W] -= v1.view(B, H-1, 1) * P[:, 1:H, 0:W]
        dJdM -= v2.view(B, 1, W) * P

        # TODO: gradient wrt r
        # TODO: gradient wrt c

        # return gradients (None for gamma, eps, and maxiters)
        return dJdM, None, None, None, None, None, None, None


class OptimalTransportLayer(nn.Module):
    """Neural network layer to implement optimal transport."""

    def __init__(self, gamma=1.0, eps=1.0e-6, maxiters=1000, approx_grad=False, block_inverse=True):
        super(OptimalTransportLayer, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.maxiters = maxiters
        self.approx_grad = approx_grad
        self.block_inverse = block_inverse

    def forward(self, M, r=None, c=None):
        return OptimalTransportFcn.apply(M, r, c, self.gamma, self.eps, self.maxiters, self.approx_grad, self.block_inverse)


#
# --- testing ---
#

if __name__ == '__main__':

    from torch.autograd import gradcheck

    M = torch.randn((2, 5, 7), dtype=torch.double, requires_grad=True)
    f = OptimalTransportFcn().apply
    test = gradcheck(f, (M, None, None, 1.0, 1.0e-6, 1000, False, True), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    f = OptimalTransportFcn().apply
    test = gradcheck(f, (M, None, None, 1.0, 1.0e-6, 1000, False, False), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    f = OptimalTransportFcn().apply
    test = gradcheck(f, (M, None, None, 10.0, 1.0e-6, 1000, False, True), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    f = OptimalTransportFcn().apply
    test = gradcheck(f, (M, None, None, 10.0, 1.0e-6, 1000, False, False), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)
