# Differentiation Eigen (Spectral) Decomposition
# Stephen Gould <stephen.gould@anu.edu.au>
#

import torch

class EigenDecompositionFcn(torch.autograd.Function):
    """PyTorch autograd function for eigen decomposition of real symmetric matrices. Returns all eigenvectors
    or just eigenvectors associated with the top-k eigenvalues. The input matrix is made symmetric within the
    forward evaluation function."""

    eps = 1.0e-9 # tolerance to consider two eigenvalues equal

    @staticmethod
    def forward(ctx, X, top_k=None):
        B, M, N = X.shape
        assert N == M
        assert (top_k is None) or (1 <= top_k <= M)

        with torch.no_grad():
            lmd, Y = torch.linalg.eigh(0.5 * (X + X.transpose(1, 2)))

        ctx.save_for_backward(lmd, Y)
        return Y if top_k is None else Y[:, :, -top_k:]

    @staticmethod
    def backward(ctx, dJdY):
        lmd, Y = ctx.saved_tensors
        B, M, K = dJdY.shape

        zero = torch.zeros(1, dtype=lmd.dtype, device=lmd.device)
        L = lmd[:, -K:].view(B, 1, K) - lmd.view(B, M, 1)
        torch.where(torch.abs(L) < EigenDecompositionFcn.eps, zero, 1.0 / L, out=L)
        dJdX = torch.bmm(torch.bmm(Y, L * torch.bmm(Y.transpose(1, 2), dJdY)), Y[:, :, -K:].transpose(1, 2))

        dJdX = 0.5 * (dJdX + dJdX.transpose(1, 2))

        return dJdX, None


#
# --- Test Gradient ---
#

if __name__ == '__main__':
    from torch.autograd import gradcheck

    for m in (5, 8, 16):
        f = EigenDecompositionFcn
        X = torch.randn((3, m, m), dtype=torch.double, requires_grad=True)
        X = 0.5 * (X + X.transpose(1, 2))
        test = gradcheck(f().apply, (X, None), eps=1e-6, atol=1e-3, rtol=1e-6)
        print("{}(X, None): {}".format(f.__name__, test))
        for n in range(1, m + 1):
            test = gradcheck(f().apply, (X, n), eps=1e-6, atol=1e-3, rtol=1e-6)
            print("{}(X, {}): {}".format(f.__name__, n, test))