# DIFFERENTIABLE TRIDIAGONAL MATRIX SOLVER
# Differentiable solver for batches of tridiagonal linear systems.
#
# Stephen Gould <stephen.gould@anu.edu.au>
#

import torch

def tridiagsolve(b, a, c, d, method='cyclic'):
    """
    Batch tridiagonal matrix algorithm based on either cyclic reduction (method='cyclic') or the Thomas algorithm
    (method='thomas') to solve systems of the form,

        | b_1 c_1     ...  0  | | x_1 |   | d_1 |
        | a_2 b_2 c_2 ...  0  | | x_2 |   | d_2 |
        |        .            | |     |   |  .  |
        |           .         | |     | = |  .  |
        |              .      | |     |   |  .  |
        |             ... b_n | | x_n |   | d_n |

    The Thomas algorithm is an efficient implementation of Gaussian elimination and good for single-threaded hardware
    or on large batches of data. The input matrix needs to be diagonally dominant or positive semi-definite for stable
    behaviour.
    Cyclic reduction recursively eliminates odd and even variables to produce two independent problems. It is good for
    multi-threaded hardware or small batches with large sequence lengths.

    :param b: main diagonal, (b_1, ..., b_n), of size (B x N)
    :param a: lower diagonal, (a_2, ..., a_n), of size (B x N-1)
    :param c: upper diagonal, (c_1, ..., c_{n-1}), of size (B x N-1)
    :param d: right-hand-size, (d_1, ..., d_n), of size (B x N x M)
    :param method: 'cyclic' or 'thomas'
    :return: x = (x_1, ..., x_n) of size (B x N x M)
    """

    assert len(d.shape) == 3, "argument 'd' must have shape (B, N, M) or (B, N, 1)"
    B, N, M = d.shape
    assert b.shape == (B, N), "argument 'b' must have shape (B, N)"
    assert a.shape == (B, N-1), "argument 'a' must have shape (B, N-1)"
    assert c.shape == (B, N-1), "argument 'c' must have shape (B, N-1)"

    if method == 'cyclic':
        # initialize
        a_dash, c_dash = a, c
        b_dash, d_dash = b.clone(), d.clone()

        # repeat until problems of size 1
        h = 1
        while (h < N):
            # eliminate odd/even terms
            alpha = -1.0 * a_dash / b_dash[:, :-h]      # i = h/2+2, ..., n-h/2
            beta = -1.0 * c_dash / b_dash[:, h:]        # i = h/2+1, ..., n-h/2-1

            b_dash[:, h:] += alpha * c_dash             # i = h/2+1, ..., n-h/2
            b_dash[:, :-h] += beta * a_dash             # i = h/2+2, ..., n-h/2-1
            d_prev = d_dash.clone()                     # i = h/2+1, ..., n-h/2
            d_dash[:, h:, :] += alpha.view(B, N-h, 1) * d_prev[:, :-h, :]    # i = h/2+2, ..., n-h/2
            d_dash[:, :-h, :] += beta.view(B, N-h, 1) * d_prev[:, h:, :]     # i = h/2+1, ..., n-h/2-1

            if (h < alpha.shape[1]):
                a_dash = alpha[:, h:] * a_dash[:, :-h]  # i = h/2+1, ..., n-h/2
                c_dash = beta[:, :-h] * c_dash[:, h:]   # i = h/2+1, ..., n-h/2

            h *= 2

        # solve
        return d_dash / b_dash.view(B, N, 1)

    elif method == 'thomas':

        # initialize
        x = torch.empty_like(d)
        c_dash = torch.empty_like(c)
        d_dash = torch.empty_like(d)

        # forward elimination
        c_dash[:, 0] = c[:, 0] / b[:, 0]
        d_dash[:, 0, :] = d[:, 0, :] / b[:, 0].view(B, 1)

        for i in range(1, N-1):
            w = b[:, i] - a[:, i-1] * c_dash[:, i-1]
            c_dash[:, i] = c[:, i] / w
            d_dash[:, i, :] = (d[:, i, :] - a[:, i-1].view(B, 1) * d_dash[:, i-1, :]) / w.view(B, 1)

        w = b[:, N-1] - a[:, N-2] * c_dash[:, N-2]
        d_dash[:, N-1, :] = (d[:, N-1, :] - a[:, N-2].view(B, 1) * d_dash[:, N-2, :]) / w.view(B, 1)

        # backward substitution
        x[:, N-1, :] = d_dash[:, N-1, :]
        for i in range(N-1, 0, -1):
            x[:, i-1, :] = d_dash[:, i-1, :] - c_dash[:, i-1].view(B, 1) * x[:, i, :]

        return x

    else:
        raise NameError("unknown method '{}'".format(method))


class TriDiagSolveFcn(torch.autograd.Function):
    """
    Differentiable tridiagonal matrix solver. See `tridiagsolve`.
    """

    @staticmethod
    def forward(ctx, b, a, c, d, method='cyclic'):
        with torch.no_grad():
            x = tridiagsolve(b, a, c, d, method)
        ctx.save_for_backward(b, a, c, d, x)
        ctx.method = method
        return x

    @staticmethod
    def backward(ctx, grad_x):
        b, a, c, d, x = ctx.saved_tensors

        w = tridiagsolve(b, c, a, grad_x, ctx.method)
        grad_b = -1.0 * torch.sum(w * x, 2) if ctx.needs_input_grad[0] else None
        grad_a = -1.0 * torch.sum(w[:, 1:] * x[:, :-1], 2) if ctx.needs_input_grad[1] else None
        grad_c = -1.0 * torch.sum(w[:, :-1] * x[:, 1:], 2) if ctx.needs_input_grad[2] else None
        grad_d = w if ctx.needs_input_grad[3] else None

        return grad_b, grad_a, grad_c, grad_d, None


#
# --- testing ---
#

if __name__ == '__main__':

    B, N, M = 2, 16, 5
    type = torch.float64
    device = torch.device("cpu")

    # arbitrary
    b = 2.0 * torch.ones((B, N), dtype=type, device=device) + 0.1 * torch.rand((B, N), dtype=type, device=device, requires_grad=True)
    a = -1.0 * torch.rand((B, N - 1), dtype=type, device=device, requires_grad=True)
    c = -1.0 * torch.rand((B, N - 1), dtype=type, device=device, requires_grad=True)
    d = torch.rand((B, N, M), dtype=type, device=device, requires_grad=True)

    print("Checking implementation accuracy on arbitrary input...")
    print(d.shape)
    A = torch.diag_embed(b) + torch.diag_embed(a, offset=-1) + torch.diag_embed(c, offset=1)

    x = tridiagsolve(b, a, c, d, 'cyclic')
    print(torch.max(torch.abs(A @ x - d)).item())

    x = tridiagsolve(b, a, c, d, 'thomas')
    print(torch.max(torch.abs(A @ x - d)).item())

    # poisson
    print("Checking implementation accuracy on poisson input...")
    neg_ones = -1.0 * torch.ones((B, N-1), dtype=type, device=device)
    A = torch.diag_embed(b) + torch.diag_embed(neg_ones, offset=-1) + torch.diag_embed(neg_ones, offset=1)

    x = tridiagsolve(b, neg_ones, neg_ones, d, 'cyclic')
    print(torch.max(torch.abs(A @ x - d)).item())

    x = tridiagsolve(b, neg_ones, neg_ones, d, 'thomas')
    print(torch.max(torch.abs(A @ x - d)).item())

    #exit(0)

    from torch.autograd import gradcheck

    torch.manual_seed(22)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")

    B, N, M = 2, 16, 5
    type = torch.float64
    
    b = 2.0 * torch.ones((B, N), dtype=type, device=device) + 0.1 * torch.rand((B, N), dtype=type, device=device, requires_grad=True)
    a = -1.0 * torch.rand((B, N-1), dtype=type, device=device, requires_grad=True)
    c = -1.0 * torch.rand((B, N-1), dtype=type, device=device, requires_grad=True)
    d = torch.rand((B, N, M), dtype=type, device=device, requires_grad=True)

    print("Checking gradients (cyclic)...")
    test = gradcheck(TriDiagSolveFcn().apply, (b, a, c, d, 'cyclic'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    print("Checking gradients (thomas)...")
    test = gradcheck(TriDiagSolveFcn().apply, (b, a, c, d, 'thomas'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    #exit(0)

    import time
    print("Testing running time...")

    N = 256
    type = torch.float32

    for B in (1, 10, 100, 1000):
        b = 2.0 * torch.ones((B, N), dtype=type, device=device) + 0.1 * torch.rand((B, N), dtype=type, device=device, requires_grad=True)
        #a = -1.0 * torch.ones((B, N-1), dtype=type, device=device, requires_grad=True)
        #c = -1.0 * torch.ones((B, N-1), dtype=type, device=device, requires_grad=True)
        a = -1.0 * torch.rand((B, N - 1), dtype=type, device=device, requires_grad=True)
        c = -1.0 * torch.rand((B, N - 1), dtype=type, device=device, requires_grad=True)
        d = torch.rand((B, N, 1), dtype=type, device=device, requires_grad=True)

        print("...data size {}".format(d.shape))

        start = time.time()
        x = TriDiagSolveFcn.apply(b.clone().detach().requires_grad_(True), a.clone().detach().requires_grad_(True),
                                  c.clone().detach().requires_grad_(True), d.clone().detach().requires_grad_(True), 'cyclic')
        x_elapsed = time.time() - start

        start = time.time()
        x.sum().backward()
        dx_elapsed = time.time() - start

        start = time.time()
        y = TriDiagSolveFcn.apply(b.clone().detach().requires_grad_(True), a.clone().detach().requires_grad_(True),
                                  c.clone().detach().requires_grad_(True), d.clone().detach().requires_grad_(True), 'thomas')
        y_elapsed = time.time() - start

        start = time.time()
        y.sum().backward()
        dy_elapsed = time.time() - start

        A = torch.diag_embed(b.clone().detach().requires_grad_(True)) + \
            torch.diag_embed(a.clone().detach().requires_grad_(True), offset=-1) + \
            torch.diag_embed(c.clone().detach().requires_grad_(True), offset=1)
        start = time.time()
        z = torch.linalg.solve(A, d)
        z_elapsed = time.time() - start
        print(z.shape)

        start = time.time()
        z.sum().backward()
        dz_elapsed = time.time() - start

        print('accuracy cyclic/thomas/linalg.solve: {:.3e}/{:.3e}/{:.3e}'.format(torch.max(torch.abs(A @ x - d)).item(),
            torch.max(torch.abs(A @ y - d)).item(), torch.max(torch.abs(A @ z - d)).item()))
        print(' forward cyclic/thomas/linalg.solve: {:.3e}/{:.3e}/{:.3e}'.format(x_elapsed, y_elapsed, z_elapsed))
        print('backward cyclic/thomas/linalg.solve: {:.3e}/{:.3e}/{:.3e}'.format(dx_elapsed, dy_elapsed, dz_elapsed))
