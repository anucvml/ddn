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
        |  0          ... b_n | | x_n |   | d_n |

    The Thomas algorithm is an efficient implementation of Gaussian elimination and good for single-threaded hardware
    or on large batches of data. The input matrix needs to be diagonally dominant or positive semi-definite for stable
    behaviour.
    Cyclic reduction recursively eliminates odd and even variables to produce two independent problems. It is good for
    multi-threaded hardware or small batches with large sequence lengths.
    The function can also solve the system using `torch.linalg.solve` (method='linalg'), which is useful for testing.

    :param b: main diagonal, (b_1, ..., b_n), of size (B x N)
    :param a: lower diagonal, (a_2, ..., a_n), of size (B x N-1)
    :param c: upper diagonal, (c_1, ..., c_{n-1}), of size (B x N-1)
    :param d: right-hand-size, (d_1, ..., d_n), of size (B x N x K) or (B x N x 1)
    :param method: 'cyclic' or 'thomas' or 'linalg'
    :return: x = (x_1, ..., x_n) of size (B x N x K) or (B x N x 1)
    """

    assert len(d.shape) == 3, "argument 'd' must have shape (B, N, K) or (B, N, 1)"
    B, N, K = d.shape
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
        #x = torch.empty_like(d)
        #x[:, N-1, :] = d_dash[:, N-1, :]
        #for i in range(N-1, 0, -1):
        #    x[:, i-1, :] = d_dash[:, i-1, :] - c_dash[:, i-1].view(B, 1) * x[:, i, :]
        x = d_dash
        for i in range(N-1, 0, -1):
            x[:, i-1, :] -= c_dash[:, i-1].view(B, 1) * x[:, i, :]

        return x

    elif method == 'linalg':
        A = torch.diag_embed(b) + torch.diag_embed(a, offset=-1) + torch.diag_embed(c, offset=1)
        return torch.linalg.solve(A, d)

    else:
        raise NameError("unknown method '{}'".format(method))


def blocktridiagsolve(b, a, c, d, method='thomas'):
    """
    Batch block tridiagonal matrix solver based on either cyclic reduction (method='cyclic'), the Thomas algorithm
    (method='thomas') or the standard PyTorch linear algebra solver (method='linalg') to solve systems of the form,

        | B_1 C_1     ...  0  | | x_1  |   | d_1  |
        | A_2 B_2 C_2 ...  0  | | x_2  |   | d_2  |
        |        .            | |      |   |  .   |
        |           .         | |      | = |  .   |
        |              .      | |      |   |  .   |
        |  0          ... B_n | | x_mn |   | d_mn |

    where B_i, A_i and C_i and m-by-m matrices. See `tridiagsolve` for the case of 1-by-1 blocks. The results of
    cyclic reduction may be inaccurate when the B_i are not very well conditioned. See Yalamov and Pavlov, "Stability
    of the Block Cyclic Reduction", Linear Algebra and its Applications, 1996 for a theoretical study. The Thomas
    algorithm appears to perform better and faster in our testing, and is recommended for the block tridiagonal case.

    :param b: main diagonal, (B_1, ..., B_n), of size (B x N x M x M)
    :param a: lower diagonal, (A_2, ..., A_n), of size (B x N-1 x M x M)
    :param c: upper diagonal, (C_1, ..., C_{n-1}), of size (B x N-1 x M x M)
    :param d: right-hand-size, (d_1, ..., d_mn), of size (B x MN x K) or (B x MN x 1)
    :param method: 'cyclic' or 'thomas' or 'linalg'
    :return: x = (x_1, ..., x_n) of size (B x MN x K) or (B x MN x 1)
    """

    assert len(d.shape) == 3, "argument 'd' must have shape (B, MN, K) or (B, MN, 1)"
    B, MN, K = d.shape
    assert len(b.shape) == 4, "argument 'b' must have shape (B, M, M, N)"
    M, N = b.shape[3], b.shape[1]
    assert M * N == MN, "argument 'd' must have shape (B, MN, K) or (B, MN, 1)"
    assert b.shape == (B, N, M, M), "argument 'b' must have shape (B, N, M, M)"
    assert a.shape == (B, N-1, M, M), "argument 'a' must have shape (B, N-1, M, M)"
    assert c.shape == (B, N-1, M, M), "argument 'c' must have shape (B, N-1, M, M)"

    if method == 'cyclic':
        # initialize
        a_dash, c_dash = a, c
        b_dash = b.clone()
        d_dash = d.clone().reshape(B, N, M, K)

        # repeat until problems of size 1
        h = 1
        while (h < N):
            # eliminate odd/even terms
            alpha = -1.0 * torch.linalg.solve(b_dash[:, :-h, :, :], a_dash, left=False)
            beta = -1.0 * torch.linalg.solve(b_dash[:, h:, :, :], c_dash, left=False)

            b_dash[:, h:, :, :] += alpha @ c_dash
            b_dash[:, :-h, :, :] += beta @ a_dash

            d_prev = d_dash.clone()
            d_dash[:, h:, :, :] += alpha @ d_prev[:, :-h, :, :]
            d_dash[:, :-h, :, :] += beta @ d_prev[:, h:, :, :]

            if (h < alpha.shape[1]):
                a_dash = alpha[:, h:, :, :] @ a_dash[:, :-h, :, :]
                c_dash = beta[:, :-h, :, :] @ c_dash[:, h:, :, :]

            h *= 2

        # solve
        return torch.linalg.solve(b_dash, d_dash).reshape(B, M * N, K)

    elif method == 'thomas':
        # initialize
        c_dash = torch.empty_like(c)
        d_dash = torch.empty_like(d)

        # forward elimination
        LU, pivots = torch.linalg.lu_factor(b[:, 0, :, :])
        c_dash[:, 0, :, :] = torch.linalg.lu_solve(LU, pivots, c[:, 0, :, :])
        d_dash[:, 0:M, :] = torch.linalg.lu_solve(LU, pivots, d[:, 0:M, :])

        for i in range(1, N-1):
            LU, pivots = torch.linalg.lu_factor(b[:, i, :, :] - a[:, i-1, :, :] @ c_dash[:, i-1, :, :])
            c_dash[:, i, :, :] = torch.linalg.lu_solve(LU, pivots, c[:, i, :, :])
            d_dash[:, i*M:(i+1)*M, :] = torch.linalg.lu_solve(LU, pivots, d[:, i*M:(i+1)*M, :] - a[:, i-1, :, :] @ d_dash[:, (i-1)*M:i*M, :])

        w = b[:, N-1, :, :] - a[:, N-2, :, :] @ c_dash[:, N-2, :, :]
        d_dash[:, (N-1)*M:MN, :] = torch.linalg.solve(w, d[:, (N-1)*M:MN, :] - a[:, N-2, :, :] @ d_dash[:, (N-2)*M:(N-1)*M, :])

        # backward substitution
        #x = torch.empty_like(d)
        #x[:, (N-1)*M:MN, :] = d_dash[:, (N-1)*M:MN, :]
        #for i in range(N-1, 0, -1):
        #    x[:, (i-1)*M:i*M, :] = d_dash[:, (i-1)*M:i*M, :] - c_dash[:, i-1, :, :] @ x[:, i*M:(i+1)*M, :]
        x = d_dash
        for i in range(N-1, 0, -1):
            x[:, (i-1)*M:i*M, :] -= c_dash[:, i-1, :, :] @ x[:, i*M:(i+1)*M, :]

        return x

    elif method == 'linalg':
        A = torch.zeros((B, MN, MN), dtype=d.dtype, device=d.device)
        for i in range(N):
            A[:, i*M:(i+1)*M, i*M:(i+1)*M] = b[:, i, :, :]
        for i in range(N-1):
            A[:, (i+1)*M:(i+2)*M, i*M:(i+1)*M] = a[:, i, :, :]
            A[:, i*M:(i+1)*M, (i+1)*M:(i+2)*M] = c[:, i, :, :]
        return torch.linalg.solve(A, d)

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


class BlockTriDiagSolveFcn(torch.autograd.Function):
    """
    Differentiable block tridiagonal matrix solver. See `blocktridiagsolve`.
    """

    @staticmethod
    def forward(ctx, b, a, c, d, method='thomas'):
        with torch.no_grad():
            x = blocktridiagsolve(b, a, c, d, method)
        ctx.save_for_backward(b, a, c, d, x)
        ctx.method = method
        return x

    @staticmethod
    def backward(ctx, grad_x):
        b, a, c, d, x = ctx.saved_tensors

        B, N, M, _ = b.shape
        w = blocktridiagsolve(b.transpose(2, 3), c.transpose(2, 3), a.transpose(2, 3), grad_x, ctx.method)
        grad_b, grad_a, grad_c = None, None, None

        if ctx.needs_input_grad[0]:
            grad_b = -1.0 * torch.matmul(w.reshape(B, N, M, -1), x.reshape(B, N, M, -1).transpose(2, 3))
        if ctx.needs_input_grad[1]:
            grad_a = -1.0 * torch.matmul(w.reshape(B, N, M, -1)[:, 1:, :, :], x.reshape(B, N, M, -1).transpose(2, 3)[:, :-1, :, :])
        if ctx.needs_input_grad[2]:
            grad_c = -1.0 * torch.matmul(w.reshape(B, N, M, -1)[:, :-1, :, :], x.reshape(B, N, M, -1).transpose(2, 3)[:, 1:, :, :])

        grad_d = w if ctx.needs_input_grad[3] else None

        return grad_b, grad_a, grad_c, grad_d, None



#
# --- testing ---
#

if __name__ == '__main__':

    # --- testing accuracy ---
    B, N, K = 2, 16, 5
    type = torch.float64
    device = torch.device("cpu")

    # arbitrary
    b = 2.0 * torch.ones((B, N), dtype=type, device=device) + 0.1 * torch.rand((B, N), dtype=type, device=device, requires_grad=True)
    a = -1.0 * torch.rand((B, N - 1), dtype=type, device=device, requires_grad=True)
    c = -1.0 * torch.rand((B, N - 1), dtype=type, device=device, requires_grad=True)
    d = torch.rand((B, N, K), dtype=type, device=device, requires_grad=True)

    print("Checking tridiagsolve accuracy on arbitrary input...")
    print(d.shape)
    A = torch.diag_embed(b) + torch.diag_embed(a, offset=-1) + torch.diag_embed(c, offset=1)

    x = tridiagsolve(b, a, c, d, 'cyclic')
    print(torch.max(torch.abs(A @ x - d)).item())

    x = tridiagsolve(b, a, c, d, 'thomas')
    print(torch.max(torch.abs(A @ x - d)).item())

    x = tridiagsolve(b, a, c, d, 'linalg')
    print(torch.max(torch.abs(A @ x - d)).item())

    # poisson
    print("Checking tridiagsolve accuracy on poisson input...")
    neg_ones = -1.0 * torch.ones((B, N-1), dtype=type, device=device)
    A = torch.diag_embed(b) + torch.diag_embed(neg_ones, offset=-1) + torch.diag_embed(neg_ones, offset=1)

    x = tridiagsolve(b, neg_ones, neg_ones, d, 'cyclic')
    print(torch.max(torch.abs(A @ x - d)).item())

    x = tridiagsolve(b, neg_ones, neg_ones, d, 'thomas')
    print(torch.max(torch.abs(A @ x - d)).item())

    x = tridiagsolve(b, neg_ones, neg_ones, d, 'linalg')
    print(torch.max(torch.abs(A @ x - d)).item())

    # block tridiagonal matrices
    M = 3

    b = 0.1 * torch.rand((B, N, M, M), dtype=type, device=device) + \
        2.0 * torch.eye(M, dtype=type, device=device).view(1, 1, M, M)
    a = -1.0 * torch.rand((B, N - 1, M, M), dtype=type, device=device)
    c = -1.0 * torch.rand((B, N - 1, M, M), dtype=type, device=device)
    #d = torch.rand((B, M*N, K), dtype=type, device=device)
    d = torch.linspace(1, B*N*M*K, B*N*M*K, dtype=type, device=device).reshape(B, N * M, K)

    A = torch.zeros((B, M*N, M*N), dtype=type, device=device)
    for n in range(N):
        A[:, n * M:(n + 1) * M, n * M:(n + 1) * M] = b[:, n, :, :]
    for n in range(N - 1):
        A[:, (n + 1) * M:(n + 2) * M, n * M:(n + 1) * M] = a[:, n, :, :]
        A[:, n * M:(n + 1) * M, (n + 1) * M:(n + 2) * M] = c[:, n, :, :]

    print("Checking blocktridiagsolve accuracy on arbitrary input...")
    print(b.shape, d.shape)

    x = blocktridiagsolve(b, a, c, d, 'cyclic')
    print(torch.max(torch.abs(A @ x - d)).item())

    x = blocktridiagsolve(b, a, c, d, 'thomas')
    print(torch.max(torch.abs(A @ x - d)).item())

    x = blocktridiagsolve(b, a, c, d, 'linalg')
    print(torch.max(torch.abs(A @ x - d)).item())

    #exit(0)

    # --- testing gradient calculation ---
    from torch.autograd import gradcheck

    torch.manual_seed(22)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")

    B, N, M, K = 2, 16, 3, 5
    type = torch.float64
    
    b = 2.0 * torch.ones((B, N), dtype=type, device=device) + 0.1 * torch.rand((B, N), dtype=type, device=device, requires_grad=True)
    a = -1.0 * torch.rand((B, N-1), dtype=type, device=device, requires_grad=True)
    c = -1.0 * torch.rand((B, N-1), dtype=type, device=device, requires_grad=True)
    d = torch.rand((B, N, K), dtype=type, device=device, requires_grad=True)

    print("Checking gradients (cyclic)...")
    test = gradcheck(TriDiagSolveFcn().apply, (b, a, c, d, 'cyclic'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    print("Checking gradients (thomas)...")
    test = gradcheck(TriDiagSolveFcn().apply, (b, a, c, d, 'thomas'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    b = 0.1 * torch.rand((B, N, M, M), dtype=type, device=device, requires_grad=True) + 2.0 * torch.eye(M, dtype=type, device=device).view(1, 1, M, M)
    a = -1.0 * torch.rand((B, N - 1, M, M), dtype=type, device=device, requires_grad=True)
    c = -1.0 * torch.rand((B, N - 1, M, M), dtype=type, device=device, requires_grad=True)
    d = torch.rand((B, M*N, K), dtype=type, device=device, requires_grad=True)

    print("Checking gradients (block cyclic)...")
    test = gradcheck(BlockTriDiagSolveFcn().apply, (b, a, c, d, 'cyclic'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    print("Checking gradients (block thomas)...")
    test = gradcheck(BlockTriDiagSolveFcn().apply, (b, a, c, d, 'thomas'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    #exit(0)

    # --- testing running time ---
    import time
    print("Testing TriDiagSolve running time...")

    N, M = 256, 7
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


    print("Testing BlockTriDiagSolve running time...")
    for B in (1, 10, 100):
        b = 0.1 * torch.rand((B, N, M, M), dtype=type, device=device, requires_grad=True) + \
             2.0 * M * torch.eye(M, dtype=type, device=device).view(1, 1, M, M)
        a = -1.0 * torch.rand((B, N - 1, M, M), dtype=type, device=device, requires_grad=True)
        c = -1.0 * torch.rand((B, N - 1, M, M), dtype=type, device=device, requires_grad=True)
        d = torch.rand((B, M * N, 1), dtype=type, device=device, requires_grad=True)

        print("...data size {} {}".format(b.shape, d.shape))

        start = time.time()
        x = BlockTriDiagSolveFcn.apply(b.clone().detach().requires_grad_(True), a.clone().detach().requires_grad_(True),
                                  c.clone().detach().requires_grad_(True), d.clone().detach().requires_grad_(True), 'cyclic')
        x_elapsed = time.time() - start

        start = time.time()
        x.sum().backward()
        dx_elapsed = time.time() - start

        start = time.time()
        y = BlockTriDiagSolveFcn.apply(b.clone().detach().requires_grad_(True), a.clone().detach().requires_grad_(True),
                                  c.clone().detach().requires_grad_(True), d.clone().detach().requires_grad_(True), 'thomas')
        y_elapsed = time.time() - start

        start = time.time()
        y.sum().backward()
        dy_elapsed = time.time() - start

        start = time.time()
        z = BlockTriDiagSolveFcn.apply(b.clone().detach().requires_grad_(True), a.clone().detach().requires_grad_(True),
                                  c.clone().detach().requires_grad_(True), d.clone().detach().requires_grad_(True), 'linalg')
        z_elapsed = time.time() - start
        print(z.shape)

        start = time.time()
        z.sum().backward()
        dz_elapsed = time.time() - start

        A = torch.zeros((B, M * N, M * N), dtype=type, device=device)
        for n in range(N):
            A[:, n * M:(n + 1) * M, n * M:(n + 1) * M] = b[:, n, :, :]
        for n in range(N - 1):
            A[:, (n + 1) * M:(n + 2) * M, n * M:(n + 1) * M] = a[:, n, :, :]
            A[:, n * M:(n + 1) * M, (n + 1) * M:(n + 2) * M] = c[:, n, :, :]

        print('accuracy cyclic/thomas/linalg.solve: {:.3e}/{:.3e}/{:.3e}'.format(torch.max(torch.abs(A @ x - d)).item(),
            torch.max(torch.abs(A @ y - d)).item(), torch.max(torch.abs(A @ z - d)).item()))
        print(' forward cyclic/thomas/linalg.solve: {:.3e}/{:.3e}/{:.3e}'.format(x_elapsed, y_elapsed, z_elapsed))
        print('backward cyclic/thomas/linalg.solve: {:.3e}/{:.3e}/{:.3e}'.format(dx_elapsed, dy_elapsed, dz_elapsed))
