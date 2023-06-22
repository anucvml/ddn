# Experiments for the paper "Towards Understanding Gradient Approximation in Equality Constrained Deep Declarative
# Networks", by S. Gould, M. Xu, Z. Xu, and Y. Liu. In the ICML Workshop on Differentiable Almost Everything, 2023.
# Stephen Gould <stephen.gould@anu.edu.au>
#

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

import torch
import torch.nn as nn

import sys
sys.path.append("../../../")

import os
os.makedirs("figures", exist_ok=True)

# --------------------------------------------------------------------------------------------------------------------
# --- optimal transport ---
# --------------------------------------------------------------------------------------------------------------------

from ddn.pytorch.optimal_transport import sinkhorn, OptimalTransportFcn

class InstrumentedApproxOptimalTransportFcn(OptimalTransportFcn):
    """OptimalTransportFcn instrumented to compute cosine similarity of exact and approx gradients."""

    trace = None

    @staticmethod
    def backward(ctx, dJdP):
        M, r, c, P = ctx.saved_tensors
        dJdM = -1.0 * ctx.gamma * P * dJdP
        output_grad = OptimalTransportFcn.backward(ctx, dJdP)

        g_sim = nn.functional.cosine_similarity(dJdM.view(dJdM.shape[0], -1), output_grad[0].view(output_grad[0].shape[0], -1))
        if InstrumentedApproxOptimalTransportFcn.trace is not None:
            InstrumentedApproxOptimalTransportFcn.trace.append(float(g_sim.mean()))

        return (dJdM, *output_grad[1:])


class OTNetwork(nn.Module):
    """Example optimal transport network comprising a MLP data processing layer followed by a
    differentiable optimal transport layer. Input is (B, Z, 1); output is (B, M, N)."""

    def __init__(self, dim_z, m, n, approx=True):
        super(OTNetwork, self).__init__()

        self.dim_z = dim_z
        self.n = n
        self.m = m
        self.approx = approx

        self.mlp = nn.Sequential(
            nn.Linear(dim_z, m * n),
            nn.ReLU(),
            nn.Linear(m * n, m * n),
            nn.ReLU(),
            nn.Linear(m * n, m * n)
        )

    def forward(self, z):
        assert z.shape[1] == self.dim_z

        # construct input for optimal transport
        x = self.mlp(z)
        x = torch.reshape(x, (z.shape[0], self.m, self.n))

        if self.approx:
            y = InstrumentedApproxOptimalTransportFcn.apply(x, None, None, 1.0, 1.0e-6, 1000, False, 'block')
        else:
            y = OptimalTransportFcn().apply(x, None, None, 1.0, 1.0e-6, 1000, False, 'block')

        return y


def OTExperiments(dim_z=5, m=10, n=10, batch=10, iters=500, trials=10, approx=False):

    learning_curves = [[] for i in range(trials)]
    cosine_sim_curves = [[] for i in range(trials)]

    for trial in range(trials):
        # prepare data and model
        torch.manual_seed(22 + trial)
        P_true = sinkhorn(torch.randn((batch, m, n), dtype=torch.float, requires_grad=False))
        z_init = torch.randn((batch, dim_z), dtype=torch.float, requires_grad=False)

        InstrumentedApproxOptimalTransportFcn.trace = cosine_sim_curves[trial]

        model = OTNetwork(dim_z, m, n, approx=approx)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

        # do optimisation
        for i in range(iters):
            optimizer.zero_grad(set_to_none=True)
            P_pred = model(z_init)
            loss = torch.linalg.norm(P_pred - P_true)
            learning_curves[trial].append(float(loss.item()))
            loss.backward()
            optimizer.step()

            if (i % 100 == 0):
                print("{: 3} {: 6}: {}".format(trial, i, loss.item()))

        InstrumentedApproxOptimalTransportFcn.trace = None

    return learning_curves, cosine_sim_curves


def PlotResults(exact_curves, approx_curves, fcn=plt.semilogy):
    """plot results of experiments."""

    exact_mean = np.mean(exact_curves, axis=0)
    approx_mean = np.mean(approx_curves, axis=0)

    fcn(exact_mean, 'b')
    fcn(approx_mean, 'r')
    for trial in range(len(exact_curves)):
        fcn(exact_curves[trial], 'b', alpha=0.1)
    for trial in range(len(approx_curves)):
        fcn(approx_curves[trial], 'r', alpha=0.1)
    fcn(exact_mean, 'b')
    fcn(approx_mean, 'r')
    plt.xlabel('iter.'); plt.ylabel('loss')
    plt.legend(('exact', 'approx.'))


# --------------------------------------------------------------------------------------------------------------------
# --- projection onto L2-sphere ---
# --------------------------------------------------------------------------------------------------------------------

class ProjectOntoSphereFcn(torch.autograd.Function):
    """PyTorch autograd function for projection onto L2-sphere."""

    trace = None

    @staticmethod
    def forward(ctx, X, method='exact'):
        B, M = X.shape

        with torch.no_grad():
            Y = nn.functional.normalize(X)

        ctx.save_for_backward(X, Y)
        ctx.method = method
        return Y

    @staticmethod
    def backward(ctx, dJdY):
        X, Y = ctx.saved_tensors
        B, M = X.shape

        if ctx.method == 'exact':
            dJdX = (dJdY - torch.matmul(torch.einsum("bi,bj->bij", Y, Y), dJdY.view(B, M, 1)).view(B, M)) / torch.linalg.vector_norm(X, dim=1).view(B, 1)
        elif ctx.method == 'approx':
            dJdX = dJdY
        else:
            assert False

        if ProjectOntoSphereFcn.trace is not None:
            dJdX_exact = (dJdY - torch.matmul(torch.einsum("bi,bj->bij", Y, Y), dJdY.view(B, M, 1)).view(B, M)) / torch.linalg.vector_norm(X, dim=1).view(B, 1)
            g_sim = nn.functional.cosine_similarity(dJdX.view(B, -1), dJdX_exact.view(B, -1))
            ProjectOntoSphereFcn.trace.append(float(g_sim.mean()))

        return dJdX, None


class ProjNetwork(nn.Module):
    """Example projection network comprising a MLP data processing layer followed by a
    differentiable projection layer. Input is (B, Z, 1); output is (B, M)."""

    def __init__(self, dim_z, m, method='exact'):
        super(ProjNetwork, self).__init__()

        self.dim_z = dim_z
        self.m = m
        self.method = method

        self.mlp = nn.Sequential(
            nn.Linear(dim_z, m),
            nn.ReLU(),
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, m)
        )

    def forward(self, z):
        assert z.shape[1] == self.dim_z

        # construct input for declarative node
        x = self.mlp(z)
        #y = nn.functional.normalize(x)
        y = ProjectOntoSphereFcn().apply(x, self.method)

        return y


def ProjExperiments(dim_z=5, m=10, batch=10, iters=500, trials=10, method='exact'):
    learning_curves = [[] for i in range(trials)]
    cosine_sim_curves = [[] for i in range(trials)]

    for trial in range(trials):
        # prepare data and model
        torch.manual_seed(22 + trial)
        y_true = nn.functional.normalize(torch.rand((batch, m), dtype=torch.float, requires_grad=False))
        z_init = torch.randn((batch, dim_z), dtype=torch.float, requires_grad=False)

        if method == 'approx':
            ProjectOntoSphereFcn.trace = cosine_sim_curves[trial]

        model = ProjNetwork(dim_z, m, method=method)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

        # do optimisation
        loss_fcn = nn.MSELoss()
        for i in range(iters):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(z_init)
            loss = loss_fcn(y_true, y_pred)
            #loss = 1.0 - torch.mean(torch.nn.functional.cosine_similarity(y_pred, y_true, dim=1))
            learning_curves[trial].append(float(loss.item()))
            loss.backward()
            optimizer.step()

            if (i % 100 == 0):
                print("{: 3} {: 6}: {}".format(trial, i, loss.item()))

        ProjectOntoSphereFcn.trace = None

    return learning_curves, cosine_sim_curves


# --------------------------------------------------------------------------------------------------------------------
# --- eigen decomposition ---
# --------------------------------------------------------------------------------------------------------------------

from ddn.pytorch.eigen_decomposition import EigenDecompositionFcn

class InstrumentedApproxEigenDecompositionFcn(EigenDecompositionFcn):
    """EigenDecompFcn instrumented to compute cosine similarity of exact and approx gradients."""

    trace = None

    @staticmethod
    def backward(ctx, dJdQ):
        V, Q = ctx.saved_tensors
        B, M, K = dJdQ.shape

        zero = torch.zeros(1, dtype=V.dtype, device=V.device)
        L = -1.0 * torch.where(torch.abs(V) < EigenDecompositionFcn.eps, zero, 1.0 / V).view(B, M, 1)
        dJdX = torch.bmm(torch.bmm(Q, L * torch.bmm(Q.transpose(1, 2), dJdQ)), Q.transpose(1, 2))
        dJdX = 0.5 * (dJdX + dJdX.transpose(1, 2))

        if InstrumentedApproxEigenDecompositionFcn.trace is not None:
            L = V[:, -K:].view(B, 1, K) - V.view(B, M, 1)
            torch.where(torch.abs(L) < EigenDecompositionFcn.eps, zero, 1.0 / L, out=L)
            dJdX_exact = torch.bmm(torch.bmm(Q, L * torch.bmm(Q.transpose(1, 2), dJdQ)), Q[:, :, -K:].transpose(1, 2))
            dJdX_exact = 0.5 * (dJdX_exact + dJdX_exact.transpose(1, 2))
            g_sim = nn.functional.cosine_similarity(dJdX.view(B, -1), dJdX_exact.view(B, -1))
            InstrumentedApproxEigenDecompositionFcn.trace.append(float(g_sim.mean()))

        return dJdX, None


class EDNetwork(nn.Module):
    """Example eigen decomposition network comprising a MLP data processing layer followed by a
    differentiable eigen decomposition layer. Input is (B, Z, 1); output is (B, M, M)."""

    def __init__(self, dim_z, m, method='exact', top_k=None, matrix_type='general'):
        super(EDNetwork, self).__init__()

        self.dim_z = dim_z
        self.m = m
        self.method = method
        self.top_k = None
        self.matrix_type = matrix_type

        self.mlp = nn.Sequential(
            nn.Linear(dim_z, m * m),
            nn.ReLU(),
            nn.Linear(m * m, m * m),
            nn.ReLU(),
            nn.Linear(m * m, m * m)
        )

    def forward(self, z):
        assert z.shape[1] == self.dim_z

        # construct input for declarative node
        x = self.mlp(z)
        x = torch.reshape(x, (z.shape[0], self.m, self.m))

        if self.matrix_type == 'general':
            pass
        elif self.matrix_type == 'psd':
            x = torch.matmul(x, x.transpose(1, 2)) # positive definite
        elif self.matrix_type == 'nsd':
            x = -1.0 * torch.matmul(x, x.transpose(1, 2))  # negative definite
        elif self.matrix_type == '1pev':
            u = x
            x = -1.0 * torch.matmul(u, u.transpose(1, 2)) # negative definite
            x = x + 2.0 * torch.einsum("bi,bj->bij", u[:, :, 0], u[:, :, 0]) # 1 positive eigenvalue
        elif self.matrix_type == 'rank1':
            u = x
            x = torch.matmul(u[:, :, 0], u[:, :, 0].transpose(1, 2))
        elif self.matrix_type == 'rank2':
            u = x
            x = torch.matmul(u[:, :, 0:1], u[:, :, 0:1].transpose(1, 2))
        else:
            assert False, "unknown matrix_type"

        if self.method == 'pytorch':
            x = 0.5 * (x + x.transpose(1, 2))
            v, y = torch.linalg.eigh(x)
        elif self.method == 'exact':
            y = EigenDecompositionFcn().apply(x, self.top_k)
        elif self.method == 'approx':
            y = InstrumentedApproxEigenDecompositionFcn().apply(x, self.top_k)
        else:
            assert False

        return y


def EDExperiments(dim_z=5, m=10, batch=10, iters=500, trials=10, method='exact', loss_on='all', mat_type='general'):

    learning_curves = [[] for i in range(trials)]
    cosine_sim_curves = [[] for i in range(trials)]

    for trial in range(trials):
        # prepare data and model
        torch.manual_seed(22 + trial)
        X_true = torch.rand((batch, m, m), dtype=torch.float, requires_grad=False)
        #X_true = torch.matmul(X_true, X_true.transpose(1, 2))
        V_true, Q_true = torch.linalg.eigh(X_true)
        z_init = torch.randn((batch, dim_z), dtype=torch.float, requires_grad=False)

        if method == 'approx':
            InstrumentedApproxEigenDecompositionFcn.trace = cosine_sim_curves[trial]

        model = EDNetwork(dim_z, m, method=method, top_k=1 if loss_on=='max' else None, matrix_type=mat_type)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

        # do optimisation
        for i in range(iters):
            optimizer.zero_grad(set_to_none=True)
            Q_pred = model(z_init)
            if loss_on == 'all':
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred, Q_true, dim=1))) # all ev
            elif loss_on == 'max':
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred[:, :, -1], Q_true[:, :, -1]))) # largest
            elif loss_on == 'min':
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred[:, :, 0], Q_true[:, :, 0]))) # smallest
            else:
                assert False, "loss_on must be one of ('all', 'max', 'min')"
            learning_curves[trial].append(float(loss.item()))
            loss.backward()
            optimizer.step()

            if (i % 100 == 0):
                print("{: 3} {: 6}: {}".format(trial, i, loss.item()))

        InstrumentedApproxEigenDecompositionFcn.trace = None

    return learning_curves, cosine_sim_curves


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(torch.__version__)
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No CUDA")

    # Enable test modules
    enable_ot_exp        = True
    enable_proj_exp      = True
    enable_ed_exp        = True

    # --------------------------------------------------------------------------------------------------------------------
    # --- optimal transport ---
    # --------------------------------------------------------------------------------------------------------------------

    if enable_ot_exp:
        # under parameterized
        exact_curves, _ = OTExperiments(dim_z=5, approx=False)
        approx_curves, approx_sim_curves = OTExperiments(dim_z=5, approx=True)

        plt.figure()
        PlotResults(exact_curves, approx_curves)
        plt.savefig('figures/descent_ot_small_z.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        approx_sim_mean = np.mean(approx_sim_curves, axis=0)
        for trial in range(len(approx_curves)):
            plt.plot(approx_sim_curves[trial], 'r', alpha=0.1)
        plt.plot(approx_sim_mean, 'r')
        plt.xlabel('iter.');
        plt.ylabel('cosine similarity')
        plt.gca().set_ylim(0.85, 1.0)
        plt.savefig('figures/descent_ot_gsim_small_z.pdf', dpi=300, bbox_inches='tight')

        # over parameterized
        exact_curves, _ = OTExperiments(dim_z=100, approx=False)
        approx_curves, approx_sim_curves = OTExperiments(dim_z=100, approx=True)

        plt.figure()
        PlotResults(exact_curves, approx_curves)
        plt.savefig('figures/descent_ot_large_z.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        approx_sim_mean = np.mean(approx_sim_curves, axis=0)
        for trial in range(len(approx_curves)):
            plt.plot(approx_sim_curves[trial], 'r', alpha=0.1)
        plt.plot(approx_sim_mean, 'r')
        plt.xlabel('iter.');
        plt.ylabel('cosine similarity')
        plt.gca().set_ylim(0.85, 1.0)
        plt.savefig('figures/descent_ot_gsim_large_z.pdf', dpi=300, bbox_inches='tight')


    # --------------------------------------------------------------------------------------------------------------------
    # --- project on sphere ---
    # --------------------------------------------------------------------------------------------------------------------

    if enable_proj_exp:
        # under parametrized
        exact_curves, _ = ProjExperiments(dim_z=5, method='exact')
        approx_curves, approx_sim_curves = ProjExperiments(dim_z=5, method='approx')

        plt.figure()
        PlotResults(exact_curves, approx_curves, fcn=plt.semilogy)
        plt.gca().set_ylim(2.0e-4, 3.0e-1)
        plt.savefig('figures/descent_proj_small_z.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        approx_sim_mean = np.mean(approx_sim_curves, axis=0)
        for trial in range(len(approx_curves)):
            plt.plot(approx_sim_curves[trial], 'r', alpha=0.1)
        plt.plot(approx_sim_mean, 'r')
        plt.xlabel('iter.');
        plt.ylabel('cosine similarity')
        plt.savefig('figures/descent_proj_gsim_small_z.pdf', dpi=300, bbox_inches='tight')

        # over parametrized
        exact_curves, _ = ProjExperiments(dim_z=100, method='exact')
        approx_curves, approx_sim_curves = ProjExperiments(dim_z=100, method='approx')

        plt.figure()
        PlotResults(exact_curves, approx_curves, fcn=plt.semilogy)
        plt.gca().set_ylim(2.0e-4, 3.0e-1)
        plt.savefig('figures/descent_proj_large_z.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        approx_sim_mean = np.mean(approx_sim_curves, axis=0)
        for trial in range(len(approx_curves)):
            plt.plot(approx_sim_curves[trial], 'r', alpha=0.1)
        plt.plot(approx_sim_mean, 'r')
        plt.xlabel('iter.');
        plt.ylabel('cosine similarity')
        plt.savefig('figures/descent_proj_gsim_large_z.pdf', dpi=300, bbox_inches='tight')


    # --------------------------------------------------------------------------------------------------------------------
    # --- eigen decomposition ---
    # --------------------------------------------------------------------------------------------------------------------

    if enable_ed_exp:
        # all evs, general matrix
        exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='all')
        approx_curves, approx_sim_curves = EDExperiments(dim_z=5, method='approx', loss_on='all')

        plt.figure()
        PlotResults(exact_curves, approx_curves, fcn=plt.plot)
        plt.savefig('figures/descent_ed_all_general.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        approx_sim_mean = np.mean(approx_sim_curves, axis=0)
        for trial in range(len(approx_curves)):
            plt.plot(approx_sim_curves[trial], 'r', alpha=0.1)
        plt.plot(approx_sim_mean, 'r')
        plt.xlabel('iter.');
        plt.ylabel('cosine similarity')
        plt.savefig('figures/descent_ed_gsim_all_general.pdf', dpi=300, bbox_inches='tight')

        # min ev, general matrix
        exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='min')
        approx_curves, approx_sim_curves = EDExperiments(dim_z=5, method='approx', loss_on='min')

        plt.figure()
        PlotResults(exact_curves, approx_curves, fcn=plt.plot)
        plt.savefig('figures/descent_ed_min_general.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        approx_sim_mean = np.mean(approx_sim_curves, axis=0)
        for trial in range(len(approx_curves)):
            plt.plot(approx_sim_curves[trial], 'r', alpha=0.1)
        plt.plot(approx_sim_mean, 'r')
        plt.xlabel('iter.');
        plt.ylabel('cosine similarity')
        plt.savefig('figures/descent_ed_gsim_min_general.pdf', dpi=300, bbox_inches='tight')
        
        # min ev, psd matrix
        exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='min', mat_type='psd')
        approx_curves, approx_sim_curves = EDExperiments(dim_z=5, method='approx', loss_on='min', mat_type='psd')

        plt.figure()
        PlotResults(exact_curves, approx_curves, fcn=plt.plot)
        plt.savefig('figures/descent_ed_min_psd.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        approx_sim_mean = np.mean(approx_sim_curves, axis=0)
        for trial in range(len(approx_curves)):
            plt.plot(approx_sim_curves[trial], 'r', alpha=0.1)
        plt.plot(approx_sim_mean, 'r')
        plt.xlabel('iter.');
        plt.ylabel('cosine similarity')
        plt.savefig('figures/descent_ed_gsim_min_psd.pdf', dpi=300, bbox_inches='tight')

        # max ev, general matrix
        exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='max')
        approx_curves, approx_sim_curves = EDExperiments(dim_z=5, method='approx', loss_on='max')

        plt.figure()
        PlotResults(exact_curves, approx_curves, fcn=plt.plot)
        plt.savefig('figures/descent_ed_max_general.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        approx_sim_mean = np.mean(approx_sim_curves, axis=0)
        for trial in range(len(approx_curves)):
            plt.plot(approx_sim_curves[trial], 'r', alpha=0.1)
        plt.plot(approx_sim_mean, 'r')
        plt.xlabel('iter.');
        plt.ylabel('cosine similarity')
        plt.savefig('figures/descent_ed_gsim_max_general.pdf', dpi=300, bbox_inches='tight')

        # max ev, nsd matrix
        exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='max', mat_type='nsd')
        approx_curves, approx_sim_curves = EDExperiments(dim_z=5, method='approx', loss_on='max', mat_type='nsd')

        plt.figure()
        PlotResults(exact_curves, approx_curves, fcn=plt.plot)
        plt.savefig('figures/descent_ed_max_nsd.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        approx_sim_mean = np.mean(approx_sim_curves, axis=0)
        for trial in range(len(approx_curves)):
            plt.plot(approx_sim_curves[trial], 'r', alpha=0.1)
        plt.plot(approx_sim_mean, 'r')
        plt.xlabel('iter.');
        plt.ylabel('cosine similarity')
        plt.savefig('figures/descent_ed_gsim_max_nsd.pdf', dpi=300, bbox_inches='tight')

        # max ev, rank2 matrix
        exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='max', mat_type='rank2')
        approx_curves, approx_sim_curves = EDExperiments(dim_z=5, method='approx', loss_on='max', mat_type='rank2')

        plt.figure()
        PlotResults(exact_curves, approx_curves, fcn=plt.plot)
        plt.savefig('figures/descent_ed_max_rank2.pdf', dpi=300, bbox_inches='tight')

        plt.figure()
        approx_sim_mean = np.mean(approx_sim_curves, axis=0)
        for trial in range(len(approx_curves)):
            plt.plot(approx_sim_curves[trial], 'r', alpha=0.1)
        plt.plot(approx_sim_mean, 'r')
        plt.xlabel('iter.');
        plt.ylabel('cosine similarity')
        plt.savefig('figures/descent_ed_gsim_max_rank2.pdf', dpi=300, bbox_inches='tight')


    if enable_proj_exp or enable_ot_exp or enable_ed_exp:
        plt.show()
