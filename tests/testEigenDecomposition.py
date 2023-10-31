# TEST EIGEN DECOMPOSITION DEEP DECLARATIVE NODE
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#
# When running from the command-line make sure that the "ddn" package has been added to the PYTHONPATH:
#   $ export PYTHONPATH=${PYTHONPATH}: ../ddn
#   $ python testEigenDecomposition.py
#

import time, copy, sys, os
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

import torch.autograd.profiler as profiler

#
# --- import optimized version ---
#

sys.path.append("..")
from ddn.pytorch.eigen_decomposition import EigenDecompositionFcn

#
# --- alternative versions ---
#

class EigenDecompositionFcn_v1(torch.autograd.Function):
    """PyTorch autograd function for eigen decomposition real symmetric matrices. Returns all eigenvectors
    or just eigenvectors associated with the top-k eigenvalues."""

    eps = 1.0e-9 # tolerance to consider two eigenvalues equal

    @staticmethod
    def forward(ctx, X, top_k=None):
        B, M, N = X.shape
        assert N == M
        assert (top_k is None) or (1 <= top_k <= M)

        with torch.no_grad():
            X = 0.5 * (X + X.transpose(1, 2))
            lmd, Y = torch.linalg.eigh(X)

        ctx.save_for_backward(X, lmd, Y)
        return Y if top_k is None else Y[:, :, -top_k:]

    @staticmethod
    def backward(ctx, dJdY):
        X, lmd, Y = ctx.saved_tensors
        B, M, K = dJdY.shape

        dJdX = torch.zeros_like(X)

        # loop over eigenvalues
        for i in range(K):
            L = torch.diag_embed(lmd[:, i + M - K].repeat(M, 1).transpose(0, 1))
            w = -0.5 * torch.bmm(torch.pinverse(X - L), dJdY[:, :, i].view(B, M, 1)).view(B, M)
            dJdX += torch.einsum("bi,bj->bij", w, Y[:, :, i + M - K]) + torch.einsum("bj,bi->bij", w, Y[:, :, i + M - K])

        return dJdX, None


class EigenDecompositionFcn_v2(torch.autograd.Function):
    """PyTorch autograd function for eigen decomposition real symmetric matrices. Returns all eigenvectors
    or just eigenvectors associated with the top-k eigenvalues."""

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

        dJdX = torch.zeros_like(Y)
        zero = torch.zeros(1, dtype=lmd.dtype, device=lmd.device)

        # loop over eigenvalues
        for i in range(K):
            L = lmd - lmd[:, i + M - K].view(B, 1)
            L = torch.where(torch.abs(L) < EigenDecompositionFcn.eps, zero, 1.0 / L)
            w = -0.5 * torch.bmm(torch.bmm(Y, L.view(B, M, 1) * Y.transpose(1, 2)), dJdY[:, :, i].view(B, M, 1)).view(B, M)
            dJdX += torch.einsum("bi,bj->bij", w, Y[:, :, i + M - K]) + torch.einsum("bj,bi->bij", w, Y[:, :, i + M - K])

        return dJdX, None


class EigenDecompositionFcn_v3(torch.autograd.Function):
    """PyTorch autograd function for eigen decomposition real symmetric matrices. Returns all eigenvectors
    or just eigenvectors associated with the top-k eigenvalues."""

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

        # do all eigenvalues in one go
        L = lmd.view(B, 1, M) - lmd[:, -K:].view(B, K, 1)
        L = torch.where(torch.abs(L) < EigenDecompositionFcn.eps, zero, 1.0 / L)
        w = torch.bmm(L * torch.bmm(dJdY.transpose(1, 2), Y), Y.transpose(1, 2))

        dJdX = torch.einsum("bik,bkj->bji", Y[:, :, -K:], w)
        dJdX = -0.5 * (dJdX + dJdX.transpose(1, 2))

        return dJdX, None

#
# --- speed and memory profiling functions
#

def generate_random_data(B, M, enable_symmetric=True, dtype=torch.float32, device='cpu'):
    X = np.random.randn(B, M, M)
    if enable_symmetric: X = 0.5 * (X + X.transpose(0, 2, 1))
    X = torch.tensor(X, dtype=dtype, device=device, requires_grad=True)

    return X

# Define a function to avoid the warning of preallocated inputs in the memory profile
def test_fnc_cpu_memory(fnc, X):
    out = fnc(X)
    eigenvector = out[1] if isinstance(out, tuple) else out
    loss = eigenvector.mean()
    loss.backward()

def speed_memory_test(fnc, data_sz, num_iter_speed=1000, num_iter_memory=5, device='cpu', dtype=torch.float32):
    B, M = data_sz
    time_forward_total, time_backward_total = 0, 0

    # Speed, the first loop is ignored
    for idx in range(num_iter_speed):
        X = generate_random_data(B, M, dtype=dtype, device=device)

        time_start = time.monotonic()
        out = fnc(X)
        eigenvector = out[1] if isinstance(out, tuple) else out
        duration = time.monotonic() - time_start if (idx > 0) else 0
        time_forward_total += duration

        loss = eigenvector.mean()

        time_start = time.monotonic()
        loss.backward()
        duration = time.monotonic() - time_start if (idx > 0) else 0
        time_backward_total += duration

    time_forward = time_forward_total * 1000 / (num_iter_speed - 1)
    time_backward = time_backward_total * 1000 / (num_iter_speed - 1)

    # Memory, the first loop is ignored, set num_iter_memory small for fast test
    if device == torch.device('cpu'):
        X = generate_random_data(B, M, dtype=dtype, device=device)

        with profiler.profile(profile_memory=True) as prof:
            for idx in range(num_iter_memory):
                test_fnc_cpu_memory(fnc, copy.deepcopy(X))

        memory = prof.total_average().cpu_memory_usage / (1024 * 1024)
    else:
        memory_total = 0

        for idx in range(num_iter_memory):
            X = generate_random_data(B, M, dtype=dtype, device=device)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            out = fnc(X)
            eigenvector = out[1] if isinstance(out, tuple) else out
            loss = eigenvector.mean()
            loss.backward()

            memory_current = torch.cuda.max_memory_allocated(None) if (idx > 0) else 0
            memory_total += memory_current

        memory = memory_total / (1024 * 1024 * (num_iter_memory - 1))

    return time_forward, time_backward, memory

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    print(torch.__version__)
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No CUDA")

    if True:
        # gradient check
        from torch.autograd import gradcheck

        for m in (5, 8, 16):
            for f in [EigenDecompositionFcn,
                      EigenDecompositionFcn_v1,
                      EigenDecompositionFcn_v2,
                      EigenDecompositionFcn_v3]:

                X = torch.randn((3, m, m), dtype=torch.double, requires_grad=True)
                X = 0.5 * (X + X.transpose(1, 2))
                test = gradcheck(f().apply, (X, None), eps=1e-6, atol=1e-3, rtol=1e-6)
                print("{}(X, None): {}".format(f.__name__, test))
                test = gradcheck(f().apply, (X, 1), eps=1e-6, atol=1e-3, rtol=1e-6)
                print("{}(X, 1): {}".format(f.__name__, test))

    if True:
        # profiling
        devices = [torch.device('cpu'), torch.device('cuda')] if torch.cuda.is_available() else [torch.device('cpu')]
        
        for device in devices:
            data = {}
            for f in [EigenDecompositionFcn,
                      EigenDecompositionFcn_v1,
                      EigenDecompositionFcn_v2,
                      EigenDecompositionFcn_v3]:

                torch.cuda.empty_cache()
                time_fwd, time_bck, mem = speed_memory_test(lambda X: f.apply(X, None),
                    (5 if device == torch.device('cpu') else 1000, 32),
                    num_iter_speed=1000, num_iter_memory=5, device=device, dtype=torch.float32)

                print(f.__name__, time_fwd, time_bck, mem)
                data[f.__name__] = {'time_fwd': time_fwd, 'time_bck': time_bck, 'total_mem': mem}

            fig, ax = plt.subplots(1, 1)
            b = plt.bar(tuple(range(5)), [data['EigenDecompositionFcn']['time_fwd'],
                                          data['EigenDecompositionFcn_v1']['time_bck'],
                                          data['EigenDecompositionFcn_v2']['time_bck'],
                                          data['EigenDecompositionFcn_v3']['time_bck'],
                                          data['EigenDecompositionFcn']['time_bck']],
                        log=True, color=['r', 'b', 'b', 'b', 'b'])
            ax.set_xticks(range(5))
            ax.set_xticklabels(['fwd', 'bck (v1)', 'bck (v2)', 'bck (v3)', 'bck (final)'])
            # add counts above the two bar graphs
            for rect in b:
                height = rect.get_height()
                value = height / data['EigenDecompositionFcn']['time_fwd']
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{value:0.1f}x', ha='center', va='bottom')

            plt.ylabel('log time (ms)')
            plt.grid(True); plt.grid(True, which='minor', axis='y', ls='--')
            plt.tight_layout()
            plt.title("Differentiable eigen decomposition implementation comparison on {}".format(device))
            #plt.savefig("ed_runtime_versions_{}.png".format(device), dpi=300, bbox_inches='tight')

        plt.show()


