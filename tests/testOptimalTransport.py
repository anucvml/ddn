# TEST OPTIMAL TRANSPORT DEEP DECLARATIVE NODE
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#
# When running from the command-line make sure that the "ddn" package has been added to the PYTHONPATH:
#   $ export PYTHONPATH=${PYTHONPATH}: ../ddn
#   $ python testOptimalTransport.py
#

import torch
import torch.optim as optim
from torch.nn.functional import normalize
from torch.autograd import gradcheck
import torch.autograd.profiler as profiler

import time
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from ddn.pytorch.optimal_transport import sinkhorn, OptimalTransportFcn, OptimalTransportLayer

import unittest
from timeit import timeit

torch.manual_seed(0)

# --- Forward and Backward Unit Tests -----------------------------------------

class TestOptimalTransport(unittest.TestCase):

    def testForward(self):
        """Test Forward Pass"""

        B, H, W = 2, 500, 700
        M = torch.randn((B, H, W), dtype=torch.float)
        r = torch.rand((B, H), dtype=M.dtype)
        c = torch.rand((B, W), dtype=M.dtype)

        P = sinkhorn(M, eps=1.0e-9)
        self.assertTrue(torch.allclose(torch.sum(P, 2), torch.full((B, H), 1.0 / H)))
        self.assertTrue(torch.allclose(torch.sum(P, 1), torch.full((B, W), 1.0 / W)))

        P = OptimalTransportFcn().apply(M, None, None, 1.0e-9)
        self.assertTrue(torch.allclose(torch.sum(P, 2), torch.full((B, H), 1.0 / H)))
        self.assertTrue(torch.allclose(torch.sum(P, 1), torch.full((B, W), 1.0 / W)))

        P = sinkhorn(M, normalize(r, p=1.0), normalize(c, p=1.0), eps=1.0e-9)
        self.assertTrue(torch.allclose(torch.sum(P, 2), normalize(r, p=1.0)))
        self.assertTrue(torch.allclose(torch.sum(P, 1), normalize(c, p=1.0)))

        P = OptimalTransportFcn().apply(M, r, c, 1.0e-9)
        self.assertTrue(torch.allclose(torch.sum(P, 2), normalize(r, p=1.0)))
        self.assertTrue(torch.allclose(torch.sum(P, 1), normalize(c, p=1.0)))

    def testBackward(self):
        """Test Backward Pass"""
        B, H, W = 2, 5, 7
        M = torch.randn((B, H, W), dtype=torch.double, requires_grad=True)
        r = torch.rand((B, H), dtype=M.dtype, requires_grad=True)
        c = torch.rand((B, W), dtype=M.dtype, requires_grad=True)

        f = OptimalTransportFcn().apply

        # default r and c
        gradcheck(f, (M, None, None, 1.0, 1.0e-9, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
        gradcheck(f, (M, None, None, 1.0, 1.0e-9, 1000, False, 'full'), eps=1e-6, atol=1e-3, rtol=1e-6)
        gradcheck(f, (M, None, None, 10.0, 1.0e-9, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
        gradcheck(f, (M, None, None, 10.0, 1.0e-9, 1000, False, 'full'), eps=1e-6, atol=1e-3, rtol=1e-6)

        # with random r and c
        gradcheck(f, (M, r, None, 1.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
        gradcheck(f, (M, None, c, 1.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)

        gradcheck(f, (M, r, c, 1.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
        gradcheck(f, (M, r, c, 10.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)

        # shared r and c
        r = torch.rand((1, H), dtype=M.dtype, requires_grad=True)
        c = torch.rand((1, W), dtype=M.dtype, requires_grad=True)

        gradcheck(f, (M, r, c, 1.0, 1.0e-6, 1000, False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
        gradcheck(f, (M, r, c, 1.0, 1.0e-6, 1000, False, 'full'), eps=1e-6, atol=1e-3, rtol=1e-6)


# --- Toy Learning Example ----------------------------------------------------

def learnM(fcns, M_init, r, c, P_true, iters=1000):
    """Find an M such that sinkhorn(M) matches P_true for each function in fcns.
    Return the M, the learning curves and running times."""

    t_all, h_all, M_all = [], [], []
    for fcn in fcns:
        M = M_init.clone()
        M.requires_grad = True

        optimizer = optim.AdamW([M], lr=1.0e-2)

        h = []
        start_time = time.monotonic()
        for i in range(iters):
            optimizer.zero_grad(set_to_none=True)
            J = torch.linalg.norm(fcn(M, r, c) - P_true)
            h.append(float(J.item()))
            J.backward()
            optimizer.step()
            
        M_all.append(M.detach().clone())
        h_all.append(h)
        t_all.append(time.monotonic() - start_time)
        
    return M_all, h_all, t_all


def learnMRC(fcns, M_init, r_init, c_init, P_true, iters=1000):
    """Find an M, r and c such that sinkhorn(M, r, c) matches P_true. Return the learning curve."""

    h_all = []
    for fcn in fcns:
        M = M_init.clone(); M.requires_grad = True
        r = r_init.clone(); r.requires_grad = True
        c = c_init.clone(); c.requires_grad = True

        optimizer = optim.AdamW([M, r, c], lr=1.0e-2)

        h = []
        for i in range(iters):
            optimizer.zero_grad(set_to_none=True)
            nr = normalize(torch.abs(r))
            nc = normalize(torch.abs(c))
            J = torch.linalg.norm(fcn(M, nr, nc) - P_true)
            h.append(float(J.item()))
            J.backward()
            optimizer.step()
            
        h_all.append(h)

    return h_all


def toy_example():
    # test sinkhorn, approximate gradient and implicit gradient

    torch.manual_seed(0)
    M_true = torch.randn((2, 50, 50), dtype=torch.float)
    #M_true = torch.log(torch.rand((2, 50, 50), dtype=torch.float))
    r_true = normalize(torch.rand((1, 50), dtype=M_true.dtype), p=1.0)
    c_true = normalize(torch.rand((1, 50), dtype=M_true.dtype), p=1.0)

    fcns = [sinkhorn, OptimalTransportLayer(method='approx'), OptimalTransportLayer()]

    # calibrated (uniform)
    print("Learning calibrated (uniform) models...")
    P_true = sinkhorn(M_true)
    M_init = torch.log(torch.rand_like(M_true))
    M_good, h_good, t_good = learnM(fcns, M_init, None, None, P_true)

    # calibrated (non-uniform)
    print("Learning calibrated (non-uniform) models...")
    P_true = sinkhorn(M_true, r_true, c_true)
    M_init = torch.log(torch.rand_like(M_true))
    M_good2, h_good2, t_good2 = learnM(fcns, M_init, r_true, c_true, P_true)

    # mis-calibrated
    print("Learning mis-calibrated models...")
    P_true = sinkhorn(M_true, r_true, c_true)
    M_bad, h_bad, t_bad = learnM(fcns, M_init, None, None, P_true)

    # learning M, r and c
    fcns = [OptimalTransportLayer()]
    r_init = normalize(torch.rand_like(r_true), p=1.0)
    c_init = normalize(torch.rand_like(c_true), p=1.0)
    h_mrc = learnMRC(fcns, M_init, r_init, c_init, P_true)

    print("...done")

    # plot learning curves
    plt.figure()
    plt.semilogy(h_good[0]); plt.semilogy(h_good[1]); plt.semilogy(h_good[2])
    plt.title('Calibrated Model (Uniform)'); plt.xlabel('iteration'); plt.ylabel('loss (log scale)')
    plt.legend(['autograd', 'approx', 'implicit'])

    plt.figure()
    plt.semilogy(h_good2[0]); plt.semilogy(h_good2[1]); plt.semilogy(h_good2[2])
    plt.title('Calibrated Model (Non-uniform)'); plt.xlabel('iteration'); plt.ylabel('loss (log scale)')
    plt.legend(['autograd', 'approx', 'implicit'])

    plt.figure()
    plt.semilogy(h_bad[0]); plt.semilogy(h_bad[1]); plt.semilogy(h_bad[2]); plt.semilogy(h_mrc[0])
    plt.title('Mis-calibrated Model'); plt.xlabel('iteration'); plt.ylabel('loss (log scale)')
    plt.legend(['autograd', 'approx', 'implicit', 'implicit w/ r and c'])


# --- Speed and Memory Comparison ---------------------------------------------

def speed_memory_test(device=None, batch_size=1, repeats=100):
    """Run speed and memory tests."""

    torch.manual_seed(0)
    if device is None:
        device = torch.device('cpu')

    n = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    t = [[], [], [], []]
    m = [[], [], [], []]

    fcns = [sinkhorn, OptimalTransportLayer(method='approx'), OptimalTransportLayer(method='full'), OptimalTransportLayer()]
    for ni in n:
        print("Profiling on {}-by-{} problem...".format(ni, ni))
        M_true = torch.randn((batch_size, ni, ni), dtype=torch.float)
        #M_true = torch.log(torch.rand((batch_size, ni, ni), dtype=torch.float))
        P_true = sinkhorn(M_true).detach().to(device)

        M_init = torch.log(torch.rand_like(M_true) + 1.0e-16).to(device)

        # profile speed
        for i in range(len(fcns)):
            try:
                _, _, ti = learnM((fcns[i],), M_init, None, None, P_true, repeats)
                t[i].append(ti[0])
            except:
                t[i].append(float('nan'))

            torch.cuda.empty_cache()
        
        # profile memory
        for i, f in enumerate(fcns):
            try:
                if device == torch.device("cpu"):
                    with profiler.profile(profile_memory=True) as prof:
                        _ = learnM([f], M_init, None, None, P_true, 1)
                    m[i].append(prof.total_average().cpu_memory_usage)
                else:
                    torch.cuda.reset_peak_memory_stats()
                    _ = learnM([f], M_init, None, None, P_true, 1)
                    m[i].append(torch.cuda.max_memory_allocated(None))
            except:
                m[i].append(float('nan'))

            torch.cuda.empty_cache()

    print("...done")

    _mb = 1.0 / (1024.0 * 1024.0)

    print("-" * 80)
    print("Profiling results on {}".format(device))
    print("-" * 80)
    print("{:<4}  {:<18} {:<18} {:<18} {:<18}".format("",
        'autograd', 'approx', 'implicit (full)', 'implicit (blk)'))
    for i in range(len(n)):
        print("{:<4}  {:6.1f}s  {:6.1f}MB  {:6.1f}s  {:6.1f}MB  {:6.1f}s  {:6.1f}MB  {:6.1f}s  {:6.1f}MB".format(n[i],
            t[0][i], m[0][i] * _mb,
            t[1][i], m[1][i] * _mb,
            t[2][i], m[2][i] * _mb,
            t[3][i], m[3][i] * _mb))
    
    plt.figure()
    plt.plot(n, t[0], n, t[1], n, t[2], n, t[3])
    plt.xlabel('problem size')
    plt.ylabel('running time')
    plt.legend(['autograd', 'approx', 'implicit (full inv)', 'implicit (blk inv)'])
    plt.title('Running time on {} with batch size {}'.format(device, batch_size))

    plt.figure()
    plt.plot(n, m[0], n, m[1], n, m[2], n, m[3])
    plt.xlabel('problem size')
    plt.ylabel('memory usage')
    plt.legend(['autograd', 'approx', 'implicit (full inv)', 'implicit (blk inv)'])
    plt.title('Memory usage on {} with batch size {}'.format(device, batch_size))


# --- Draw Time and Memory Curves Identical with The Tutorials ----------------
# A slightly modified copy of optimal transport tutorial for better visualization

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def plot_running_time(batch_size, device, enable_legend=False):
    """Plot running time for given device."""

    torch.manual_seed(22)
    print("Running on {} with batch size of {}...".format(device, batch_size))

    n = [5, 10, 25, 50, 100, 200, 300, 500]
    t1, t2, t3, t4 = [], [], [], []

    for ni in n:
        print("Timing on {}-by-{} problem...".format(ni, ni))
        M_true = torch.randn((batch_size, ni, ni), dtype=torch.float)
        P_true = sinkhorn(M_true).to(device)
        M_init = torch.log(torch.rand_like(M_true)).to(device)

        t1.append(timeit(wrapper(learnM, [sinkhorn], M_init, None, None, P_true, iters=500), number=1))
        t3.append(timeit(wrapper(learnM, [OptimalTransportLayer(method='full')], M_init, None, None, P_true, iters=500), number=1))
        t2.append(timeit(wrapper(learnM, [OptimalTransportLayer(method='approx')], M_init, None, None, P_true, iters=500), number=1))
        t4.append(timeit(wrapper(learnM, [OptimalTransportLayer()], M_init, None, None, P_true, iters=500), number=1))

    print("...done")

    plt.figure(figsize=(7, 7))
    plt.plot(n, t1, marker='x', markersize=14)
    plt.plot(n, t2, marker='*', markersize=14)
    plt.plot(n, t3, marker='o', markersize=14)
    plt.plot(n, t4, marker='<', markersize=14)
    plt.xlabel('problem size', fontsize=30)
    plt.ylabel('running time (s)', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20, rotation=90)
    # plt.title('Running time on {} with batch size {}'.format(device, batch_size), fontsize=30)
    plt.tight_layout()

    if enable_legend:
        plt.legend(['autograd', 'approx', 'implicit (full inv)', 'implicit (blk inv)'], fontsize=30)



def plot_memory():
    M_init = torch.randn((1, 500, 500), dtype=torch.float)

    maxiters_range = list(range(1, 11))
    probsize_range = [5, 10, 25, 50, 100, 200, 500, 800, 1000]

    memory_by_maxiters = [[], []]
    memory_by_probsize = [[], []]

    for maxiters in maxiters_range:
        # profile autograd
        M = M_init.clone()
        M.requires_grad = True
        with profiler.profile(profile_memory=True) as prof:
            P = sinkhorn(M, eps=0.0, maxiters=maxiters)
            torch.linalg.norm(P - torch.eye(M.shape[1])).backward()
        memory_by_maxiters[0].append(prof.total_average().cpu_memory_usage / (1024 * 1024))

        # profile implicit
        M = M_init.clone()
        M.requires_grad = True
        f = OptimalTransportLayer(eps=0.0, maxiters=maxiters)
        with profiler.profile(profile_memory=True) as prof:
            P = f(M)
            torch.linalg.norm(P - torch.eye(M.shape[1])).backward()
        memory_by_maxiters[1].append(prof.total_average().cpu_memory_usage / (1024 * 1024))

    for n in probsize_range:
        M_init = torch.randn((1, n, n), dtype=torch.float)

        # profile autograd
        M = M_init.clone()
        M.requires_grad = True
        with profiler.profile(profile_memory=True) as prof:
            P = sinkhorn(M, eps=0.0, maxiters=10)
            torch.linalg.norm(P - torch.eye(n)).backward()
        memory_by_probsize[0].append(prof.total_average().cpu_memory_usage / (1024 * 1024))

        # profile implicit
        M = M_init.clone()
        M.requires_grad = True
        f = OptimalTransportLayer(eps=0.0, maxiters=10)
        with profiler.profile(profile_memory=True) as prof:
            P = f(M)
            torch.linalg.norm(P - torch.eye(n)).backward()
        memory_by_probsize[1].append(prof.total_average().cpu_memory_usage / (1024 * 1024))

    plt.figure(figsize=(7, 7))
    plt.plot(maxiters_range, memory_by_maxiters[0], linestyle='-')
    plt.plot(maxiters_range, memory_by_maxiters[1], linestyle='-.')
    plt.xlabel('iterations', fontsize=30)
    plt.ylabel('memory usage (MB)', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20, rotation=90)
    plt.legend(['autograd', 'implicit'], fontsize=30)
    # plt.title("Memory usage for problem of size 500-by-500", fontsize=30)
    plt.tight_layout()

    plt.figure(figsize=(7, 7))
    plt.plot(probsize_range, memory_by_probsize[0], linestyle='-')
    plt.plot(probsize_range, memory_by_probsize[1], linestyle='-.')
    plt.xlabel('problem size', fontsize=30)
    plt.ylabel('memory usage (MB)', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20, rotation=90)
    # plt.legend(['autograd', 'implicit'], fontsize=30)
    # plt.title("Memory usage for 10 Sinkhorn iterations", fontsize=30)
    plt.tight_layout()


# --- Run Unit Tests ----------------------------------------------------------

if __name__ == '__main__':
    # Set `True` to run unit tests and then terminate. Set `False` to allow other tests to run.
    if False:
        unittest.main()

    if True:
        toy_example()

    if True:
        speed_memory_test(torch.device('cpu'))
        if torch.cuda.is_available():
            speed_memory_test(torch.device("cuda"), batch_size=16)

    if True:
        plot_running_time(1, torch.device("cpu"), enable_legend=True)
        if torch.cuda.is_available():
            plot_running_time(16, torch.device("cuda"), enable_legend=False)
        plot_memory()

    plt.show()
