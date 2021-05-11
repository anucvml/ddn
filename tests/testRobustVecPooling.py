# TEST ROBUST VECTOR POOLING DEEP DECLARATIVE NODE
#
# Stephen Gould <stephen.gould@anu.edu.au>
#
# When running from the command-line make sure that the "ddn" package has been added to the PYTHONPATH:
#   $ export PYTHONPATH=${PYTHONPATH}: ../ddn
#   $ python testOptimalTransport.py
#

import torch
import torch.optim as optim
from torch.autograd import gradcheck
import torch.autograd.profiler as profiler

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

import sys
sys.path.append("../")
import ddn.pytorch.robust_vec_pool as rvp

import unittest

torch.manual_seed(0)

# --- Speed and Memory Comparison ---------------------------------------------

def speed_memory_test(device=None, batch_size=1, outlier_ratio=0.1, repeats=10):
    """Run speed and memory tests."""

    torch.manual_seed(0)
    if device is None:
        device = torch.device('cpu')

    n = [10, 100]
    f = [16, 256, 1024]

    penalties = [rvp.Quadratic, rvp.PseudoHuber, rvp.Huber, rvp.Welsch, rvp.TruncQuad]
    t_fwd = [[] for p in penalties]
    t_bck = [[] for p in penalties]
    m_fwd = [[] for p in penalties]
    m_bck = [[] for p in penalties]

    for ni in n:
        for fi in f:
            print("Profiling on ({}, {}, {}, {})...".format(batch_size, fi, ni, ni), end='')

            x = torch.randn((batch_size, fi, ni, ni), dtype=torch.float, device=device)
            x = torch.where(torch.rand_like(x) < outlier_ratio, 10.0 * x, x)
            x.requires_grad = True

            fcn = rvp.RobustVectorPool2dFcn().apply

            # profile time
            for i, p in enumerate(penalties):
                print(".", end='')

                y = [[] for j in range(repeats)]

                start_time = time.monotonic()
                for j in range(repeats):
                    y[j] = fcn(x.clone(), p, 1.0)
                t_fwd[i].append((time.monotonic() - start_time) / repeats)

                start_time = time.monotonic()
                for j in range(repeats):
                    loss = torch.linalg.norm(y[j].view(batch_size, fi, -1), dim=1).sum()
                    loss.backward()
                t_bck[i].append((time.monotonic() - start_time) / repeats)

            # profile memory
            for i, p in enumerate(penalties):
                print(".", end='')
                with profiler.profile(profile_memory=True) as prof:
                    y = fcn(x.clone(), p, 1.0)
                m_fwd[i].append(prof.total_average().cpu_memory_usage / (1024 * 1024))

                with profiler.profile(profile_memory=True) as prof:
                    loss = torch.linalg.norm(y.view(batch_size, fi, -1), dim=1).sum()
                    loss.backward()
                m_bck[i].append(prof.total_average().cpu_memory_usage / (1024 * 1024))

            print("")

    print(t_fwd)
    print(t_bck)
    print(m_fwd)
    print(m_bck)

# --- Optimization Example (from Jupyter tutorial) ----------------------------

def toy_example(iters=100):

    # setup data with outliers
    torch.manual_seed(0)

    x_init = 1.5 * (torch.rand(1, 2, 10) - 0.5)
    x_init[0, 0, 9] += 6.0; x_init[0, 1, 9] -= 2.0
    x_init[0, 0, 0] += 3.0; x_init[0, 1, 0] += 2.0

    opt_data = [('quadratic', rvp.Quadratic, 'b', x_init.clone(), []),
                ('pseudo-huber', rvp.PseudoHuber, 'r', x_init.clone(), []),
                ('welsch', rvp.Welsch, 'g', x_init.clone(), [])]


    for name, penalty, colour, x_init, history in opt_data:
        print("Running {}...".format(name))

        x = x_init.clone()
        x.requires_grad = True
        optimizer = optim.SGD([x], lr=0.2)
        for i in range(iters):
            optimizer.zero_grad(set_to_none=True)
            y = rvp.RobustVectorPool2dFcn().apply(x, penalty)
            J = torch.sum(torch.square(y))
            history.append((x.clone().detach(), y.clone().detach()))
            J.backward()
            optimizer.step()

    plt.figure()
    for name, penalty, colour, x_init, history in opt_data:
        plt.plot([torch.sum(torch.square(y)).item() for x, y in history], color=colour)
    plt.xlabel('iters.'); plt.ylabel('loss')

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x_init[0, 0, :], x_init[0, 1, :], 'o', markeredgecolor='k', markerfacecolor='w', markeredgewidth=1.0)
    plt.gca().set_xlim(-2.0, 7.0); plt.gca().set_ylim(-3.0, 3.0)
    for i, (name, penalty, colour, x_init, history) in enumerate(opt_data):
        plt.plot(history[0][1][0, 0], history[0][1][0, 1], 'D', markeredgecolor='k', markerfacecolor=colour, markeredgewidth=1.0)

    for i, (name, penalty, colour, x_init, history) in enumerate(opt_data):
        plt.subplot(2, 2, i+2)
        plt.plot(x_init[0, 0, :], x_init[0, 1, :], 'o', markeredgecolor='k', markerfacecolor='w', markeredgewidth=1.0)
        plt.plot(history[-1][0][0, 0, :], history[-1][0][0, 1, :], 'o', markeredgecolor='k', markerfacecolor=colour, markeredgewidth=1.0)
        plt.gca().set_xlim(-2.0, 7.0); plt.gca().set_ylim(-3.0, 3.0)

    plt.show()


# --- Run Unit Tests ----------------------------------------------------------

if __name__ == '__main__':
    if True:
        toy_example()

    if False:
        speed_memory_test(torch.device('cpu'))
        if torch.cuda.is_available():
            speed_memory_test(torch.device("cuda"), batch_size=16)
