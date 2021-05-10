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
from torch.nn.functional import normalize
from torch.autograd import gradcheck
import torch.autograd.profiler as profiler

import time
import matplotlib.pyplot as plt

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


# --- Run Unit Tests ----------------------------------------------------------

if __name__ == '__main__':
    speed_memory_test(torch.device('cpu'))
    if torch.cuda.is_available():
        speed_memory_test(torch.device("cuda"), batch_size=16)
