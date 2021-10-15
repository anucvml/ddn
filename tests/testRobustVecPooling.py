# TEST ROBUST VECTOR POOLING DEEP DECLARATIVE NODE
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
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


# --- Display Figures with Different Feature Sizes ---------------------------------------------

# Features: (method,feat_dim,feat_size) using idx_feat_dim for feat_dim and idx_feat_size for feat_size,
# The motivation: one can fix eithor or both of feat_dim and feat_size; if both are varied,
# one feat_dim and feat_size are one-to-one correspondent.
# Notation: method<-penalities, feat_dims<-f, feat_sizes<-n
def draw_figure(data, feat_dims, feat_sizes, sizes, disp_type, title, legend, markers=None, fixed_feat_dim=None, fixed_feat_size=None):
    plt.figure(figsize=(7, 7))
    num_methods = len(data)  # equal len(penalties)
    num_feat_nm = len(feat_dims)
    num_feat_sz = len(feat_sizes)

    for idx_method in range(len(data)):
        y_list = []
        data_per = data[idx_method]

        for idx_size, disp_size in enumerate(sizes):
            feat_dim_stored, feat_size_stored = int(disp_size[1]), int(disp_size[2])

            if (fixed_feat_dim is None) and (fixed_feat_size is None):
                num_valid = min(num_feat_nm, num_feat_sz)
                x_list = feat_sizes[:num_valid]

                # Both increase, one-by-one order between feat_dims and feat_sizes
                for feat_dim_tar, feat_size_tar in zip(feat_dims[:num_valid], feat_sizes[:num_valid]):
                    y_list.append(data_per[idx_size]) if ((feat_dim_tar == feat_dim_stored) and \
                        (feat_size_tar == feat_size_stored)) else None
                x_label = 'feature map size'
            elif (fixed_feat_dim is None) and (fixed_feat_size is not None):
                # Fix feat_size
                x_list = feat_dims
                y_list.append(data_per[idx_size]) if (fixed_feat_size == feat_size_stored) else None
                x_label = 'feature map channel'
            elif (fixed_feat_dim is not None) and (fixed_feat_size is None):
                # Fix feat_dim
                x_list = feat_sizes
                y_list.append(data_per[idx_size]) if (fixed_feat_dim == feat_dim_stored) else None
                x_label = 'feature map size'
            else:
                # Fix both, display a point
                x_list = [feat_sizes[fixed_feat_size]]
                y_list.append(data_per[idx_size]) if ((fixed_feat_size == feat_size_stored) and \
                    (fixed_feat_dim == feat_dim_stored)) else None
                x_label = 'feature map channel {} & size {}-by-{}' \
                    .format(fixed_feat_dim, fixed_feat_size, fixed_feat_size)

        if markers is None:
            plt.plot(x_list, y_list)
        else:
            plt.plot(x_list, y_list, marker=markers[idx_method], markersize=14)

    plt.xlabel(x_label, fontsize=30)
    plt.ylabel('running time (ms)' if (disp_type == 'time') else r'memory usage (MB)', fontsize=30)
    # plt.title(title, fontsize=30)
    # plt.legend(legend, fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20, rotation=60)
    plt.ticklabel_format(axis='y', style='plain', scilimits=(0, 0))
    plt.tight_layout()

def plot_figures(penalties, t_fwd, t_bck, m_fwd, m_bck, feat_dims, feat_sizes, sizes, device, batch_size):
    # plot figures
    t_list = [[] for _ in penalties]
    m_list = [[] for _ in penalties]
    _ms = 1000.0
    _mb = 1.0 / (1024.0 * 1024.0)

    for idx_method in range(len(penalties)):
        for idx in range(len(t_fwd[idx_method])):
            t_fwd[idx_method][idx] *= _ms
            t_bck[idx_method][idx] *= _ms
            t_list[idx_method].append(t_fwd[idx_method][idx] + t_bck[idx_method][idx])

            m_fwd[idx_method][idx] *= _mb
            m_bck[idx_method][idx] *= _mb
            m_list[idx_method].append(m_fwd[idx_method][idx] + m_bck[idx_method][idx])


    dir_list = ['Forward', 'Backward']  # Forward/Backward/Total
    type_list = ['time', 'memory']
    legend = ['quadratic', 'pseudo-huber', 'huber', 'welsch', 'trunc-quad']
    markers = ['x', '*', 'o', '<', '^']
    fixed_feat_dim, fixed_feat_size = 128, None

    for disp_dir in dir_list:
        for disp_type in type_list:
            if (disp_dir == 'Forward') and (disp_type == 'time'):
                data = t_fwd
            elif (disp_dir == 'Forward') and (disp_type == 'memory'):
                data = m_fwd
            elif (disp_dir == 'Backward') and (disp_type == 'time'):
                data = t_bck
            elif (disp_dir == 'Backward') and (disp_type == 'memory'):
                data = m_bck
            elif (disp_dir == 'Total') and (disp_type == 'time'):
                data = t_list
            elif (disp_dir == 'Total') and (disp_type == 'memory'):
                data = m_list
            else:
                assert False, '!!!Unknown direction {} or type {}.'.format(disp_dir, disp_type)

            title = '{} {} on {} with batch size {}'.format(disp_dir, disp_type, device, batch_size)
            draw_figure(data, feat_dims, feat_sizes, sizes, disp_type, title, legend, markers=markers,
                fixed_feat_dim=fixed_feat_dim, fixed_feat_size=fixed_feat_size)


# --- Speed and Memory Comparison ---------------------------------------------

def speed_memory_test(device=None, batch_size=1, outlier_ratio=0.1, repeats=10):
    """Run speed and memory tests."""

    torch.manual_seed(0)
    if device is None:
        device = torch.device('cpu')

    n = [10, 40, 80, 100, 200, 500]
    f = [128]  # [8, 16, 32, 64, 128, 256]

    penalties = [rvp.Quadratic, rvp.PseudoHuber, rvp.Huber, rvp.Welsch, rvp.TruncQuad]
    t_fwd = [[] for p in penalties]
    t_bck = [[] for p in penalties]
    m_fwd = [[] for p in penalties]
    m_bck = [[] for p in penalties]
    sizes = []
    
    for ni in n:
        for fi in f:
            sizes.append((batch_size, fi, ni, ni))
            print("Profiling on ({}, {}, {}, {})...".format(batch_size, fi, ni, ni), end='')

            x = torch.randn((batch_size, fi, ni, ni), dtype=torch.float, device=device)
            x = torch.where(torch.rand_like(x) < outlier_ratio, 10.0 * x, x)
            x.requires_grad = True

            fcn = rvp.RobustVectorPool2dFcn().apply

            # profile time
            for i, p in enumerate(penalties):
                print(".", end='')
                y = [None for j in range(repeats)]

                torch.cuda.empty_cache()
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

                if device == torch.device("cpu"):
                    with profiler.profile(profile_memory=True) as prof:
                        x_clone = x.clone().detach()
                        x_clone.requires_grad = True
                        y = fcn(x_clone, p, 1.0)
                    m_fwd[i].append(prof.total_average().cpu_memory_usage)
                    #m_fwd[i].append(max([evt.cpu_memory_usage for evt in prof.function_events]))

                    with profiler.profile(profile_memory=True) as prof:
                        loss = torch.linalg.norm(y.view(batch_size, fi, -1), dim=1).sum()
                        loss.backward()
                    m_bck[i].append(prof.total_average().cpu_memory_usage)
                    #m_bck[i].append(max([evt.cpu_memory_usage for evt in prof.function_events]))
                else:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    y = fcn(x.clone(), p, 1.0)
                    m_fwd[i].append(torch.cuda.max_memory_allocated(None))

                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    loss = torch.linalg.norm(y.view(batch_size, fi, -1), dim=1).sum()
                    loss.backward()
                    m_bck[i].append(torch.cuda.max_memory_allocated(None))
                    
            print("")

    _ms = 1000.0
    _mb = 1.0 / (1024.0 * 1024.0)
    
    print("-" * 80)
    print("Profiling results on {}".format(device))
    print("-" * 80)
    print("{:<16}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format("", "fwd time", "bck time", "fwd mem", "bck mem"))
    for i, p in enumerate(penalties):
        print("--- {} ---".format(p.__name__))
        for j, sz in enumerate(sizes):
            print("{:<16}\t{:6.1f}ms\t{:6.1f}ms\t{:6.1f}MB\t{:6.1f}MB".format(str(sz),
                t_fwd[i][j] * _ms, t_bck[i][j] * _ms, m_fwd[i][j] * _mb, m_bck[i][j] * _mb))

    return penalties, t_fwd, t_bck, m_fwd, m_bck, f, n, sizes


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


# --- Run Unit Tests ----------------------------------------------------------

if __name__ == '__main__':
    if False:
        toy_example()

    # Sepearte CPu and GPU test since they cannot display all figures across
    # different modules, I think so
    if True:
        penalties, t_fwd, t_bck, m_fwd, m_bck, feat_dims, feat_sizes, sizes = \
            speed_memory_test(torch.device('cpu'))
        plot_figures(penalties, t_fwd, t_bck, m_fwd, m_bck, feat_dims, feat_sizes, sizes,
            torch.device('cpu'), 1)

        if torch.cuda.is_available():
            batch_size = 8
            # Run twice speed_memory_test, the first time is ignored due to the long
            # time consuming for triggering CUDA to avoid unfair comparison
            speed_memory_test(torch.device("cuda"), batch_size=batch_size, repeats=1)

            penalties, t_fwd, t_bck, m_fwd, m_bck, feat_dims, feat_sizes, sizes = \
                speed_memory_test(torch.device("cuda"), batch_size=batch_size, repeats=1)
            plot_figures(penalties, t_fwd, t_bck, m_fwd, m_bck, feat_dims, feat_sizes, sizes,
                torch.device("cuda"), batch_size)

    plt.show()
