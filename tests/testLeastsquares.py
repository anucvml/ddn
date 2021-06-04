# TEST LEAST SQUARES DEEP DECLARATIVE NODE
#
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import torch, time, sys
import torch.autograd.profiler as profiler
sys.path.append("..")
from ddn.pytorch.leastsquares import WeightedLeastSquaresFcn

#
# --- Test Speed and Time ---
#

def speed_memory_test(inverse_mode, enable_bias, cache_decomposition, fcn, data_sz_list, num_iter_speed=5000, num_iter_memory=5, device=None):
    device = torch.device('cpu') if (device is None) else device
    data_sz_list = [data_sz_list] if (not isinstance(data_sz_list, list)) else data_sz_list
    time_forward, time_backward, memory = [], [], []

    for data_sz in data_sz_list:
        B, C, T = data_sz
        time_forward_total, time_backward_total = 0, 0

        # Speed, the first loop is ignored
        for idx in range(num_iter_speed):
            X = torch.randn((B, C, T), dtype=torch.double, device=device, requires_grad=True)
            W1 = torch.rand((B, 1, T), dtype=torch.double, device=device, requires_grad=True)
            T1 = torch.rand((B, 1, T), dtype=torch.double, device=device, requires_grad=True)

            time_start = time.monotonic()
            grad, grad0 = fcn(X, T1, W1, 1.0e-3, cache_decomposition, enable_bias, inverse_mode)
            duration = time.monotonic() - time_start if (idx > 0) else 0
            time_forward_total += duration

            loss = grad.mean() + grad0.mean()

            time_start = time.monotonic()
            loss.backward()
            duration = time.monotonic() - time_start if (idx > 0) else 0
            time_backward_total += duration

        time_forward.append(time_forward_total * 1000 / (num_iter_speed - 1))
        time_backward.append(time_backward_total * 1000 / (num_iter_speed - 1))

        # Memory, the first loop is ignored, set num_iter_memory small for fast test
        if device == torch.device('cpu'):
            with profiler.profile(profile_memory=True) as prof:
                for idx in range(num_iter_memory):
                    X = torch.randn((B, C, T), dtype=torch.double, device=device, requires_grad=True)
                    W1 = torch.rand((B, 1, T), dtype=torch.double, device=device, requires_grad=True)
                    T1 = torch.rand((B, 1, T), dtype=torch.double, device=device, requires_grad=True)

                    # Define a function to avoid the warning of preallocated inputs in the memory profile
                    def test_fcn_cpu_memory(X, T1, W1, cache_decomposition, enable_bias, inverse_mode):
                        grad, grad0 = fcn(X, T1, W1, 1.0e-3, cache_decomposition, enable_bias, inverse_mode)
                        loss = grad.mean() + grad0.mean()
                        loss.backward()

                    test_fcn_cpu_memory(X, T1, W1, cache_decomposition, enable_bias, inverse_mode)

            memory.append(prof.total_average().cpu_memory_usage / (1024 * 1024))
        else:
            memory_total = 0
            for idx in range(num_iter_memory):
                X = torch.randn((B, C, T), dtype=torch.double, device=device, requires_grad=True)
                W1 = torch.rand((B, 1, T), dtype=torch.double, device=device, requires_grad=True)
                T1 = torch.rand((B, 1, T), dtype=torch.double, device=device, requires_grad=True)

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                grad, grad0 = fcn(X.clone(), T1.clone(), W1.clone(), 1.0e-3, cache_decomposition, enable_bias, inverse_mode)
                loss = grad.mean() + grad0.mean()
                loss.backward()

                memory_current = torch.cuda.max_memory_allocated(None) if (idx > 0) else 0
                memory_total += memory_current

            memory.append(memory_total / (1024 * 1024 * (num_iter_memory - 1)))

    return time_forward, time_backward, memory


if __name__ == '__main__':
    B, C, T = 2, 64, 12
    # device = torch.device("cuda")
    device = torch.device("cpu")
    inverse_mode_list = ['cholesky', 'qr']
    enable_bias_list = [True, False]
    cache_decomp_list = [True, False]
    f = WeightedLeastSquaresFcn.apply

    X = torch.randn((B, C, T), dtype=torch.double, device=device, requires_grad=True)
    W1 = torch.rand((B, 1, T), dtype=torch.double, device=device, requires_grad=True)
    T1 = torch.rand((B, 1, T), dtype=torch.double, device=device, requires_grad=True)
    W2 = torch.rand((1, 1, T), dtype=torch.double, device=device, requires_grad=True)
    T2 = torch.rand((1, 1, T), dtype=torch.double, device=device, requires_grad=True)

    for inverse_mode in inverse_mode_list:
        beta_test = 1.0e-3 if (inverse_mode == 'qr') else 0.0

        for enable_bias in enable_bias_list:

            for cache_decomp in cache_decomp_list:
                # Time comparison: QR is slower than Cholesky, the size of A is larger than AtA, (m+n)*(n+1) vs (n+1)*(n+1)
                data_sz_list = [(B, C, T)]
                time_forward, time_backward, memory = speed_memory_test(inverse_mode,
                                                                        enable_bias,
                                                                        cache_decomp,
                                                                        f,
                                                                        data_sz_list,
                                                                        num_iter_speed=5000,
                                                                        num_iter_memory=5,
                                                                        device=device)

                for idx, data_sz in enumerate(data_sz_list):
                    print('Time/memory, mode: {}, bias: {}, cache: {}, size: {}, forward: {:.4f}ms, backward: {:.4f}ms; memory: {:.4f}MB.' \
                          .format(inverse_mode, enable_bias, cache_decomp, data_sz, time_forward[idx], time_backward[idx], memory[idx]))
