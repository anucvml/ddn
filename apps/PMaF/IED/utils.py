#!/usr/bin/env python
#
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import matplotlib as mpl
# mpl.use('tkagg')
import numpy as np
import pandas as pd
import torch, time, csv, os, torchvision
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------

def method_mode_explaination(enable_display=False):
    if enable_display:
        print('')
        print('=====================')
        print('AT: forward solver with autogradient eigh().')
        print('+dLdx_DDN_fcn: use auto Jacobian to calculate A, B, H for dydx.')
        print('+nn_back: auto backpropagate eigh().')
        print('')
        print('SI: forward solver with Simultaneous Iteration with QR decomposition.')
        print('+unroll: unroll to auto backpropagate SI algorithm.')
        print('+nn_back: use exploited structure to calculate A, B, H for dydx.')
        print('=====================')

# ----------------------------------------------------------------------------------------------------------------------

def generate_random_data(
        b, m, n, seed, device='cuda', enable_symmetric=False, dtype=torch.float32, enable_grad_one=False,
        distribution_mode='gaussian', uniform_sample_max=1.0, choice_max=10.0):
    """
    Random data generation
    :param b:
    :param m:
    :param n:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # This makes PI and SI better than eigh()
    if distribution_mode.find('resnet50') > -1:
        data_size = (b, 3, 256, 256)
    else:
        data_size = (b, m, m)

    if distribution_mode.find('uniform') > -1:
        x = np.random.uniform(0, uniform_sample_max, data_size)
    elif distribution_mode.find('vonmise') > -1:
        x = np.random.vonmises(0, 1, size=data_size)
    elif distribution_mode.find('choice') > -1:
        x = np.random.choice(np.arange(choice_max), data_size)
    else:
        x = np.random.randn(*data_size)

    if distribution_mode.find('resnet50') > -1:
        x = torch.from_numpy(x).float().cuda()
        model = torchvision.models.resnet50(pretrained=True).cuda()
        model.avgpool = torch.nn.Identity()
        model.fc = torch.nn.Linear(2048 * 8 * 8, m * m, device='cuda')
        x = model(x)
        x = x.reshape(b, m, m).cpu().numpy()

    # Ensure the greatest Eigenvalus is positive
    x = np.abs(x)

    if enable_symmetric:
        x = x + x.transpose(0, 2, 1)
        x = torch.tensor(x, dtype=dtype, requires_grad=True, device=device)
    else:
        x = torch.tensor(x, dtype=dtype, requires_grad=True, device=device)

    if enable_grad_one:
        dLdy = torch.ones(b, m, n, dtype=x.dtype, device=x.device) # mimic loss for sum over y
    else:
        dLdy = torch.randn(b, m, n, dtype=x.dtype, device=x.device)

    return x, dLdy

# ----------------------------------------------------------------------------------------------------------------------

def mergecells(table, ix0, ix1):
    ix0,ix1 = np.asarray(ix0), np.asarray(ix1)
    d = ix1 - ix0
    if not (0 in d and 1 in np.abs(d)):
        raise ValueError("ix0 and ix1 should be the indices of adjacent cells. ix0: %s, ix1: %s" % (ix0, ix1))

    if d[0]==-1:
        edges = ('BRL', 'TRL')
    elif d[0]==1:
        edges = ('TRL', 'BRL')
    elif d[1]==-1:
        edges = ('BTR', 'BTL')
    else:
        edges = ('BTL', 'BTR')

    # hide the merged edges
    for ix,e in zip((ix0, ix1), edges):
        table[ix[0], ix[1]].visible_edges = e

    txts = [table[ix[0], ix[1]].get_text() for ix in (ix0, ix1)]
    tpos = [np.array(t.get_position()) for t in txts]

    # center the text of the 0th cell between the two merged cells
    trans = (tpos[1] - tpos[0])/2
    if trans[0] > 0 and txts[0].get_ha() == 'right':
        # reduce the transform distance in order to center the text
        trans[0] /= 2
    elif trans[0] < 0 and txts[0].get_ha() == 'right':
        # increase the transform distance...
        trans[0] *= 2

    txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))

    # hide the text in the 1st cell
    txts[1].set_visible(False)

# ----------------------------------------------------------------------------------------------------------------------

def run_speed_memory_statistics(num_seeds, methods, data_sizes, mode_dict, obj_dict, enable_symmetric, dtype):
    info = {}

    for method in methods:
        for mode in mode_dict[method]:
            method_mode = '{}+{}'.format(method, mode)
            info.update({method_mode: {}})

            for data_size in data_sizes:
                print('[ Method: {}, Size: {} ]'.format(method_mode, data_size))
                batch, m, n = data_size
                size_str = 'b{}_m{}_n{}'.format(batch, m, n)
                current_info = info[method_mode]
                current_info.update(
                    {size_str: {'time': {'cpu': {'forward': 0.0, 'backward': 0.0},
                                         'gpu': {'forward': 0.0, 'backward': 0.0}},
                                'memory': {'cpu': {'all': 0.0, 'forward': 0.0, 'backward': 0.0},
                                           'gpu': {'all': 0.0, 'forward': 0.0, 'backward': 0.0}},
                                'count': 0}})
                obj = obj_dict[method]
                obj.set_num_eigen_values(n)
                current_info = current_info[size_str]

                # ==== Time
                for device in ['cpu', 'gpu']:
                    current_info['count'] = 0
                    data_device = 'cuda' if (device == 'gpu') else device

                    if device == 'cpu' and m >= 512:
                        print('!!!CPU time skipped.')
                        current_info['time'][device]['forward'] = -1
                        current_info['time'][device]['backward'] = -1
                        current_info['count'] = 1
                    elif device == 'gpu' and method.find('IFT') > -1 and mode == 'dLdx_fnc' and m >= 1024:
                        print('!!!CPU time skipped for IFT.')
                        current_info['time'][device]['forward'] = -1
                        current_info['time'][device]['backward'] = -1
                        current_info['count'] = 1
                    else:
                        for idx, seed in enumerate(range(num_seeds)):
                            torch.cuda.empty_cache()

                            x, dLdy = generate_random_data(
                                batch, m, n, seed, device=data_device, enable_symmetric=enable_symmetric,
                                dtype=dtype, enable_grad_one=True)

                            time_start = time.monotonic()
                            lambds, y = obj(x, backward_fnc_name=mode)
                            forward_duration = time.monotonic() - time_start

                            loss = y.sum()

                            time_start = time.monotonic()
                            loss.backward()
                            backward_duration = time.monotonic() - time_start

                            if idx > 0:
                                current_info['time'][device]['forward'] += forward_duration
                                current_info['time'][device]['backward'] += backward_duration
                                current_info['count'] += 1

                            del x, dLdy, lambds, y, loss

                    current_info['time'][device]['forward'] /= max(1, current_info['count'])
                    current_info['time'][device]['backward'] /= max(1, current_info['count'])

                # ==== CPU memory
                print('!!!CPU memory skipped.')

                # ==== GPU memory
                current_info['count'] = 0

                # IFT with auto Jacobian will be out of memory (due to B with m * (m * m)) so block it here
                if method.find('IFT') > -1 and mode == 'dLdx_fnc' and m >= 1024:
                    print('!!! GPU memory skipped for IFT+auto with size >= 1024.')
                    current_info['memory']['gpu']['forward'] = -1
                    current_info['memory']['gpu']['backward'] = -1
                    current_info['count'] = 1
                    continue

                for idx, seed in enumerate(range(num_seeds)):
                    torch.cuda.empty_cache()

                    x, dLdy = generate_random_data(
                        batch, m, n, seed, device='cuda', enable_symmetric=enable_symmetric, dtype=dtype,
                        enable_grad_one=True)

                    # Forward memory
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    lambds, y = obj(x, backward_fnc_name=mode)

                    # ====
                    forward_memory_used = torch.cuda.max_memory_allocated()

                    loss = y.sum()
                    del x, y, dLdy, lambds

                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    # Backward memory
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    memory_start = torch.cuda.max_memory_allocated()

                    loss.backward()
                    del loss

                    # ====
                    backward_memory_used = torch.cuda.max_memory_allocated() - memory_start

                    if idx > 0:
                        current_info['memory']['gpu']['forward'] += forward_memory_used / (1024 * 1024)
                        current_info['memory']['gpu']['backward'] += backward_memory_used / (1024 * 1024)
                        current_info['count'] += 1

                current_info['memory']['gpu']['forward'] /= max(1, current_info['count'])
                current_info['memory']['gpu']['backward'] /= max(1, current_info['count'])

        print('')  # method

    return info

# ----------------------------------------------------------------------------------------------------------------------

def run_precision_statistics(
        num_seeds, methods, data_sizes, mode_dict, obj_dict, enable_symmetric, dtype, distribution_mode='gaussian',
        uniform_sample_max=1.0, choice_max=10.0):
    info = {}

    for method in methods:
        for mode in mode_dict[method]:
            method_mode = '{}+{}'.format(method, mode)
            info.update({method_mode: {}})

            for data_size in data_sizes:
                print('[ Method: {}, Size: {} ]'.format(method_mode, data_size))
                batch, m, n = data_size
                size_str = 'b{}_m{}_n{}'.format(batch, m, n)
                current_info = info[method_mode]
                current_info.update(
                    {size_str: {'eigen_gap': {'cpu': 0.0, 'gpu': 0.0},
                                'fp_gap': {'cpu': 0.0, 'gpu': 0.0}, 'count': 0}})
                obj = obj_dict[method]
                obj.set_num_eigen_values(n)
                current_info = current_info[size_str]

                if m >= 512: num_seeds = min(num_seeds, 100)

                # ==== precision
                for device in ['gpu']:
                    current_info['count'] = 0
                    data_device = 'cuda' if (device == 'gpu') else device

                    for seed in range(num_seeds):
                        x, dLdy = generate_random_data(
                            batch, m, n, seed, device=data_device, enable_symmetric=enable_symmetric,
                            dtype=dtype, enable_grad_one=True, distribution_mode=distribution_mode,
                            uniform_sample_max=uniform_sample_max, choice_max=choice_max)

                        lambds, y = obj(x, backward_fnc_name=mode)
                        
                        eigen_gap = obj.check_eigen_gap(lambds, y, x)
                        fp_gap = obj.check_fixed_point_gap(y, x)

                        current_info['eigen_gap'][device] += eigen_gap.item() * n * batch
                        current_info['fp_gap'][device] += fp_gap.item() * n * batch
                        current_info['count'] += 1
                        del x, dLdy, lambds, y

                    current_info['eigen_gap'][device] /= max(1, current_info['count']) * n * batch
                    current_info['fp_gap'][device] /= max(1, current_info['count']) * n * batch

            print('')  # method+mode

    return info

# ----------------------------------------------------------------------------------------------------------------------

def visual_speed_memory(cost_info, data_sizes, g_save_path):
    data_size_list = []
    time_cpu = {}
    time_gpu = {}
    memory_cpu = {}
    memory_gpu = {}

    for data_size in data_sizes:
        batch, m, n = data_size
        data_size_list.append("")
        data_size_list.append(f"b {batch}, m {m}, n {n}")

    for method in cost_info.keys():
        time_cpu.update({method: []})
        time_gpu.update({method: []})
        memory_cpu.update({method: []})
        memory_gpu.update({method: []})

        for data_size in data_sizes:
            batch, m, n = data_size
            data_size_str = f"b{batch}_m{m}_n{n}"
            obj = cost_info[method][data_size_str]

            time_obj = obj['time']
            time_cpu[method] += [
                f"{time_obj['cpu']['forward']:.4f}",
                f"{time_obj['cpu']['backward']:.4f}"]
            time_gpu[method] += [
                f"{time_obj['gpu']['forward']:.4f}",
                f"{time_obj['gpu']['backward']:.4f}"]

            memory_obj = obj['memory']
            memory_cpu[method] += [
                f"{memory_obj['cpu']['forward']:.4f}",
                f"{memory_obj['cpu']['backward']:.4f}"]
            memory_gpu[method] += [
                f"{memory_obj['gpu']['forward']:.4f}",
                f"{memory_obj['gpu']['backward']:.4f}"]

    # ==== Save to csv
    os.makedirs(f"{g_save_path}/csv", exist_ok=True)
    csv_head_list = ['Method']

    for v in data_size_list:
        if v != '':
            v = v.replace(' ', '').replace(',', '')
            csv_head_list.append(f"Forward")
            csv_head_list.append(f"Backward")

    csv_path_list = [
        f"{g_save_path}/csv/time_cpu.csv",
        f"{g_save_path}/csv/time_gpu.csv",
        f"{g_save_path}/csv/memory_gpu.csv"
    ]
    value_list = [
        time_cpu,
        time_gpu,
        memory_gpu
    ]

    for values, save_path in zip(value_list, csv_path_list):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_head_list)

            for key_v in values.keys():
                value_list = values[key_v]
                key_v = key_v.replace('AT', 'AutoDiff')
                key_v = key_v.replace('dLdx_DDN_fnc_B', 'DDN-J')
                key_v = key_v.replace('dLdx_DDN_fnc', 'DDN-E')
                key_v = key_v.replace('dLdx_fnc', 'J')
                key_v = key_v.replace('dLdx_structured_fnc', 'E')
                key_v = key_v.replace('_', '-').replace('+', '-')

                if key_v in ['PI-DDN-E', 'SI-DDN-E', 'PI-IFT-E', 'SI-IFT-E']:
                    key_v = "\\rowcolor{lightgray} "+f"{key_v}"

                row_data = [key_v]
                row_data += [f'{float(v):.04f}' if float(v) >= 0 else '-' for v in value_list]
                writer.writerow(row_data)

    # ==== Draw
    num_data_sizes = len(data_sizes)
    obj_dict = {'Time on CPU (s)': time_cpu, 'Time on GPU (s)': time_gpu, 'Memory on GPU (MB)': memory_gpu}
    fig, ax = plt.subplots(3, 1, figsize=(25, 10))
    count = 0

    for obj_name in obj_dict.keys():
        obj = obj_dict[obj_name]
        df = pd.DataFrame()
        df[''] = data_size_list
        df['method'] = ['forward', 'backward'] * len(data_sizes)

        ax[count].set_title(obj_name, fontdict={'fontsize': 10, 'color': 'red'})
        ax[count].axis('off')
        ax[count].axis('tight')

        for method in obj.keys():
            method_viz = method.replace('dLdx_DDN_fnc', 'ddn')

            if method_viz.find('IFT') > -1:
                method_viz = method_viz.replace('dLdx_fnc', 'auto')
                method_viz = method_viz.replace('dLdx_structured_fnc', 'structured')

            df[method_viz] = obj[method]

        table = ax[count].table(
            cellText=np.vstack([df.columns, df.values]).T,
            cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        cells = table.properties()["celld"]
        for i in range(8):
            for j in range(2 * num_data_sizes + 1):
                if i == 0 or (i == 1 and j == 0):
                    cells[i, j]._text.set_weight('bold')

        fig.canvas.draw()

        for i in range(num_data_sizes):
            mergecells(table, (0,  i * 2 + 2), (0, i * 2 + 1))

        mergecells(table, (1,0), (0,0))
        count += 1

    fig.savefig(f"{g_save_path}_speed_memory.png")
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------

def get_method_viz(v):
    v = v.replace('AT', 'AutoDiff')
    v = v.replace('PI', 'Power Iteration')
    v = v.replace('SI', 'Simultaneous Iteration')

    return v

# ----------------------------------------------------------------------------------------------------------------------

def visual_precision(precision_info, data_sizes, g_save_path, enable_legend=False):
    eigen_gaps = {}
    fp_gaps = {}

    #
    data_size_strings = []

    for data_size in data_sizes:
        batch, m, n = data_size
        data_size_strings.append(f"({batch},{m})")

    #
    for method in precision_info.keys():
        eigen_gaps.update({method: {'cpu': [], 'gpu': []}})
        fp_gaps.update({method: {'cpu': [], 'gpu': []}})

        for data_size in data_sizes:
            batch, m, n = data_size
            data_size_str = f"b{batch}_m{m}_n{n}"
            obj = precision_info[method][data_size_str]

            for device in obj['eigen_gap'].keys():
                eigen_gaps[method][device].append(obj['eigen_gap'][device])
                fp_gaps[method][device].append(obj['fp_gap'][device])

    colors = ['black', 'blue', 'red', 'pink', 'purple', 'brown', 'cyan', 'olive']
    marks = ['^', 'o', 'P', '*', 'h', 'p', '.', 'd']

    for obj_name in ['eigen_gap', 'fp_gap']:
        obj = eigen_gaps if obj_name == 'eigen_gap' else fp_gaps

        for device in ['gpu']:
            _, ax = plt.subplots(1, 1)
            legends = []

            for idx, method in enumerate(obj.keys()):
                data = obj[method][device]
                if data is None: continue
                plt.plot(data, color=colors[idx], marker=marks[idx],
                    markeredgewidth=2.0, markersize=10, linewidth=2)

            legends = [get_method_viz(v.split('+')[0]) for v in obj.keys()]
            ylabel = f'Eigen Distance ({device.upper()})' \
                if obj_name == 'eigen_gap' else f'Fixed-Point Distance ({device.upper()})'

            if enable_legend:
                plt.legend(legends, fontsize=20, loc=2)
                enable_legend = False

            ax.set_xticks(range(len(data_size_strings)))
            ax.set_xticklabels(data_size_strings, fontsize=16)
            ax.spines['right'].set_color((.8,.8,.8))
            ax.spines['top'].set_color((.8,.8,.8))
            ax.grid('on')
            plt.xticks(fontsize=16)
            plt.yticks(rotation=60, fontsize=16)
            plt.xlabel('Data Size (batch,feature)', fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            plt.tight_layout()
            plt.savefig(f"{g_save_path}_{obj_name}_{device}.png")
            plt.close()

    return enable_legend

# ----------------------------------------------------------------------------------------------------------------------

def plot_hor(ax, data, xlabel=None, ylabel=None, methods=None):
    data_indices = np.arange(data.shape[0])
    bar_list = ax.barh(data_indices, data, color='gray')

    # Set y up side down due to barh()
    ax.invert_yaxis()

    # Set color for the same solver
    if methods is not None:
        for i, v in enumerate(methods):
            if v.find('AutoDiff') > -1:
                color = 'gray'
            elif v.find('PI') > -1:
                color = 'bisque'
                if v.find('-E') > -1: color = 'green'
            elif v.find('SI') > -1:
                color = 'pink'
                if v.find('-E') > -1: color = 'green'

            bar_list[i].set_color(color)

    # Set labels
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=10)

    if ylabel is not None:
        ax.set_yticks(data_indices)
        ax.set_yticklabels(ylabel, fontsize=10)
    else:
        ax.spines[['left']].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])

    ax.spines[['right', 'top']].set_visible(False)

# ----------------------------------------------------------------------------------------------------------------------

def draw_bar_chart():
    # Data sizes: 32, 64, 128, 256, 512, 1024
    csv_dir = 'results/20230702-121221/100iters/solve_symmetric_stopFalse_float32/csv'
    data_paths = [
        os.path.join(csv_dir, 'time_gpu.csv'),
        os.path.join(csv_dir, 'memory_gpu.csv')]

    # ====
    for data_path in data_paths:
        # Read data from .csv
        data_matrix = []
        methods = []

        with open(data_path, 'r', newline='') as f:
            data = csv.reader(f)
            next(data)

            for row in data:
                methods.append(row[0].replace('\\rowcolor{lightgray} ', ''))
                data_matrix.append([float(v.replace('-', '-1')) for v in row[1:]])

        # Convert to numpy array and log10
        data_matrix = np.array(data_matrix).astype(np.float32)

        # Set draw sizes
        dis_data_size = [32, 128, 512]
        num_plts = 2 * len(dis_data_size)

        # Draw bar chart
        _, axes = plt.subplots(1, num_plts, figsize=(12, 3))
        plt.tight_layout()
        count = 0

        for idx, data_size in enumerate([32, 64, 128, 256, 512, 1024]):
            if data_size not in dis_data_size: continue

            # Set y labels
            ylabel = methods if count == 0 else None

            # Set fwd and bwd data
            data_fwd = np.log10(data_matrix[:, idx * 2])
            data_bwd = np.log10(data_matrix[:, idx * 2 + 1])

            # Plot
            plot_hor(axes[count], data_fwd, xlabel=fr'5$\times${data_size}, fwd', ylabel=ylabel, methods=methods)
            plot_hor(axes[count + 1], data_bwd, xlabel=fr'5$\times${data_size}, bwd', methods=methods)
            count += 2

        # Save figure
        plt.savefig(f"{data_path.replace('.csv', '.png')}", bbox_inches='tight')

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    draw_bar_chart()