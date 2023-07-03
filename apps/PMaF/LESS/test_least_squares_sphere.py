#!/usr/bin/env python
#
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import os, torch
import numpy as np
# import matplotlib as mpl
# mpl.use('tkagg')
import matplotlib.pyplot as plt
import scipy.optimize as sciop
from datetime import datetime

# ----------------------------------------------------------------------------------------------------------------------

# The following visualization code is modified based on tutorials/01_simple_worked_example.ipynb
def solve_quadratic(x):
    """Analytical solution to min. f s.t. h = 0. Returns both optimal primal and dual variables."""
    return x / np.sqrt(np.dot(x, x)), None

# ----------------------------------------------------------------------------------------------------------------------

# b: center (target), y: optimal solution for circle, x: optimal solution for ellipse
def viz_problem_quadratic(b, y, ax, A, x, x_projected=None, enable_viz_proj=False):
    b = b.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    A = A.detach().cpu().numpy()
    A_inv = np.linalg.pinv(A)
    x = x.reshape(x.shape[0], 1).detach().cpu().numpy()
    axes_cat = np.concatenate(
        (np.cos(np.linspace(0.0, 2.0 * np.pi, num=1000)), np.sin(np.linspace(0.0, 2.0 * np.pi, num=1000))),
        axis=0).reshape(2, 1000)
    axes_v = lambda r: np.matmul(A_inv, r * axes_cat + b)

    # Draw central circle (constraint)
    ax.plot(
        np.cos(np.linspace(0.0, 2.0 * np.pi, num=1000)),
        np.sin(np.linspace(0.0, 2.0 * np.pi, num=1000)),
        '-', color='red', linewidth=1)

    # Draw samples
    for r in [0.125, 0.5, 1.125, 2.0, 3.125]:
        axes_v_x_y = axes_v(r)
        ax.plot(axes_v_x_y[0], axes_v_x_y[1], '--', color='blue', linewidth=1)

    # Draw center connection
    if not enable_viz_proj:
        c_detransform = np.matmul(A_inv, b).flatten()
        ax.plot([0, c_detransform[0]], [0, c_detransform[1]], '-.', color='black')  # recover the center

    # Draw ellipse for optimal solution, num=1000 aligns curve and solution
    r_optimal_in = np.matmul(A, x) - b
    r_optimal = np.sqrt(np.matmul(r_optimal_in.transpose(1, 0), r_optimal_in))
    axes_v_x_y = axes_v(r_optimal)
    ax.plot(axes_v_x_y[0], axes_v_x_y[1], '-.', color='green', linewidth=1)

    ax.plot(y[0], y[1], 'o', color='red', markersize=6, linewidth=1)
    if x_projected is not None: ax.plot(x_projected[:, 0], x_projected[:, 1], '-', color='green', linewidth=1)
    if x is not None: ax.plot(x[0], x[1], '*', color='black', markersize=9.5, linewidth=1)

# ----------------------------------------------------------------------------------------------------------------------

def objective_lagrange(A, x, b, lr=1.0, enable_debug=False):
    opt = torch.optim.LBFGS(
        [x], lr=lr, max_iter=200, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09,
        history_size=1000, line_search_fn=None)

    def opt_iter():
        opt.zero_grad()

        dGdY = 2 * x.t()  # A is dGdu
        DfDY = torch.matmul(A.t(), torch.matmul(A, x) - b).t()
        lambda_v = torch.matmul(torch.matmul(1 / torch.matmul(dGdY, dGdY.t()), dGdY), DfDY.t())
        f_hat_constraint = torch.matmul(lambda_v, (torch.matmul(x.t(), x) - 1))
        f_hat_org = 0.5 * torch.matmul((torch.matmul(A, x) - b).t(), torch.matmul(A, x) - b)
        loss = f_hat_org + f_hat_constraint
        loss.backward()
        if enable_debug: print(lambda_v.item(), loss.item())

        return loss

    opt.step(opt_iter)

# ----------------------------------------------------------------------------------------------------------------------

def object_func(A, x, b):
    m, n = A.shape
    x = np.reshape(x, (n, 1))  # otherwise it will be (x.shape[0],)
    if len(b.shape) == 1: b = np.reshape(b, (m, 1))
    common = np.matmul(A, x) - b

    return 0.5 * np.matmul(np.transpose(common, (1, 0)), common).sum()

# ----------------------------------------------------------------------------------------------------------------------

def constraint_func(x):
    x = np.reshape(x, (x.shape[0], 1))

    return np.matmul(np.transpose(x, (1, 0)), x).sum()

# ----------------------------------------------------------------------------------------------------------------------

def objective_scipy(A, x, b, disable_constraint=False, constraint_magnitude=1.0, enable_ball_constraint=False):
    if disable_constraint:
        constraints = ()
    else:
        if enable_ball_constraint:
            constraints = sciop.NonlinearConstraint(constraint_func, 0, constraint_magnitude ** 2)
        else:
            constraints = sciop.NonlinearConstraint(
                constraint_func, constraint_magnitude ** 2, constraint_magnitude ** 2)

    result = sciop.minimize(
        lambda x: object_func(A, x, b), np.reshape(x, (x.shape[0],)), method='SLSQP', constraints=constraints)
    x = result.x

    if disable_constraint:
        x_final = x
    elif enable_ball_constraint:
        x_final = x
        invalid_idx = np.sqrt(np.sum(np.power(x, 2))) > constraint_magnitude
        x_final[invalid_idx] = constraint_magnitude * x[invalid_idx] / np.sqrt(np.sum(np.power(x[invalid_idx], 2)))
    else:
        x_final = constraint_magnitude * x / np.sqrt(np.sum(np.power(x, 2)))

    loss = object_func(A, x_final, b)

    return x_final, loss

# ----------------------------------------------------------------------------------------------------------------------

def proj_solution_sphere(x, constraint_magnitude=1.0):
    x = constraint_magnitude * x / torch.sqrt(torch.matmul(x.t(), x))

    return x

# ----------------------------------------------------------------------------------------------------------------------

def cal_riemannian_grad(x, grad_fn):
    grads = grad_fn(x)
    x_unit = x / torch.norm(x, 2, dim=0, keepdim=True)
    I_xx_trans = torch.eye(x.shape[0], device=x.device, dtype=x.dtype) - torch.matmul(x_unit, x_unit.permute(1, 0))
    grads = torch.matmul(I_xx_trans, grads)

    return grads

# ----------------------------------------------------------------------------------------------------------------------

def iter_projection(A, x, b, num_iters=10, tolerance=1.0e-7, method='proj'):
    x_projected = [x]
    loss = lambda x: 0.5 * torch.matmul((torch.matmul(A, x) - b).t(), torch.matmul(A, x) - b)
    x_grad = lambda x: torch.matmul(A.t(), (torch.matmul(A, x) - b))
    obj_center_dist = torch.matmul(torch.pinverse(A), b).pow(2).sum().sqrt()
    count = 0
    max_backtrack_ls_count = 0

    for t in range(num_iters):
        count += 1
        x_history = x

        if method.find('rieman') > -1:
            eta_numerator_in = torch.matmul(A.t(), (torch.matmul(A, x) - b))
            x_unit = x / torch.norm(x, 2, dim=0, keepdim=True)
            I_xx_trans = torch.eye(x.shape[0], device=x.device, dtype=x.dtype) \
                - torch.matmul(x_unit, x_unit.permute(1, 0))
            eta_numerator = torch.matmul(torch.matmul(eta_numerator_in.t(), I_xx_trans), eta_numerator_in)
            eta_denominator_in = torch.matmul(torch.matmul(A, I_xx_trans), eta_numerator_in)
            eta_denominator = torch.matmul(eta_denominator_in.t(), eta_denominator_in)
            eta_denominator += 1.0e-8
            eta = eta_numerator / eta_denominator
            grads = cal_riemannian_grad(x, x_grad)
        else:
            eta_numerator_in = torch.matmul(A.t(), (torch.matmul(A, x) - b))
            eta_numerator = torch.matmul(eta_numerator_in.t(), eta_numerator_in)
            eta_denominator_in = torch.matmul(A, eta_numerator_in)
            eta_denominator = torch.matmul(eta_denominator_in.t(), eta_denominator_in)
            eta_denominator += 1.0e-8
            eta = eta_numerator / eta_denominator
            grads = x_grad(x)

        descend_dir = -grads
        if method.find('tandecay') > -1: y_grad_his = grads

        if method.find('dirweight') > -1:
            x_dir = -x_grad(x) if (obj_center_dist >= 1) else x_grad(x)
            lr_weight = 1 - torch.cosine_similarity(x, x_dir, dim=0)
        else:
            lr_weight = 1

        if method.find('backtrack') > -1:
            enable_proj_loop_solution = False
            eta = 1. if (method.find('rieman') > -1 and method.find('etaone') > -1) else eta
            backtrack_ls_count = 0
            alpha = 0.5
            beta = 0.8
            loss_x = loss(x)
            x_update = x + lr_weight * eta * descend_dir
            if enable_proj_loop_solution: x_update = proj_solution_sphere(x_update)

            while loss(x_update) > loss_x + alpha * lr_weight * eta * torch.matmul(grads.t(), descend_dir):
                eta = beta * eta
                x_update = x + lr_weight * eta * descend_dir
                if enable_proj_loop_solution: x_update = proj_solution_sphere(x_update)
                backtrack_ls_count += 1

            max_backtrack_ls_count = backtrack_ls_count if (backtrack_ls_count > max_backtrack_ls_count) \
                else max_backtrack_ls_count
        elif method.find('tandecay') > -1:
            enable_proj_loop_solution = True
            beta = 0.9
            backtrack_ls_count = 0

            x_update = x + lr_weight * eta * descend_dir
            if enable_proj_loop_solution: x_update = proj_solution_sphere(x_update)
            y_grad_update = cal_riemannian_grad(x_update, x_grad)
            dir_divergence = torch.matmul(y_grad_his.t(), y_grad_update)

            while dir_divergence < 0:  # when the direction is divergent
                if dir_divergence.abs() <= 1.0e-5:
                    break  # in case of 90 degrees that 1.0e-5 due to numerical issue

                eta = beta * eta
                x_update = x + lr_weight * eta * descend_dir
                if enable_proj_loop_solution: x_update = proj_solution_sphere(x_update)
                y_grad_update = cal_riemannian_grad(x_update, x_grad)
                dir_divergence = torch.matmul(y_grad_his.t(), y_grad_update)
                backtrack_ls_count += 1

            max_backtrack_ls_count = backtrack_ls_count \
                if (backtrack_ls_count > max_backtrack_ls_count) else max_backtrack_ls_count

        x = x + lr_weight * eta * descend_dir
        x_projected.append(x)
        x = proj_solution_sphere(x)
        x_projected.append(x)

        if method.find('tandecay') > -1 and dir_divergence.abs() <= 1.0e-5: break
        if (loss(x) - loss(x_history)).abs() < tolerance: break

    if x_projected[-1].shape[0] == 2:
        x_projected = torch.stack(x_projected, dim=0).view(-1, 2).detach().cpu().numpy()
    else:
        x_projected = None

    return x, x_projected, count, loss(x), max_backtrack_ls_count

# ----------------------------------------------------------------------------------------------------------------------

def draw_fnc(
        method, viz_dict, sub_count, count_row, count_col, b, y, A, u_final, enable_viz_proj,
        loss, count=None, x_projected=None):
    # Last row shows x-axis
    enable_show_xaxis = True if np.floor(sub_count / count_col) == count_row - 1 else False
    enable_show_yaxis = True if sub_count % count_col == 0 else False

    plt.subplot(count_row, count_col, sub_count + 1)
    a = plt.gca()
    viz_problem_quadratic(
        b, y, a, A, u_final, enable_viz_proj=enable_viz_proj, x_projected=x_projected)
    a.axis('square')
    a.set_xlim(-1.9, 1.9)
    a.set_ylim(-1.9, 1.9)
    plt.subplots_adjust(wspace=0.1)

    if enable_show_yaxis:
        plt.yticks(rotation=60, fontsize=14)
    else:
        plt.yticks([])

    if enable_show_xaxis:
        plt.subplots_adjust(hspace=0.15)
        plt.xticks(fontsize=14)
        hpad = -37
    else:
        plt.subplots_adjust(hspace=0)
        plt.xticks([])
        hpad = -20

    info_string = ''
    if count is not None: info_string += f'Iterations: {count:d}'
    info_string += f'\nFPD: {np.sqrt(2 * loss):.4f}'

    method_viz = method
    for key in viz_dict.keys(): method_viz = method_viz.replace(key, viz_dict[key])
    method_viz = method_viz.replace('-', '+')
    plt.title(f'({chr(97 + sub_count)}) {method_viz}', y=0, pad=hpad, fontsize=14)
    plt.text(1.88, -1.88, info_string, ha='right', va='bottom', fontsize=14)

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    viz_dict = {
        'scipy': 'SciPy',
        'proj': 'PGD',
        'dirweight': 'DW',
        'rieman': 'RM',
        'tandecay': 'TWD',
        'backtrack-etaone': 'BLSOne',
        'backtrack': 'BLS'
        }

    # ==== Edit those parameters
    enable_rand_data = True
    enable_viz_proj = True  # True: draw the figures in paper, False: statistics for the tables in paper
    time_string = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join('results', time_string)
    os.makedirs(save_dir, exist_ok=True)

    if enable_viz_proj:
        methods = [
            'scipy',
            'proj',
            'proj-dirweight',
            'proj-rieman',
            'proj-rieman-backtrack-etaone',
            'proj-rieman-tandecay'
            ]
    else:
        methods = [
            'scipy',
            'proj',
            'proj-dirweight',
            'proj-rieman',
            'proj-rieman-backtrack-etaone',
            'proj-rieman-backtrack-etaone-dirweight',
            'proj-rieman-backtrack',
            'proj-rieman-backtrack-dirweight',
            'proj-rieman-tandecay',
            'proj-rieman-tandecay-dirweight'
            ]

    # Set the number of samples
    device = 'cuda'
    loss_dict = {'dist': []}

    #
    num_iters = 100  # max optimization iteration, suppose reach num_iters it fails

    # Then copy fail list to data_idx_list, set enable_plot=True
    if enable_rand_data:
        if enable_viz_proj:
            enable_plot = True
            data_idx_list = [19, 38, 41, 66, 70, 72, 92, 170]
        else:
            data_idx_list = range(1000)
            enable_plot = False
    else:
        data_idx_list = [0]
        enable_plot = True

    num_rand_seeds = len(data_idx_list)

    for disp_idx, rand_idx in enumerate(data_idx_list):
        if ((disp_idx % 10 == 0) or (disp_idx == num_rand_seeds - 1)): print(disp_idx)

        torch.manual_seed(rand_idx)
        torch.cuda.manual_seed(rand_idx)
        np.random.seed(rand_idx)

        # ==== Objective function min_u = 0.5 (wu-b)^2, s.t. |x|^2=1
        if enable_rand_data:
            m, n = 2, 2  # 2, 2; 64, 32; 1024, 256
            A = torch.randn(m, n, dtype=torch.float32).to(device)  # A: (m, n), x: (n, 1)
            b = torch.randn(m, 1, dtype=torch.float32).to(device)  # b: (m, 1)
        else:
            m, n = 2, 2
            A = torch.tensor([[0.569525, -1.254572], [0.414020, 0.124439]], dtype=torch.float32).to(device)
            b = torch.tensor([[-1.583332], [-0.286124]], dtype=torch.float32).to(device)

        if (m != 2 or n != 2): enable_plot = False

        # ==== Solve for circle
        A_inv = torch.pinverse(A)
        center = torch.matmul(A_inv, b).reshape(n)

        y = torch.einsum('nm,mk->nk', torch.pinverse(A), b).view(-1, 1)
        y = proj_solution_sphere(y)

        # ==== Check objective function center is inside or outside or on the unit circle
        obj_center_dist = center.pow(2).sum().sqrt()
        loss_dict['dist'].append(obj_center_dist.item())

        # ==== Solve for ellipse
        # Initialization
        x = y

        # Run on different methods:
        # A: matrix weights; x: input matrix; b: shifted center
        if enable_plot:
            sub_count, count_total = 0, len(methods)
            count_row = 2 if (count_total == 4) and enable_viz_proj else 1
            count_col = int(np.ceil(count_total / count_row))

            if count_row != 1:
                plt.figure(figsize=(8, 8))
            else:
                if len(methods) == 3:
                    plt.figure(figsize=(9, 2.5))
                else:
                    plt.figure(figsize=(20, 5))

        # ====
        if 'scipy' in methods:
            if 'scipy' not in loss_dict.keys(): loss_dict.update({'scipy': {'loss': []}})
            u_hat, loss = objective_scipy(A.cpu().numpy(), x.detach().cpu().numpy(), b.cpu().numpy())

            if True:  # to compare with proj losses which are calculated on PyTorch, the above loss is on numpy
                Ax_b_torch = torch.matmul(A.view(m, n), torch.from_numpy(u_hat).float().to(A.device).view(n, 1)) - b.view(m, 1)
                loss = 0.5 * torch.matmul(Ax_b_torch.t(), Ax_b_torch).mean().item()

            u_final = u_hat
            loss_dict['scipy']['loss'].append(loss)

            if len(data_idx_list) <= 10:
                print(f'SciPy solution: {u_final.flatten()}, loss: {loss:.8f}.')

            if enable_plot:
                u_final = torch.from_numpy(u_final)
                draw_fnc('SciPy', viz_dict, sub_count, count_row, count_col, b, y, A, u_final, enable_viz_proj, loss)
                sub_count += 1

        # ====
        for method in methods:
            if method.find('proj') > -1:
                u_final, x_projected, count, loss, max_backtrack_ls_count = \
                    iter_projection(A, x, b, num_iters=num_iters, tolerance=1.0e-7, method=method)
                u_final = u_final.flatten().detach().cpu()
                loss = loss.item()

                # ====
                if method not in loss_dict.keys(): loss_dict.update({method: {'loss': [], 'count': []}})
                loss_dict[method]['loss'].append(loss)
                loss_dict[method]['count'].append(count)

                if len(data_idx_list) <= 10:
                    print(f'Projection {method}, solution: {u_final.flatten()} for {count} iterations, loss: {loss:.8f}.')

                if enable_plot:
                    draw_fnc(
                        method, viz_dict, sub_count, count_row, count_col, b, y, A, u_final, enable_viz_proj,
                        loss, count=count, x_projected=x_projected)
                    sub_count += 1

        if enable_plot:
            if len(data_idx_list) > 1:
                os.makedirs(f'{save_dir}/case_study', exist_ok=True)
                save_path = f'{save_dir}/case_study/case_study_{rand_idx}.png'
            else:
                save_path = f'{save_dir}/case_study.png'

            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            # plt.show()

    # ====
    num_dist_in, num_dist_out = len([v for v in loss_dict['dist'] if v < 1]), len([v for v in loss_dict['dist'] if v > 1])
    save_txt_path = f'{save_dir}/sample{num_rand_seeds}-m{m}-n{n}'
    save_txt_path += f'_{device}.txt'
    num_digits = 6

    with open(save_txt_path, 'w') as f:
        for method in loss_dict.keys():
            if method not in ['scipy', 'dist']:
                failed_indices = [idx for idx, v in enumerate(loss_dict[method]['count']) if v >= num_iters]
                failed_losses = [loss_dict[method]['loss'][v] for v in failed_indices]
                failed_dist = [loss_dict['dist'][v] for v in failed_indices]
                num_dist_in_failed = len([v for v in failed_dist if v < 1])
                num_dist_out_failed = len([v for v in failed_dist if v > 1])

                if 'scipy' in methods:
                    scipy_losses = [loss_dict['scipy']['loss'][v] for v in failed_indices]

                    if False:
                        gap_epsilon = 1.0e-7
                        better_list, mean_diff_list = [], []

                        for v1, v2 in zip(loss_dict[method]['loss'], loss_dict['scipy']['loss']):
                            if abs(v1 - v2) >= gap_epsilon:
                                mean_diff_list.append((v1 - v2) / v2)
                                better_list.append(1) if (v1 < v2) else None
                            else:
                                mean_diff_list.append(0)
                    else:
                        better_list, mean_diff_list = [], []

                        for v1, v2 in zip(loss_dict[method]['loss'], loss_dict['scipy']['loss']):
                            mean_diff_list.append((v1 - v2) / v2)
                            num_decimal = num_digits - np.floor(np.log10(v2) + 1)
                            gap_epsilon = np.power(10, -num_decimal)

                            if abs(v1 - v2) >= gap_epsilon:
                                better_list.append(1) if (v1 < v2) else None
                            else:
                                better_list.append(1)

                    mean_diff = np.mean(mean_diff_list)
                else:
                    scipy_losses, better_list, mean_diff = None, None, None

                f.writelines(f'====>{method}, m: {m}, n: {n}\n')
                f.writelines(f'failed in: {num_dist_in_failed}/{num_dist_in}, out: {num_dist_out_failed}/{num_dist_out}\n')
                if 'scipy' in methods: f.writelines(f'bett: {len(better_list)}, mean diff: {mean_diff * 100}%\n')

                if len(failed_dist) == 0:
                    f.writelines('!!! all passed\n')
                    f.writelines(f"dist: {loss_dict['dist']}\n")
                    f.writelines(f"cout: {loss_dict[method]['count']}\n")
                    if 'scipy' in methods: f.writelines(f"scip: {loss_dict['scipy']['loss']}\n")
                    f.writelines(f"used: {loss_dict[method]['loss']}\n")
                else:
                    f.writelines(f'indx: {failed_indices}\n')
                    f.writelines(f'dist: {failed_dist}\n')
                    if 'scipy' in methods: f.writelines(f'scip: {scipy_losses}\n')
                    f.writelines(f'used: {failed_losses}\n')

        f.close()