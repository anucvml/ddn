#!/usr/bin/env python
#
# This is a PyTorch autograd version without considering batch dimension
# One can use it for proofread the manual version.
#
# Two base modules:
# - EigenDecomposeBase (in ed_forward.py): for forward solver
#   - EigenAuto (AT)             : autogradient torch.linalg.eigh() method
#   - PowerIteration (PI)        : power iteration method for the largest Eigenvalue
#   - Simultaneous Iteration (SI): simultaneous iteration with QR decomposition for multiple Eigenvalues
#
# - BackpropBase (in ed_backward.py): will be passed to EigenDecomposeBase() and form backward()
#   - AutoBackprop      : Jacobian based, also contains an automatic calculation of DDN components
#   - DDNBackprop       : DDN based
#   - JRMagnusBackprop  : feasible for symmetric matrix (leads to SPD)
#   - FixedPointBackprop: this won't work, too huge errors
#
# - Note:
#   - PI and SI are useful for the largest Eigenvalue of an asymmetric matrix
#
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------------

class EigenFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, solver, backward_fnc_name='dLdx'):
        with torch.no_grad():
            lambds, y = solver.solve(x)
            ctx.intermediate_results = x, lambds, y, solver, backward_fnc_name

        return lambds, y

    @staticmethod
    def backward(ctx, dLdl, dLdy):
        with torch.no_grad():
            x, lambds, y, solver, backward_fnc_name = ctx.intermediate_results
            del ctx.intermediate_results
            dLdx = solver.backward(x, y, lambds, dLdy, backward_fnc_name)

        return dLdx, None, None, None

# ----------------------------------------------------------------------------------------------------------------------

class EigenDecomposeBase(nn.Module):
    """
    Base for different upper algorithms
    """
    def __init__(self, num_iters=0, uniform_solution_method='skip', epsilon=1.0e-8, solver_back=None,
                 num_eigen_values=-1, enable_stop_condition=False, backprop_inverse_mode='solve'):
        super(EigenDecomposeBase, self).__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.uniform_solution_method = uniform_solution_method
        self.solver_back = None
        self.num_eigen_values = num_eigen_values
        self.enable_stop_condition = enable_stop_condition

        if solver_back is not None:
            self.solver_back = solver_back(
                solve_fnc=self.solve, objective_fnc=self.objective_fnc,
                constraint_fnc=self.constraint_fnc, backprop_inverse_mode=backprop_inverse_mode)

    def set_num_eigen_values(self, num_eign_values):
        self.num_eigen_values = num_eign_values

    def get_num_eigen_values(self):
        return self.num_eigen_values

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_epsilon(self, epsilon):
        return self.epsilon

    def set_uniform_solution_method(self, uniform_solution_method):
        self.uniform_solution_method = uniform_solution_method

    def objective_fnc(self, x, u):
        uTx = torch.einsum('bmn,bmk->bnk', u, x)
        f = -torch.diagonal(torch.einsum('bnk,bkm->bnm', uTx, u), dim1=1, dim2=2)

        return f.sum()

    def constraint_fnc(self, u):
        n = u.shape[2]
        uTu = torch.einsum('bmn,bmk->bnk', u, u)
        h = uTu - torch.eye(n, dtype=u.dtype, device=u.device).view(1, n, n)

        return torch.diagonal(h, dim1=1, dim2=2)

    def stop_condition(self, q, q_t, mode='abs'):
        if mode == 'abs':
            diff = (q - q_t).abs().max()
        else:
            diff = (q - q_t).pow(2).max()

        status = diff <= self.epsilon

        return status

    def check_eigen_gap(self, l, u, x):
        """
        :param l: Eigenvalues b*n
        :param u: Eigenvectors b*m*n
        :param x: source matrix b*m*m
        :return:
        """
        lu = torch.einsum('bn,bmn->bmn', l, u)
        xu = torch.einsum('bmn,bnk->bmk', x, u)

        if False:
            eigen_gap = (lu - xu).abs()
            eigen_gap = eigen_gap.max(1)[0].mean(1).mean(0)
        else:
            eigen_gap = torch.sqrt(torch.sum(torch.pow(lu - xu, 2), dim=2))
            eigen_gap = eigen_gap.max(1)[0].mean(0)

        return eigen_gap

    def check_fixed_point_gap(self, u, x, method='max'):
        """
        This is to validate for
          1. Eigen decomposition that if l u = x u
          2. fixed point theory that if
             u_{k+1}=(x u_k)/norm_2(x u_k)=(l u_k)/norm_2(l u_k)

        :param u: Eigenvectors b*m*n
        :param x: source matrix b*m*m
        :return:
        """
        xu = torch.einsum('bmn,bnk->bmk', x, u)
        xu_normed = F.normalize(xu, p=2.0, dim=1)

        if False:
            fixed_point_gap = (u - xu_normed).abs()
            fixed_point_gap = fixed_point_gap.max(1)[0].mean(1).mean(0)
        else:
            fixed_point_gap = torch.sqrt(torch.sum(torch.pow(u - xu_normed, 2), dim=2))
            fixed_point_gap = fixed_point_gap.max(1)[0].mean(0)

        return fixed_point_gap

    def uniform_solution_direction(self, u, u_ref=None):
        batch, m, n = u.shape
        direction_factor = 1.0

        if self.uniform_solution_method != 'skip':
            if u_ref is None:
                u_ref = u.new_ones(1, m, 1).detach()

            direction = torch.einsum('bmk,bmn->bkn', u_ref, u)

            if u_ref.shape[2] == n:
                direction = torch.diagonal(direction, dim1=1, dim2=2).view(batch, 1, n)

            if self.uniform_solution_method == 'positive':
                direction_factor = (direction >= 0).float()
            elif self.uniform_solution_method == 'negative':
                direction_factor = (direction <= 0).float()

        u = u * (direction_factor - 0.5) * 2

        return u

    def solve(self, x):
        raise NotImplementedError

    def backward(self, x, y, lambds, dLdy, backward_fnc_name):
        dLdx = None

        if self.solver_back is not None:
            enable_B = backward_fnc_name.find('_B') > -1
            backward_fnc_name = backward_fnc_name.replace('_B', '')

            assert hasattr(self.solver_back, backward_fnc_name), f'!!!{self.solver_back} has no function {backward_fnc_name}.'
            dLdx_fnc = getattr(self.solver_back, backward_fnc_name)

            # Multiple Eigenvalues together is the same as individual Eigenvalue to save memory
            if backward_fnc_name in ['dLdx_DDN_fnc', 'dLdx_fnc', 'dLdx_structured_fnc']:
                dLdx = 0.0

                for i in range(y.shape[-1]):
                    # Set enable_B as False to avoid store B explicitly due to the special structure of B
                    dLdx += dLdx_fnc(x, y[:, :, i:i + 1], lambds[:, i:i + 1], dLdy[:, :, i:i + 1], enable_B=enable_B)
            else:
                assert False, f"!!!Unknown backward_fnc_name: {backward_fnc_name}."

        return dLdx

    def forward(self, x, backward_fnc_name='dLdx_DDN_fnc'):
        if backward_fnc_name == 'unroll':
            l, u = self.solve(x)
        else:
            l, u = EigenFunction.apply(x, self, backward_fnc_name)

        return l, u

# ----------------------------------------------------------------------------------------------------------------------

class EigenAuto(EigenDecomposeBase):
    """
    Supports autogradient for gradient check with the DDN version
    """
    def __init__(self, enable_symmetric=False, **kwargs):
        super(EigenAuto, self).__init__(**kwargs)
        self.enable_symmetric = enable_symmetric

    def solve(self, x):
        m = x.shape[1]

        if True:  # self.enable_symmetric:
            l, u = torch.linalg.eigh(x)
        else:
            l, u = torch.linalg.eig(x)

        u = self.uniform_solution_direction(u)

        # The greatest Eigenvalue first, the default is the greatest last
        l = torch.flip(l, dims=[1])
        u = torch.flip(u, dims=[2])

        if (self.num_eigen_values <= 0)  or (self.num_eigen_values > m):
            self.num_eigen_values = m

        return l[:, :self.num_eigen_values], u[:, :, :self.num_eigen_values]

# ----------------------------------------------------------------------------------------------------------------------

class PowerIteration(EigenDecomposeBase):
    """
    Supports the calculation of the greatest (only) Eigenvalue and the corresponding Eigenvector
    """
    def __init__(self, **kwargs):
        super(PowerIteration, self).__init__(**kwargs)

    def solve(self, x):
        b, m = x.shape[:2]
        n = 1
        u = torch.randn(b, m, n, device=x.device, dtype=x.dtype)
        if self.enable_stop_condition: u_his = u.detach()

        for i in range(self.num_iters):
            u = F.normalize(torch.einsum('bkm,bmn->bkn', x, u), p=2.0, dim=1)
            if self.enable_stop_condition:
                cur_u = u.detach()
                if self.stop_condition(cur_u, u_his): break
                u_his = cur_u

        u = self.uniform_solution_direction(u)
        xu = torch.einsum('bkm,bmn->bkn', x, u)
        l = torch.einsum('bmn,bmk->bnk', u, xu) / torch.einsum('bmn,bmk->bnk', u, u)
        l = torch.diagonal(l, dim1=1, dim2=2)

        if (self.num_eigen_values <= 0)  or (self.num_eigen_values > m):
            self.num_eigen_values = m

        return l[:, :self.num_eigen_values], u[:, :, :self.num_eigen_values]

# ----------------------------------------------------------------------------------------------------------------------

class SimultaneousIteration(EigenDecomposeBase):
    """
    simultaneous iteration with QR decomposition supporting multiple Eigenvalues
    """
    def __init__(self, **kwargs):
        super(SimultaneousIteration, self).__init__(**kwargs)

    def solve(self, x):
        batch, m = x.shape[:2]
        if (self.num_eigen_values <= 0) or (self.num_eigen_values > m): self.num_eigen_values = m
        Q, R = torch.linalg.qr(x)
        if self.enable_stop_condition: Q_his = Q[:, :, :self.num_eigen_values].detach()

        for _ in range(self.num_iters):
            x_t = torch.einsum('bmn,bnk->bmk', x, Q)
            Q, R = torch.linalg.qr(x_t)

            if self.enable_stop_condition:
                cur_Q = Q[:, :, :self.num_eigen_values].detach()
                if self.stop_condition(cur_Q, Q_his): break
                Q_his = cur_Q

        l, sort_indices = torch.sort(torch.diagonal(R, dim1=1, dim2=2), dim=1, descending=True)
        l = l[:, :self.num_eigen_values]
        Q = Q[
            torch.arange(batch).view(batch, 1, 1), torch.arange(m).view(1, m, 1),
            sort_indices.view(batch, 1, m)[:, :, :self.num_eigen_values]]
        Q = self.uniform_solution_direction(Q)

        return l, Q