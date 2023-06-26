#!/usr/bin/env python
#
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
from torch.autograd.functional import jacobian as J

# ----------------------------------------------------------------------------------------------------------------------

class BackpropBase(nn.Module):
    """
    Base function for implicit differentiation
    """
    def __init__(
            self, solve_fnc=None, objective_fnc=None, constraint_fnc=None, epsilon=1.0e-8,
            enable_lambd_dhhdyy=False, backprop_instance='DDNBackprop', backprop_inverse_mode='solve',
            enable_symmetric=True):
        super(BackpropBase, self).__init__()
        self.solve_fnc = solve_fnc
        self.objective_fnc = objective_fnc
        self.constraint_fnc = constraint_fnc
        self.epsilon = epsilon
        self.enable_lambd_dhhdyy = enable_lambd_dhhdyy
        self.backprop_instance = backprop_instance
        self.backprop_inverse_mode = backprop_inverse_mode
        self.enable_symmetric = enable_symmetric
        assert backprop_inverse_mode in ['solve', 'pinverse']

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_epsilon(self):
        return self.epsilon

    def dfdy_fnc(self, x, y):
        raise NotImplementedError

    def dhdy_fnc(self, x, y):
        raise NotImplementedError

    def dhdx_fnc(self, x, y):
        raise NotImplementedError

    def dffdxy_fnc(self, x, y):
        raise NotImplementedError

    def dffdyy_fnc(self, x, y):
        raise NotImplementedError

    def dhhdyy_fnc(self, x, y, lambds=None):
        raise NotImplementedError

    def dhhdxy_fnc(self, x, y):
        raise NotImplementedError

    def H_fnc(self, x, y, lambds=None):
        batch, m, n = y.shape

        if self.backprop_instance == 'DDNBackprop':
            I = torch.eye(m, dtype=x.dtype, device=x.device).view(1, m, m)
            lambd_reg = 2 * lambds.view(batch, n, 1, 1) + self.epsilon

            # Merge dffdyy, lambda * dhhdyy, and epsilon
            if n == 1:
                # lambda is changed to -eigenvalues, so use -
                H = -(x + x.permute(0, 2, 1)) - lambd_reg.view(batch, 1, 1) * I
            else:
                H = x.new_zeros(batch, m, n, m, n)

                for i in range(n):
                    # lambda is changed to -eigenvalues, so use -
                    H[:, :, i, :, i] += -(x + x.permute(0, 2, 1)) - lambd_reg[:, i] * I

            H = H.view(batch, m * n, m * n)
        else:  # this requires more memory
            dffdyy = self.dffdyy_fnc(x, y)
            dffdyy = dffdyy.view(batch, m * n, m * n)

            if self.enable_lambd_dhhdyy:
                lambd_dhhdyy = self.dhhdyy_fnc(x, y, lambds=lambds)
                lambd_dhhdyy = lambd_dhhdyy.view(batch, m * n, m * n)
            else:
                dhhdyy = self.dhhdyy_fnc(x, y)
                dhhdyy = dhhdyy.view(batch, n, m * n, m * n)
                lambd_dhhdyy = torch.einsum('bn,bnmk->bmk', lambds, dhhdyy)

            H = dffdyy - lambd_dhhdyy  # lambda is changed to -eigenvalues, so use -
            H += self.epsilon * torch.eye(m * n, dtype=x.dtype, device=x.device).view(1, m * n, m * n)

        return H

    def _lambda_fnc(self, A, x, y):
        # Note this would be -eigenvalues due to the -lambda*h in the Lagrangian form in DDNs
        # while the commonly-used one is +lambda*h, both are correct due to the equality constraint.
        # Just need to pay attention that when calculating B and H, change the sign of eigenvalues
        # for lambda.
        batch, m, n = y.shape
        lambds = torch.linalg.solve(
            torch.einsum('bnm,bkm->bnk', A, A),
            torch.einsum('bnm,bm->bn', A, self.dfdy_fnc(x, y).view(batch, m * n)))

        return lambds

    def get_components(self, x, y, lambds, enable_B=False):
        batch, m, n = y.shape
        dhdy = self.dhdy_fnc(x, y)
        dhdx = self.dhdx_fnc(x, y)
        dhdy = dhdy.view(batch, n, m * n) if (dhdy is not None) else None
        dhdx = dhdx.view(batch, n, m * m) if (dhdx is not None) else None

        A = dhdy
        C = dhdx if (dhdx is not None) else None
        H = self.H_fnc(x, y, lambds)

        if enable_B:
            dffdxy = self.dffdxy_fnc(x, y)
            B = dffdxy.view(batch, m * n, m * m)
            dhhdxy = self.dhhdxy_fnc(x, y)

            if dhhdxy is not None:
                dhhdxy = dhhdxy.view(batch, n, m * n, m * m)
                B -= torch.einsum('bn,bnmk->bmk', lambds, dhhdxy)
        else:
            B = None

        return A, B, C, H

    def dLdx_DDN_fnc(self, x, y, lambds, dLdy, enable_B=True):
        """
        :param x:
        :param y:
        :param lambds:
        :param dLdy:
        :return:
        Very important to use epsilon * eye for all matrices to be inversed by linalg.solve()
        """
        batch, m, n = y.shape
        D = dLdy.view(batch, -1, m * n)
        lambds = -lambds  # due to different signs of DDN lambda and eigenvalues, also see self._lambda_fnc(*)
        A, B, C, H = self.get_components(x, y, lambds, enable_B=enable_B)
        # I_epsilon = torch.eye(n, dtype=x.dtype, device=x.device).view(1, n, n) * self.epsilon

        if self.backprop_inverse_mode == 'pinverse':
            # As found that torch.linalg.solve is as bad as torch.pinverse for float32, so use torch.linalg.pinv so far
            Hinv = torch.linalg.pinv(H)
            DHinv = torch.einsum('bmn,bnk->bmk', D, Hinv)
            DHinvAT = torch.einsum('bmn,bkn->bmk', DHinv, A)
            AHinvAT = torch.einsum('bmn,bkn->bmk', torch.einsum('bmn,bnk->bmk', A, Hinv), A)
            AHinvAT_inv = torch.linalg.pinv(AHinvAT)  # + I_epsilon)
            P = DHinvAT_AHinvAT_inv = torch.einsum('bmn,bnk->bmk', DHinvAT, AHinvAT_inv)
            PAHinv = torch.einsum('bmn,bnk->bmk', torch.einsum('bmn,bnk->bmk', P, A), Hinv)
            PAHinv_DHinv = PAHinv - DHinv

            if enable_B:
                dLdx = torch.einsum('bmn,bnk->bmk', PAHinv_DHinv, B)
            else:
                dLdx_comp = torch.einsum('bmn,bkm->bnk', PAHinv_DHinv, -y)
                dLdx = dLdx_comp + dLdx_comp.permute(0, 2, 1)

            if C is not None: dLdx -= torch.einsum('bmn,bnk->bmk', P, C)
        elif self.backprop_inverse_mode == 'solve':
            DHinv = torch.linalg.solve(H, D.permute(0, 2, 1)).permute(0, 2, 1)
            DHinvAT = torch.einsum('bmn,bkn->bmk', DHinv, A)
            HinvAT = torch.linalg.solve(H, A.permute(0, 2, 1))
            AHinvAT = torch.einsum('bmn,bnk->bmk', A, HinvAT)
            # AHinvAT += I_epsilon
            DHinvAT_AHinvAT_inv = torch.linalg.solve(AHinvAT, DHinvAT.permute(0, 2, 1)).permute(0, 2, 1)
            P = DHinvAT_AHinvAT_inv
            PA = torch.einsum('bmn,bnk->bmk', P, A)
            PAHinv = torch.linalg.solve(H, PA.permute(0, 2, 1)).permute(0, 2, 1)
            PAHinv_DHinv = PAHinv - DHinv

            # To avoid B which takes batch*m*(m*m) memory. Note that PAHinv is a vector
            if enable_B:
                dLdx = torch.einsum('bmn,bnk->bmk', PAHinv_DHinv, B)
            else:
                dLdx_comp = torch.einsum('bmn,bkm->bnk', PAHinv_DHinv, -y)
                dLdx = dLdx_comp + dLdx_comp.permute(0, 2, 1)

            if C is not None: dLdx -= torch.einsum('bmn,bnk->bmk', P, C)
        else:
            HinvAT = torch.linalg.solve(H, torch.einsum('bmn->bnm', A))
            HinvB = torch.linalg.solve(H, B)
            AHinvAT = torch.einsum('bmn,bnk->bmk', A, HinvAT)  # + I_epsilon
            AHinvB = torch.einsum('bmn,bnk->bmk', A, HinvB)
            if C is not None: AHinvB -= C
            AHinvAT_inv_AHinvB = torch.linalg.solve(AHinvAT, AHinvB)
            dydx = torch.einsum('bmn,bnk->bmk', HinvAT, AHinvAT_inv_AHinvB) - HinvB
            dLdx = torch.einsum('bmn,bnk->bmk', D, dydx)

        return dLdx.view(batch, m, m)

# ----------------------------------------------------------------------------------------------------------------------

class AutoBackprop(BackpropBase):
    """
    Autogradient based gradient method, to generate ground truth components to be compared with the DDN version
    """
    def __init__(self, **kwargs):
        super(AutoBackprop, self).__init__(backprop_instance='AutoBackprop', **kwargs)

    def dfdy_fnc(self, x, y):
        dfdy = grad(self.objective_fnc(x, y), (y), create_graph=True)[0]

        return dfdy

    def dhdy_fnc(self, x, y):
        batch, m, n = y.shape
        dhdy = []

        for i in range(batch):
            dhdy.append(J(lambda y_per: self.constraint_fnc(y_per), (y[i:i+1]), create_graph=True))

        return torch.cat(dhdy, dim=0)

    def dhdx_fnc(self, x, y):
        return None

    def dffdxy_fnc(self, x, y):
        batch, m, n = y.shape
        dffdxy = []

        for i in range(batch):
            dffdxy.append(J(lambda x_per: self.dfdy_fnc(x_per, y[i:i+1]), (x[i:i+1])))

        return torch.cat(dffdxy, dim=0)

    def dffdyy_fnc(self, x, y):
        batch, m, n = y.shape
        dffdyy = []

        for i in range(batch):
            dffdyy.append(J(lambda y_per: self.dfdy_fnc(x[i:i+1], y_per), (y[i:i+1])))

        return torch.cat(dffdyy, dim=0)

    def dhhdyy_fnc(self, x, y, lambds=None):
        batch, m, n = y.shape
        dhhdyy = []

        for i in range(batch):
            dhhdyy.append(J(lambda y_per: self.dhdy_fnc(x, y_per), (y[i:i+1])))

        return torch.cat(dhhdyy, dim=0)

    def dhhdxy_fnc(self, x, y):
        return None

    def dydx_fnc(self, x, y):
        batch, m, n = y.shape
        dydx = []

        for i in range(batch):
            dydx.append(J(lambda x: self.solve_fnc(x)[1], (x[i:i+1])))

        return torch.cat(dydx, dim=0)

    def dLdx_auto_iter_fnc(self, x, y, lambds, dLdy, enable_B=None):
        batch, m, n = y.shape
        dLdy = dLdy.view(batch, -1, m * n)
        dydx = self.dydx_fnc(x, y).view(batch, m * n, m * m)
        dLdx = torch.einsum('bmn,bnk->bmk', dLdy, dydx)
        dLdx = dLdx.view(batch, -1, m, m)

        return dLdx

# ----------------------------------------------------------------------------------------------------------------------

class FixedPointBackprop(BackpropBase):
    def __init__(self, **kwargs):
        super(FixedPointBackprop, self).__init__(**kwargs)
        self.enable_global_identity_matrix = False

        if self.enable_global_identity_matrix:
            self.m = None
            self.identity_matrix = self.generate_identity_matrix(self.m)

    def generate_identity_matrix(self, m, dtype=torch.float32, device='cpu'):
        if m is None:
            return None
        else:
            return torch.eye(m, dtype=dtype, device=device).view(1, m, m)

    def fixed_point_fnc(self, x, y):
        xy = torch.einsum('bmn,bnk->bmk', x, y)
        xy_normed = F.normalize(xy, p=2.0, dim=1)
        f = y - xy_normed

        return f

    def get_dLdx(self, batch, m, n, B, H, dLdy):
        Q = dLdy.view(batch, 1, m * n)

        if self.backprop_inverse_mode == 'solve':
            HHT = torch.einsum('bmn,bkn->bmk', H, H)
            HQT = torch.einsum('bmn,bkn->bmk', H, Q)
            QHinv = torch.linalg.solve(HHT, HQT).permute(0, 2, 1)
        else:
            QHinv = torch.einsum('bmn,bnk->bmk', Q, torch.linalg.pinv(H))

        dLdx = -torch.einsum('bmn,bnk->bmk', QHinv, B).view(batch, m, m)
        if self.enable_symmetric: dLdx = 0.5 * (dLdx + dLdx.permute(0, 2, 1))

        return dLdx

    def dLdx_fnc(self, x, y, lambds, dLdy, enable_B=False):
        """
        The fixed point function is f(x, y)=y_{k+1}-(x y_k)/(x y_k)=0 due to y_{k+1}=y_k;
        therefore, dydx=-(dfdy)^{-1} dfdx -> dLdx=dLdy*dydx

        :param x: source matrix b*m*m
        :param y: Eigenvectors b*m*n
        :param, dLdy: gradient from loss to Eigenvectors b*m*n
        :return: dLdx
        """
        batch, m, n = y.shape
        dfdx, dfdy = [], []

        for i in range(batch):
            dfdx_per, dfdy_per = J(lambda x, y: self.fixed_point_fnc(x, y), (x[i:i+1], y[i:i+1]))
            dfdx.append(dfdx_per)
            dfdy.append(dfdy_per)

        B = dfdx = torch.stack(dfdx, 0).view(batch, m, m * m)
        H = dfdy = torch.stack(dfdy, 0).view(batch, m, m * n)

        dLdx = self.get_dLdx(batch, m, n, B, H, dLdy)

        return dLdx

    # Different from dLdx_auto_fnc(), this uses exploited matrix structure.
    def dLdx_structured_fnc_per_eigen(self, x, y, lambds, dLdy, enable_B=False):
        batch, m, n = y.shape
        device = x.device
        dLdy = dLdy.view(batch, m)

        if self.enable_global_identity_matrix:  # !!!do not use this to do memory test
            if m != self.m:
                self.m = m
                self.identity_matrix = self.generate_identity_matrix(
                    m, dtype=x.dtype, device=device)

            if device != self.identity_matrix.device:
                self.identity_matrix = self.identity_matrix.to(device)

            I = self.identity_matrix.view(1, m, m)
        else:
            I = self.generate_identity_matrix(m, dtype=x.dtype, device=device)

        yyT = torch.einsum('bmn,bkn->bmk', y, y)
        I_minus_yyT_div_lambda = (I - yyT) / lambds.view(batch, 1, 1)

        #
        if False:  # use Ay
            xy = torch.einsum('bmn,bnk->bmk', x, y)
            xy_norm = xy.pow(2).sum(1, keepdim=True).sqrt()
            xy_T_x = torch.einsum('bmn,bmk->bnk', xy, x)
            xy_xy_T_x = torch.einsum('bmn,bnk->bmk', xy, xy_T_x)
            dfdy = self.identity_matrix - (x / xy_norm - xy_xy_T_x / xy_norm.pow(3))
            H = dfdy = dfdy.view(batch, m, m)
        else:  # use lambda y
            H = dfdy = I - torch.einsum('bmn,bnk->bmk', I_minus_yyT_div_lambda, x)

        if enable_B:  # requires B with lots of memory
            # 1. Use Ay
            # xy_xy_T = torch.einsum('bmn,bkn->bmk', xy, xy).view(batch, m * m, 1)
            # I_minus_xy_xy_T = (I.view(1, m * m, 1) / xy_norm - xy_xy_T / xy_norm.pow(3))
            # B = dfdx = -torch.einsum('bmn,bkn->bmk', I_minus_xy_xy_T, y).view(batch, m, m * m)

            # 2. Use lambda y (since it equals Ay)
            B = - torch.einsum('bmn,bkn->bmk', I_minus_yyT_div_lambda, y)
            B = B.view(batch, m, m * m)

            dLdx = self.get_dLdx(batch, m, n, B, H, dLdy)
        else:  # without B to save memory
            if self.backprop_inverse_mode == 'pinverse':
                H_inv = torch.linalg.pinv(H)
                K = torch.einsum('bm,bmn->bn', dLdy, H_inv)
            else:
                K = torch.linalg.solve(H, dLdy).view(batch, m)

            dLdx = -torch.einsum('bm,bn->bmn', torch.einsum('bm,bmn->bn', K, -I_minus_yyT_div_lambda), y.view(batch, m))
            if self.enable_symmetric: dLdx = 0.5 * (dLdx + dLdx.permute(0, 2, 1))

        return dLdx

    def dLdx_structured_fnc(self, x, y, lambds, dLdy, enable_B=False):
        n = y.shape[-1]
        dLdx = 0.0

        for i in range(n):  # TODO: this would slow down
            dLdx += self.dLdx_structured_fnc_per_eigen(
                x, y[:, :, i:i+1], lambds[:, i], dLdy[:, :, i:i+1], enable_B=enable_B)

        return dLdx

# ----------------------------------------------------------------------------------------------------------------------

# From the book "Matrix Differential Calculus with Applications in Statistics and Econometrics"
class JRMagnusBackprop(BackpropBase):
    def __init__(self, **kwargs):
        super(JRMagnusBackprop, self).__init__(**kwargs)

    def dLdx_fnc(self, x, y, lambd, dLdy, enable_B=False):
        """
        m                    : number of features
        n                    : number of Eigenvalue(s)
        x square matrix      : batch * m * m
        y n Eigenvector(s)   : batch * m * n
        dLdy                 : batch * m * n
        lambd n Eigenvalue(s): batch * n
        """
        batch, m, n = y.shape
        I_matrix = torch.eye(m, dtype=x.dtype, device=x.device).view(1, m, m)
        dy = []

        for i in range(n):
            cur_y = y[:, :, i]
            cur_lambd = lambd[:, i].view(batch, 1, 1)
            D = cur_lambd * I_matrix - x
            lambdI_sub_x_inv = torch.linalg.pinv(D)  # torch.pinverse has huge errors

            # Matrix outer product via wiki, different from vectorize matrix for vector outer product
            cur_dy = torch.kron(cur_y, lambdI_sub_x_inv)

            dy.append(cur_dy)

        dy = torch.stack(dy, dim=3)
        dLdx = torch.einsum('bmn,bmkn->bk', dLdy, dy).view(batch, m, m)

        if self.enable_symmetric:  # this is strange but the same as eigh for symmetric matrix
            dLdx = 0.5 * (dLdx + dLdx.permute(0, 2, 1))

        return dLdx

# ----------------------------------------------------------------------------------------------------------------------

class DDNBackprop(BackpropBase):
    """
    DDN based backpropagation
    """
    def __init__(self, enable_lambd_dhhdyy=True, **kwargs):
        super(DDNBackprop, self).__init__(
            enable_lambd_dhhdyy=enable_lambd_dhhdyy, **kwargs)

    def dfdy_fnc(self, x, y):
        batch, m, n = y.shape
        dfdy = y.new_zeros(batch, m, n)
        xt_plus_x = x + x.permute(0, 2, 1)

        for i in range(n):
            dfdy[:, :, i] = -torch.einsum('bnm,bm->bn', xt_plus_x, y[:, :, i])

        return dfdy

    def dhdy_fnc(self, x, y):
        batch, m, n = y.shape
        dhdy = x.new_zeros(batch, n, m, n)

        for i in range(n):
            dhdy[:, i,:, i] = 2 * y[:, :, i]

        return dhdy

    def dhdx_fnc(self, x, y):
        return None

    def dffdyy_fnc(self, x, y):
        batch, m, n = y.shape
        dffdyy = x.new_zeros(batch, m, n, m, n)
        xt_plus_x = x + x.permute(0, 2, 1)

        for i in range(n):
            dffdyy[:, :, i, :, i] = -xt_plus_x

        return dffdyy

    def dffdxy_fnc(self, x, y):
        batch, m, n = y.shape
        dffdxy = x.new_zeros(batch, m, n, m, m)

        for i in range(n):
            for j in range(m):
                dffdxy[:, j, i, :, j] -= y[:, :, i]
                dffdxy[:, j, i, j, :] -= y[:, :, i]

        return dffdxy

    def dhhdyy_fnc(self, x, y, lambds=None):
        batch, m, n = y.shape

        if lambds is None:
            dhhdyy = x.new_zeros(batch, n, m, n, m, n)
            I = torch.eye(m, dtype=x.dtype, device=x.device).view(1, m, m)

            for i in range(n):
                dhhdyy[:, i, :, i, :, i] = 2 * I
        else:
            if False:  # this is over m not easy to understand
                dhhdyy = x.new_zeros(batch, m * n, m * n)
                I = torch.eye(n, dtype=x.dtype, device=x.device).view(1, n, n)
                lambd_I = 2 * torch.einsum('bn,bnk->bnk', lambds, I)

                for i in range(m):
                    in_v = i * n
                    inp1_v = in_v + n
                    dhhdyy[:, in_v : inp1_v, in_v : inp1_v] = lambd_I
            else:  # this is similar to dffdyy, easy to merge with dffdyy
                dhhdyy = x.new_zeros(batch, m, n, m, n)
                I = torch.eye(m, dtype=x.dtype, device=x.device).view(1, m, m)

                for i in range(n):
                    dhhdyy[:, :, i, :, i] = 2 * lambds[:, i].view(batch, 1, 1) * I

            dhhdyy = dhhdyy.view(batch, m, n, m, n)

        return dhhdyy

    def dhhdxy_fnc(self, x, y):
        return None