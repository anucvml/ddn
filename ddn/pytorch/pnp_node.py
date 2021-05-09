# PnP Node
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>

import torch
import cv2
import numpy as np
from ddn.pytorch.node import *
import ddn.pytorch.geometry_utilities as geo

class PnP(AbstractDeclarativeNode):
    """"""
    def __init__(self,
        eps=1e-8,
        gamma=None,
        objective_type='reproj',
        alpha=1.0,
        chunk_size=None,
        ransac_max_num_iterations=1000,
        ransac_threshold=0.1
        ):
        super().__init__(eps=eps, gamma=gamma, chunk_size=chunk_size)
        self.objective_type = objective_type
        self.alpha = alpha
        self.ransac_max_num_iterations = ransac_max_num_iterations
        self.ransac_threshold = ransac_threshold
        
    def objective(self, p2d, p3d, w, K, y):
        """Weighted PnP error

        Arguments:
            p2d: (b, n, 2) Torch tensor,
                batch of 2D point-sets

            p3d: (b, n, 3) Torch tensor,
                batch of 3D point-sets,

            w: (b, n) Torch tensor,
                batch of weight vectors

            K: (b, 4) Torch tensor or None,
                batch of camera intrinsic parameters (fx, fy, cx, cy),
                set to None if points are already K-normalised

            y: (b, 6) Torch tensor,
                batch of transformation parameters
                format:
                    y[:, 0:3]: angle-axis rotation vector
                    y[:, 3:6]: translation vector

        Return Values:
            objective value: (b, ) Torch tensor
        """
        if self.objective_type is 'cosine':
            return self.objective_cosine(p2d, p3d, w, K, y)
        elif self.objective_type is 'reproj':
            return self.objective_reproj(p2d, p3d, w, K, y)
        elif self.objective_type is 'reproj_huber':
            return self.objective_reproj_huber(p2d, p3d, w, K, y)

    def objective_cosine(self, p2d, p3d, w, K, theta):
        """Weighted cosine distance error
        f(p2d, p3d, w, y) = sum_{i=1}^n
            w_i * (1 - p2d_i^T N(R(y) p3d_i + t(y)))
            where N(p) = p / ||p||
        """
        p2d_bearings = geo.points_to_bearings(p2d, K)
        p3d_transform = geo.transform_and_normalise_points_by_theta(p3d, theta)
        return torch.einsum('bn,bn->b', (w, 1.0 - torch.einsum('bnd,bnd->bn',
            (p2d_bearings, p3d_transform))))

    def objective_reproj(self, p2d, p3d, w, K, theta):
        """Weighted squared reprojection error
        f(p2d, p3d, w, K, y) = sum_{i=1}^n
            w_i * ||pi(p3d_i, K, y) - p2d_i||_2^2
            where pi(p, K, y) = h2i(K * (R(y) * p + t(y)))
            where h2i(x) = [x1 / x3, x2 / x3]
        """
        p3d_projected = geo.project_points_by_theta(p3d, theta, K)
        z2 = torch.sum((p3d_projected - p2d) ** 2, dim=-1)
        return torch.einsum('bn,bn->b', (w, z2))

    def objective_reproj_huber(self, p2d, p3d, w, K, theta):
        """Weighted Huber reprojection error
        f(p2d, p3d, w, K, y) = sum_{i=1}^n
            w_i * rho(pi(p3d_i, K, y) - p2d_i, alpha)
            where rho(z, alpha) = / 0.5 z^2 for |z| <= alpha
                                  \ alpha * (|z| - 0.5 * alpha) else
            and pi(p, K, y) = h2i(K * (R(y) * p + t(y)))
            where h2i(x) = [x1 / x3, x2 / x3]
        """
        def huber(z2, alpha=1.0):
            return torch.where(z2 <= alpha ** 2, 0.5 * z2, alpha * (
                z2.sqrt() - 0.5 * alpha))
        p3d_projected = geo.project_points_by_theta(p3d, theta, K)
        z2 = torch.sum((p3d_projected - p2d) ** 2, dim=-1)
        return torch.einsum('bn,bn->b', (w, huber(z2, self.alpha)))

    def solve(self, p2d, p3d, w, K=None):
        p2d = p2d.detach()
        p3d = p3d.detach()
        w = w.detach()
        K = K.detach() if K is not None else None
        theta = self._initialise_theta(p2d, p3d, w, K).requires_grad_()
        theta = self._run_optimisation(p2d, p3d, w, K, y=theta)
        # # Alternatively, disentangle batch element optimisation:
        # for i in range(p2d.size(0)):
        #     Ki = K[i:(i+1),...] if K is not None else None
        #     theta[i, :] = self._run_optimisation(p2d[i:(i+1),...],
        #         p3d[i:(i+1),...], w[i:(i+1),...], Ki, y=theta[i:(i+1),...])
        return theta.detach(), None

    def _initialise_theta(self, p2d, p3d, w, K):
        return self._ransac_p3p(p2d, p3d, K,
            self.ransac_max_num_iterations, self.ransac_threshold)

    def _ransac_p3p(self, p2d, p3d, K, max_num_iterations,
        reprojection_error_threshold):
        theta = p2d.new_zeros(p2d.size(0), 6)
        p2d_np = p2d.cpu().numpy()
        p3d_np = p3d.cpu().numpy()
        if K is None:
            K_np = np.float32(np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        dist_coeff_np = None
        for i in range(p2d_np.shape[0]): # loop over batch
            if K is not None:
                K_np = np.float32(np.array(
                    [[K[i, 0], 0.0, K[i, 2]],
                    [0.0, K[i, 1], K[i, 3]],
                    [0.0, 0.0, 1.0]]))
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                p3d_np[i, :, :], p2d_np[i, :, :], K_np, dist_coeff_np,
                iterationsCount=max_num_iterations,
                reprojectionError=reprojection_error_threshold,
                flags=cv2.SOLVEPNP_EPNP)
            # print(inliers.shape[0], '/',  p2d_np.shape[1])
            # Optionally, refine estimate with LM:
            # rvec, tvec = cv2.solvePnPRefineLM(
            #     p3d_np[i, :, :], p2d_np[i, :, :], K_np, dist_coeff_np,
            #     rvec, tvec)
            if rvec is not None and tvec is not None and retval:
                rvec = torch.as_tensor(
                    rvec, dtype=p2d.dtype, device=p2d.device).squeeze(-1)
                tvec = torch.as_tensor(
                    tvec, dtype=p2d.dtype, device=p2d.device).squeeze(-1)
                if torch.isfinite(rvec).all() and torch.isfinite(tvec).all():
                    theta[i, :3] = rvec
                    theta[i, 3:] = tvec
        return theta

    def _run_optimisation(self, *xs, y):
        with torch.enable_grad():
            opt = torch.optim.LBFGS([y],
                                    lr=1.0,
                                    max_iter=1000,
                                    max_eval=None,
                                    tolerance_grad=1e-40,
                                    tolerance_change=1e-40,
                                    history_size=100,
                                    line_search_fn="strong_wolfe"
                                    )
            def reevaluate():
                opt.zero_grad()
                f = self.objective(*xs, y=y).sum() # sum over batch elements
                f.backward()
                return f
            opt.step(reevaluate)
        return y