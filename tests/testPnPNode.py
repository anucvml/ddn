# TEST PNP DEEP DECLARATIVE NODES
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>
#
# When running from the command-line make sure that the "ddn" package has been added to the PYTHONPATH:
#   $ export PYTHONPATH=${PYTHONPATH}: ../ddn
#   $ python testPyTorchDeclNodes.py

import torch
from torch.autograd import grad
from torch.autograd import gradcheck

import sys
sys.path.append("../")
from ddn.pytorch.pnp_node import *

# Generate data
b = 2
n = 10
theta = torch.randn(b, 6, dtype=torch.double)
R = geo.angle_axis_to_rotation_matrix(theta[..., :3])
t = theta[..., 3:]

# Generate image first, uniformly at random, then assign depths
xy = 2.0 * torch.rand(b, n, 2, dtype=torch.double) - 1.0 # [-1, 1]
z = 2.0 * torch.rand(b, n, 1, dtype=torch.double) + 1.0 # [1, 3]
p3d_transformed = torch.cat((z * xy, z), dim=-1)
p3d = torch.einsum('brs,bms->bmr', (R.transpose(-2,-1), p3d_transformed - t.unsqueeze(-2)))
p2d = xy.clone()
p2d = p2d + 0.04 * torch.randn(b, n, 2, dtype=torch.double) # add noise
# p2d[:, 0:1, :] = torch.randn(b, 1, 2, dtype=torch.double) # add outliers

# Plot:
# import matplotlib.pyplot as plt
# p2d_np = p2d.cpu().numpy()
# p3d_proj_np = geo.project_points_by_theta(p3d, theta).cpu().numpy()
# plt.scatter(p2d_np[0, :, 0], p2d_np[0, :, 1], s=10, c='k', alpha=1.0, marker='s')
# plt.scatter(p3d_proj_np[0, :, 0], p3d_proj_np[0, :, 1], s=10, c='r', alpha=1.0, marker='o')
# plt.show()

w = torch.ones(b, n, dtype=torch.double) # bxn
w = w.abs() # Weights must be positive and sum to 1 per batch element
w = w.div(w.sum(-1).unsqueeze(-1))

# Create a PnP problem and create a declarative layer:
# node = PnP(objective_type='cosine')
node = PnP(objective_type='reproj', chunk_size=None)
# node = PnP(objective_type='reproj_huber', alpha=0.1)
DL = DeclarativeLayer(node)

p2d = p2d.requires_grad_()
p3d = p3d.requires_grad_()
w = w.requires_grad_()
K = None

# DL, p2d, p3d, w, K = DL.cuda(0), p2d.cuda(0), p3d.cuda(0), w.cuda(0), K.cuda(0) if K is not None else None # Move everything to GPU

# Run forward pass:
y = DL(p2d, p3d, w, K)

# Compute objective function value:
f = node.objective(p2d, p3d, w, K, y=y)

# Compute gradient:
Dy = grad(y, (p2d, p3d, w), grad_outputs=torch.ones_like(y))

# print("Input p2d:\n{}".format(p2d.detach().cpu().numpy()))
# print("Input p3d:\n{}".format(p3d.detach().cpu().numpy()))
# print("Input w:\n{}".format(w.detach().cpu().numpy()))
# print("Input K:\n{}".format(K))
print("Theta Ground-Truth:\n{}".format(theta.detach().cpu().numpy()))
print("Theta Estimated:\n{}".format(y.detach().cpu().numpy()))
print("Objective Function Value:\n{}".format(f.detach().cpu().numpy()))
# print("Dy:\n{}\n{}\n{}".format(Dy[0].detach().cpu().numpy(), Dy[1].detach().cpu().numpy(), Dy[2].detach().cpu().numpy()))

# Run gradcheck:
# DL, p2d, p3d, w, K = DL.cpu(), p2d.cpu(), p3d.cpu(), w.cpu(), K.cpu() if K is not None else None # Move everything to CPU
test = gradcheck(DL, (p2d, p3d, w, K), eps=1e-4, atol=1e-4, rtol=1e-4, raise_exception=True)
print("gradcheck passed:", test)