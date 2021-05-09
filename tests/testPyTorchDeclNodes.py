# TEST SAMPLE DEEP DECLARATIVE NODES
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
import numpy as np

import sys
sys.path.append("../")
from ddn.pytorch.node import *
from ddn.pytorch.sample_nodes import *

def test_node(node, xs):
	y, ctx = torch.no_grad()(node.solve)(*xs)
	y.requires_grad = True
	fxy = node.objective(*xs, y=y)

	# print("Input:\n{}".format(xs[0].detach()))
	# print("Output:\n{}".format(y.detach()))
	# print("Fn Value:\n{}".format(fxy.detach()))

	Dys = super(type(node), node).gradient(*xs, y=None, v=None, ctx=None) # call parent gradient method
	Dys_analytic = node.gradient(*xs, y=None, v=None, ctx=None)
	# print("Dy:\n{}".format(Dys[0].detach()))
	# print("Dy analytic:\n{}".format(Dys_analytic[0].detach()))
	print("Autograd and analytic gradients agree?", torch.allclose(Dys[0], Dys_analytic[0], rtol=0.0, atol=1e-12))

	node.gradient = super(type(node), node).gradient # Use generic gradient for gradcheck
	DL = DeclarativeLayer(node)
	y = DL(*xs)
	Dy = grad(y, xs[0], grad_outputs=torch.ones_like(y))[0]
	# print("Output:   {}".format(y.detach()))
	# print("Dy:\n{}".format(Dy.detach()))
	test = gradcheck(DL, xs, eps=1e-6, atol=1e-5, rtol=1e-5, raise_exception=False)
	print("gradcheck passed:", test)

def scalar_to_batched_tensor(a, batch_size, dtype=None, device=None, requires_grad=False):
	a = torch.tensor([a], dtype=dtype, device=device, requires_grad=requires_grad).unsqueeze(0).expand(batch_size, -1) # bx1
	if requires_grad:
		return a.clone() # If you want the gradient of a, the copies of a cannot share memory
	else:
		return a

# Polynomial
print("\nPolynomial Example:\n")
node = UnconstPolynomial()
x = torch.tensor([[0.25], [1.0]], dtype=torch.double, requires_grad=True)
xs = (x,)
test_node(node, xs)
# print("DYf:", 4*x*(y**3)+6*(x**2)*(y**2)-24*y) # DYf
# print("DYYf:", 12*x*(y**2)+12*(x**2)*y-24) # DYYf
# print("DXYf:", 4*(y**3)+12*x*(y**2)) # DXYf

# PseudoHuberPool
print("\nPseudoHuber Pooling Example:\n")
node = GlobalPseudoHuberPool2d(chunk_size=None)
b = 2
m = 7
x = torch.randn(b, m, m, dtype=torch.double, requires_grad=True)
alpha = 0.5
alpha = scalar_to_batched_tensor(alpha, b, dtype=torch.double, requires_grad=False)
# alpha = scalar_to_batched_tensor(alpha, b, dtype=torch.double, requires_grad=True)
xs = (x, alpha)
test_node(node, xs)

# LinFcnOnUnitCircle:
print("\nLinFcnOnUnitCircle:\n")
node = LinFcnOnUnitCircle()
x = torch.randn(3, 1, dtype=torch.double, requires_grad=True)
xs = (x,)
test_node(node, xs)

# ConstLinFcnOnParameterizedCircle:
print("\nConstLinFcnOnParameterizedCircle:\n")
node = ConstLinFcnOnParameterizedCircle()
x = torch.randn(3, 1, dtype=torch.double, requires_grad=True)
xs = (x,)
test_node(node, xs)

# LinFcnOnParameterizedCircle:
print("\nLinFcnOnParameterizedCircle:\n")
node = LinFcnOnParameterizedCircle()
x = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
xs = (x,)
test_node(node, xs)

# QuadFcnOnSphere:
print("\nQuadFcnOnSphere:\n")
node = QuadFcnOnSphere()
x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
xs = (x,)
test_node(node, xs)

# QuadFcnOnBall:
print("\nQuadFcnOnBall:\n")
node = QuadFcnOnBall()
x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
# Force at least one xi to be ||xi|| < 1.0
x[1, :] = x[1, :] / torch.sqrt(torch.einsum('bm,bm->b', (x, x))[1]) / 1.1
xs = (x,)
test_node(node, xs)
