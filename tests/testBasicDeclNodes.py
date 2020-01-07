#!/usr/bin/env python
#
# TEST FOR DEEP DECLARATIVE NODES
# Stephen Gould <stephen.gould@anu.edu.au>
#
# When running from the command-line make sure that the "ddn" package has been added to the PYTHONPATH:
#   $ export PYTHONPATH=${PYTHONPATH}:../../ddn
#   $ python testBasicDeclNodes.py
#

from ddn.basic.node import *
from ddn.basic.composition import *
from ddn.basic.sample_nodes import *
from ddn.basic.robust_nodes import *

import numpy as np
import unittest

def computeValueAndGradients(node, x_indx=0, x_input=None, x_min=-2.0, x_max=2.0, x_num=51):
    """
    Utility function for computing value and gradients for a declarative node.

    :param node: The deep declarative node subclass of AbstractDeclarativeNode
    :param x_indx: Index of the input dimension to vary over (0-based).
    :param x_input: Input vector for setting all other input values.
    :param x_min: Minimum value for x[x_indx].
    :param x_max: Maximum value for x[x_indx].
    :param x_num: Number of values in the range.
    :return: x, y, Dy_analytic, Dy_implicit
    """

    assert x_indx < node.dim_x

    # generate test data by varying one dimension of the input
    x = np.linspace(x_min, x_max, num=x_num)
    if x_input is None:
        x_input = np.zeros((node.dim_x,))
    assert len(x_input) == node.dim_x, "mismatch between input size given and expected"
    y, Dy_analytic, Dy_implicit = [], [], []
    for xi in x:
        x_input[x_indx] = xi
        yi, ctx = node.solve(x_input)
        y.append(yi)
        Dy_analytic.append(node.gradient(x_input, yi, ctx))
        Dy_implicit.append(super(type(node), node).gradient(x_input, yi, ctx))

    return x, y, Dy_analytic, Dy_implicit


class TestBasicDeclNodes(unittest.TestCase):

    def testUnconstPolynomial(self):
        x, y, Da, Di = computeValueAndGradients(UnconstPolynomial(), x_min=0.5, x_max=2.5)
        self.assertLess(np.max(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)

    def testLinFcnOnUnitCircle(self):
        x, y, Da, Di = computeValueAndGradients(LinFcnOnUnitCircle())
        self.assertLess(np.max(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)

    def testConstLinFcnOnParameterizedCircle(self):
        x, y, Da, Di = computeValueAndGradients(ConstLinFcnOnParameterizedCircle())
        self.assertLess(np.nanmax(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)

    def testLinFcnOnParameterizedCircle(self):
        x, y, Da, Di = computeValueAndGradients(LinFcnOnParameterizedCircle(), x_indx=0, x_input=np.ones((2,)))
        self.assertLess(np.nanmax(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)
        x, y, Da, Di = computeValueAndGradients(LinFcnOnParameterizedCircle(), x_indx=1, x_input=np.ones((2,)))
        self.assertLess(np.nanmax(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)

    def testQuadFcnOnSphere(self):
        x, y, Da, Di = computeValueAndGradients(QuadFcnOnSphere(), x_indx=0, x_input=0.5 * np.ones((2,)))
        self.assertLess(np.max(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)

    def testQuadFcnOnBall(self):
        x, y, Da, Di = computeValueAndGradients(QuadFcnOnBall(), x_indx=0, x_input=0.5 * np.ones((2,)))
        self.assertLess(np.max(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)

    def testCosineDistance(self):
        x, y, Da, Di = computeValueAndGradients(CosineDistance(), x_indx=0, x_input=np.ones((2,)), x_min=0.5, x_max=2.5)
        self.assertLess(np.max(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)

    def testRobustAverage(self):
        x, y, Da, Di = computeValueAndGradients(RobustAverage(5), x_indx=0, x_input=np.random.rand(5))
        self.assertLess(np.max(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)
        x, y, Da, Di = computeValueAndGradients(RobustAverage(5, alpha=0.5), x_indx=0, x_input=np.random.rand(5))
        self.assertLess(np.max(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)
        x, y, Da, Di = computeValueAndGradients(RobustAverage(5, 'pseudo-huber'), x_indx=0, x_input=np.random.rand(5))
        self.assertLess(np.max(np.abs(np.array(Da) - np.array(Di))), 1.0e-12)

    def testComposedNode(self):
        # TODO
        pass

    def testParallelNode(self):
        # TODO
        pass

    def testSelectNode(self):
        n = 10
        x = np.random.randn(n)
        y, _ = SelectNode(n, 5).solve(x)
        np.testing.assert_array_almost_equal(x[5:], y)
        y, _ = SelectNode(n, 5, 7).solve(x)
        np.testing.assert_array_almost_equal(x[5:8], y)


if __name__ == '__main__':
    unittest.main()
