#!/usr/bin/env python
#
# VISUALIZATION OF GRADIENTS FOR DEEP DECLARATIVE NODES
# Stephen Gould <stephen.gould@anu.edu.au>
#
# When running from the command-line make sure that the "ddn" package has been added to the PYTHONPATH:
#   $ export PYTHONPATH=${PYTHONPATH}:../../ddn
#   $ python vizBasicDeclNodes.py
#

import numpy as np

from ddn.basic.node import *
from ddn.basic.sample_nodes import *
from ddn.basic.robust_nodes import *

from testBasicDeclNodes import computeValueAndGradients

import matplotlib.pyplot as plt
plt.rc('font', family='serif')

# --- visualization and test routine ----------------------------------------------------------------------------------

def vizAndTestProblem(node, title, x_indx=0, x_input=None, x_min=-2.0, x_max=2.0, x_num=51):
    """
    Utility function for checking and visualizing operation of a declarative node.

    :param node: The deep declarative node subclass of AbstractDeclarativeNode
    :param title: The name of the node.
    :param x_indx: Index of the input dimension to vary over (0-based).
    :param x_input: Input vector for setting all other input values.
    :param x_min: Minimum value for x[x_indx].
    :param x_max: Maximum value for x[x_indx].
    :param x_num: Number of values in the range.
    :return: None
    """

    x, y, Dy_analytic, Dy_implicit = computeValueAndGradients(node, x_indx, x_input, x_min, x_max, x_num)

    # compare implicit and true gradients
    print("-" * 80)
    print(title)

    assert np.array(Dy_implicit).shape == np.array(Dy_analytic).shape
    err = np.abs(np.array(Dy_implicit) - np.array(Dy_analytic))
    indx = ~np.isnan(err)
    if indx.any():
        print("Max. Difference: {:0.3e}".format(np.max(err[indx])))
    else:
        print("Max. Difference: NaN")
    print("Number of NaNs: {}".format(np.sum(np.isnan(err))))

    # plot function and gradients
    plt.figure()
    plt.subplot(3, 1, 1)
    for i in range(node.dim_y):
        plt.plot(x, [yi[i] for yi in y])
    plt.ylabel(r"$y$")
    plt.title(title)
    plt.legend([r"$y_{}$".format(i + 1) for i in range(node.dim_y)])

    plt.subplot(3, 1, 2)
    for i in range(node.dim_y):
        if node.dim_x == 1:
            plt.plot(x, [di[i] for di in Dy_analytic])
        else:
            plt.plot(x, [di[i][x_indx] for di in Dy_analytic])
    plt.ylabel(r"$D^{true}y$")

    plt.subplot(3, 1, 3)
    for i in range(node.dim_y):
        if node.dim_x == 1:
            plt.plot(x, [di[i] for di in Dy_implicit])
        else:
            plt.plot(x, [di[i][x_indx] for di in Dy_implicit])
    plt.ylabel(r"$D^{implicit}y$")
    if node.dim_x == 1:
        plt.xlabel(r"$x$")
    else:
        plt.xlabel(r"$x_{}$ (with $x$ = {})".format(x_indx + 1, [x_input[i] if i != x_indx else '?' for i in range(node.dim_x)]))


# --- main -----------------------------------------------------------------------------------------------------------
# test a few problems

if __name__ == '__main__':
    vizAndTestProblem(UnconstPolynomial(), r"minimize $xy^4 + 2x^2y^3 - 12y^2$", x_min=0.5, x_max=2.5)
    vizAndTestProblem(LinFcnOnUnitCircle(), r"minimize $(1, x)^Ty$ subject to $\|y\|^2 = 1$")
    vizAndTestProblem(ConstLinFcnOnParameterizedCircle(), r"minimize $1^Ty$ subject to $\|y\|^2 = x^2$")
    vizAndTestProblem(LinFcnOnParameterizedCircle(), r"minimize $(1, x_1)^Ty$ subject to $\|y\|^2 = x_2^2$", x_indx=0, x_input=np.ones((2,)))
    vizAndTestProblem(LinFcnOnParameterizedCircle(), r"minimize $(1, x_1)^Ty$ subject to $\|y\|^2 = x_2^2$", x_indx=1, x_input=np.ones((2,)))
    vizAndTestProblem(QuadFcnOnSphere(), r"minimize $1/2 y^Ty - x^Ty$ subject to $\|y\|^2 = 1$", x_indx=0, x_input=0.5 * np.ones((2,)))
    vizAndTestProblem(QuadFcnOnBall(), r"minimize $1/2 y^Ty - x^Ty$ subject to $\|y\|^2 \leq 1$", x_indx=0, x_input=0.5 * np.ones((2,)))
    vizAndTestProblem(CosineDistance(), r"minimize $x^T y / \|y\|$", x_indx=0, x_input=np.ones((2,)), x_min=0.5, x_max=2.5)
    vizAndTestProblem(RobustAverage(5), r"minimize $\sum_i \phi^{huber}(y - x_i; 1)$", x_indx=0, x_input=np.random.rand(5))
    vizAndTestProblem(RobustAverage(5, alpha=0.5), r"minimize $\sum_i \phi^{huber}(y - x_i; 0.5)$", x_indx=0, x_input=np.random.rand(5))
    vizAndTestProblem(RobustAverage(5, 'pseudo-huber'), r"minimize $\sum_i \phi^{pseudo\_huber}(y - x_i; 1)$", x_indx=0, x_input=np.random.rand(5))
    plt.show()
