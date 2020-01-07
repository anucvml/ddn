# COMPOSITION OF DEEP DECLARATIVE NODES
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
#

from ddn.basic.node import *

class ComposedNode(AbstractNode):
    """Composes two deep declarative nodes f and g to produce y = g(f(x)). The resulting composition
    behaves exactly like a single deep declarative node, that is, it has the same interface and
    as such can be further composed with other nodes to form a chain."""

    def __init__(self, nodeA, nodeB):
        assert (nodeA.dim_y == nodeB.dim_x)
        super().__init__(nodeA.dim_x, nodeB.dim_y)
        self.nodeA = nodeA
        self.nodeB = nodeB

    def solve(self, x):
        """Overrides the solve method to first compute z = nodeA.solve(x) and then y = nodeB.solve(z). Returns
        the dual variables from both nodes."""
        z, ctxA = self.nodeA.solve(x)
        y, ctxB = self.nodeB.solve(z)
        return y, {'ctxA': ctxA, 'ctxB': ctxB, 'z': z}

    def gradient(self, x, y=None, ctx=None):
        """Overrides the gradient method to compute the composed gradient by the chain rule."""

        # we need to resolve for z since there is currently no way to store this
        if ctx is None:
            z, _ = self.nodeA.solve(x)
        else:
            z = ctx['z']
        Dz = self.nodeA.gradient(x, z).reshape(self.nodeA.dim_y, self.nodeA.dim_x)
        Dy = self.nodeB.gradient(z, y).reshape(self.nodeB.dim_y, self.nodeB.dim_x)
        return np.dot(Dy, Dz)


class ParallelNode(AbstractNode):
    """Combines the output from two nodes f and g running in parallel to produce y(x) = (f(x), g(x)). Each
    node can itself be made of component nodes (i.e., `ComposedNode` or `ParallelNode`)."""

    def __init__(self, nodeA, nodeB):
        assert nodeA.dim_x == nodeB.dim_x
        super().__init__(nodeA.dim_x, nodeA.dim_y + nodeB.dim_y)
        self.nodeA = nodeA
        self.nodeB = nodeB

    def solve(self, x):
        """Overrides the solve method to concatenate the output from both nodes."""
        yA, ctxA = self.nodeA.solve(x)
        yB, ctxB = self.nodeB.solve(x)
        return np.vstack((yA, yB)), {'ctxA': ctxA, 'ctxB': ctxB}

    def gradient(self, x, y=None, ctx=None):
        """Overrides the gradient method to combine the gradient from both nodes."""
        yA, yB = None, None
        if y is not None:
            yA = y[0:self.nodeA.dim_y]
            yB = y[self.nodeA.dim_y:]
        if ctx is None:
            ctx = {'ctxA': None, 'ctxB': None}
        dA = self.nodeA.gradient(x, yA, ctx['ctxA'])
        dB = self.nodeB.gradient(x, yB, ctx['ctxB'])
        return np.vstack((dA, dB))

# TODO: move to utility_nodes
class SelectNode(AbstractNode):
    """Extracts a subvector from the input node, allowing for example one component of
    a `ParallelNode` to be separated from the other."""

    def __init__(self, n, startIndx=0, endIndx=-1):
        if endIndx == -1: endIndx = n - 1
        assert (startIndx >= 0) and (endIndx < n) and (startIndx <= endIndx)
        super().__init__(n, endIndx - startIndx + 1)
        self.startIndx = startIndx
        self.endIndx = endIndx

    def solve(self, x):
        return x[self.startIndx:self.endIndx + 1], None

    def gradient(self, x, y=None, ctx=None):
        Dy = np.zeros((self.dim_y, self.dim_x))
        Dy[:, self.startIndx:self.endIndx + 1] = np.eye(self.dim_y)
        return Dy