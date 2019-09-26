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
        super().__init__(nodeA.dim_x, nodeB.dim_y)
        self.nodeA = nodeA
        self.nodeB = nodeB

    def solve(self, x):
        """Overrides the solve method to first compute z = nodeA.solve(x) and then y = nodeB.solve(z). Returns
        the dual variables from both nodes."""
        z, nuA = self.nodeA.solve(x)
        y, nuB = self.nodeB.solve(z)
        return y, {'nuA': nuB, 'nuB': nuB, 'z': z}

    def gradient(self, x, y=None, ctx=None):
        """Overrides the gradient method to compute the composed gradient by the chain rule."""

        # we need to resolve for z since there is currently no way to store this
        if ctx is None:
            z, _ = self.nodeA.solve(x)
        else:
            z = ctx['z']
        Dz = self.nodeA.gradient(x, z)
        Dy = self.nodeB.gradient(z, y)
        return np.dot(Dy, Dz)
