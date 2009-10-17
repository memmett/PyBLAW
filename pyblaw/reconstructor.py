"""PyBLAW abstract Reconstructor class and a concrete WENO reconstructor.

"""

import os

import pyblaw.base
import pyblaw.grid
import pyblaw.system

import pyweno.weno
import scipy.sparse


class Reconstructor(pyblaw.base.Base):
    """Abstract reconstructor.

       Given the cell averages q, reconstruct the unkown at various
       points.

       Instance variables:

         * *n*          - number of quadrature points per cell (for source)

       Instance variables pulled from elsewhere:

         * *grid*   - pyblaw.grid.Grid
         * *system* - pyblaw.system.System

       Methods that should be overridden:

         * *allocate*    - allocate memory etc
         * *reconstruct* - reconstruct

    """

    n = 0                               # number of quadrature points

    grid   = None
    system = None

    def set_grid(self, grid):
        self.grid = grid

    def set_system(self, system):
        self.system = system

    def reconstruct(self, q, qm, qp, qq):
        """Reconstruct q and store the result in qm (-), qp (+), and
        qq (quadrature)."""

        raise NotImplementedError


class WENOReconstructor(Reconstructor):
    """WENO Reconstructor.

       XXX
    """

    def __init__(self, order, quad, cache):
        self.k = order
        self.n = quad
        self.cache = cache

    def allocate(self):

        #### weno reconstructor for qm, qp
        if not os.access(self.cache, os.F_OK):
            weno = pyweno.weno.WENO(grid=self.grid, order=self.k)
            weno.cache(self.cache)
        else:
            weno = pyweno.weno.WENO(order=self.k, cache=self.cache)

        self.weno = weno

        #### centered polynomial for qq
        N = self.grid.N
        k = 2*self.k - 1
        r = self.k
        n = self.n
        stencil = pyweno.stencil.Stencil(order=k, shift=r, quad=n, grid=self.grid)

        # build reconstructor matrix
        QUAD = scipy.sparse.lil_matrix((N*n,N))

        for i in xrange(2*k):
            for l in xrange(n):
                QUAD[i+l,i] = 1.0

        for i in xrange(2*k, N-2*k):
            for l in xrange(n):
                QUAD[i*n+l,i-r:i-r+k] = stencil.c_q[i,l,:]

        for i in xrange(N-2*k, N):
            for l in xrange(n):
                QUAD[i+l,i] = 1.0

        self.QUAD   = QUAD.tocsr()

        # XXX: cache the QUAD matrix...

    def reconstruct(self, q, qm, qp, qq):

        p = self.system.p

        for m in xrange(p):
            self.weno.reconstruct(q[:,m], qm[:,m], qp[:,m])
            qq[:,m] = self.QUAD * q[:,m]
