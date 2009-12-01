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
            weno.precompute_reconstruction('left')
            weno.precompute_reconstruction('right')
            if self.n > 0:
                weno.precompute_reconstruction('gauss_quad%d' % (self.n))
            weno.cache(self.cache)
        else:
            weno = pyweno.weno.WENO(order=self.k, cache=self.cache)

        self.weno = weno


    def reconstruct(self, q, qm, qp, qq):

        p = self.system.p

        for m in xrange(p):
            self.weno.smoothness(q[:,m])
            self.weno.reconstruct(q[:,m], 'left', qp[:,m])
            self.weno.reconstruct(q[:,m], 'right', qm[:,m])
            if self.n > 0:
                self.weno.reconstruct(q[:,m], 'gauss_quad3', qq[:,:,m])

        qm[1:,:] = qm[:-1,:]            # XXX, this is sick, and will require
                                        # some niggly changes to
                                        # PyWENO to fix

