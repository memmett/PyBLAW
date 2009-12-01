"""PyBLAW abstract Flux and concrete LinearLFFlux classes.

"""

import numpy as np

import pyblaw.base
import pyblaw.grid
import pyblaw.system

import scipy.linalg
import pyblaw.clfflux


######################################################################

class Flux(pyblaw.base.Base):
    """Abstract flux.

       Compute the net flux for each cell given the cell averages q.

       Instance variables pulled from elsewhere:

         * *grid*   - pyblaw.grid.Grid
         * *system* - pyblaw.system.System
         * *reconstructor* - pyblaw.system.Reconstructor

       Methods that should be overridden:

         * *allocate* - allocate memory etc
         * *flux*     - compute net fluxes

    """

    grid          = None
    system        = None
    reconstructor = None

    def set_grid(self, grid):
        self.grid = grid

    def set_system(self, system):
        self.system = system

    def set_reconstructor(self, reconstructor):
        self.reconstructor = reconstructor

    def flux(self, qm, qp, f):
        """Return net fluxes for each cell given the left (-) and
           right (+) reconstructions qm and qp, and store the result
           in f."""

        raise NotImplementedError


######################################################################

class LFFlux(Flux):
    """Lax-Friedrichs flux.

       This flux uses the Lax-Friedrichs numerical flux associated
       with the flux *f*, and is implemented in C (clfflux).

       Arguments:

         * *f* - flux function (callable)
         * *alpha* - XXX

       The flux function *f* is called as ``f(q, f)`` where ``q`` is a
       NumPy XXX

       Implementing the flux *f* in Cython is strongly recommended.

    """

    def __init__(self, f, alpha):
        self.f = f
        self.alpha = alpha

    def allocate(self):

        self.fl = np.zeros((self.grid.N,self.system.p))
        self.fr = np.zeros((self.grid.N,self.system.p))
        self.fm = np.zeros((self.grid.N,self.system.p))
        self.fp = np.zeros((self.grid.N,self.system.p))


    def pre_run(self, **kwargs):
        self.dx = self.grid.x[1:] - self.grid.x[:-1]

        pyblaw.clfflux.init_lf_flux(self.alpha, self.dx, self.fl, self.fr)

    def flux(self, qm, qp, f):

        self.f(qm, self.fm)
        self.f(qp, self.fp)

        pyblaw.clfflux.lf_flux(qm, qp, self.fm, self.fp, f)
