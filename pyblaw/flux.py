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
        """Return net flux for each cell given the left (-) and right
           (+) reconstructions *qm* and *qp*, and store the result in
           *f*."""

        raise NotImplementedError


######################################################################

class SimpleFlux(Flux):
    """Simple numerical flux.

       This flux uses a user supplied numerical flux.

       Arguments:

         * *flux* - flux function (callable)

       The flux function is called as ``flux(qm, qp, dx, f)``.

       Implementing the flux function in Cython (or similar) is
       strongly recommended.

    """

    def __init__(self, flux, init=None, **kwargs):
        self.f = flux

    def pre_run(self, **kwargs):
        self.dx = self.grid.x[1:] - self.grid.x[:-1]

    def flux(self, qm, qp, f):
        self.f(qm, qp, self.dx, f)


######################################################################

class LFFlux(Flux):
    """Lax-Friedrichs flux.

       This flux uses the Lax-Friedrichs numerical flux associated
       with the flux *f*, and is implemented in C (clfflux).

       Arguments:

         * *flux* - flux function (callable)
         * *alpha* - maximum wave speed
         * *virtual* - number of virtual cells on each side of the domain

       The flux function *f* is called as ``f(q, f)`` where ``q`` is
       the state vector and ``f`` is the resulting flux.

       Implementing the flux *f* in Cython is strongly recommended.

    """

    def __init__(self, flux, alpha, virtual):
        self.f = flux
        self.alpha = alpha
        self.virtual = virtual

    def allocate(self):
        self.fl = np.zeros((self.grid.N,self.system.p))
        self.fr = np.zeros((self.grid.N,self.system.p))
        self.fm = np.zeros((self.grid.N,self.system.p))
        self.fp = np.zeros((self.grid.N,self.system.p))

    def pre_run(self, **kwargs):
        self.dx = self.grid.x[1:] - self.grid.x[:-1]

        pyblaw.clfflux.init_lf_flux(self.alpha, self.virtual, self.dx, self.fl, self.fr)

    def flux(self, qm, qp, f):

        self.f(qm, self.fm)
        self.f(qp, self.fp)

        pyblaw.clfflux.lf_flux(qm, qp, self.fm, self.fp, f)
