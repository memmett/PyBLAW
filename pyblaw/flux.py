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

       **Instance variables pulled from elsewhere**

       * *grid*   - pyblaw.grid.Grid
       * *system* - pyblaw.system.System
       * *reconstructor* - pyblaw.system.Reconstructor

       **Methods that should be overridden**

       * *allocate* - allocate memory etc
       * *flux*     - compute net fluxes

       **Methods**

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

    def flux(self, qm, qp, f, **kwargs):
        """Return net flux for each cell given the left (-) and right
           (+) reconstructions *qm* and *qp*, and store the result in
           *f*."""

        raise NotImplementedError


######################################################################

class SimpleFlux(Flux):
    """Simple numerical flux.

       This flux uses a user supplied numerical flux.

       Implementing the flux function in Cython (or similar) is
       strongly recommended.

       **Arguments:**

       * *flux* - flux function (callable)

       The numerical flux function is called as ``flux(qm, qp, dx, f,
       **kwargs)`` where:

       * ``qm[i,:]`` and ``qp[i,:]`` are the reconstructions of q at
         the cell boundaries from the left (-) and right (+)
         respectively;

       * ``dx`` are the cell sizes;

       * ``f`` is the resulting flux; and

       * ``kwargs`` contains:

         * ``n``: the current step,
         * ``t``: the current time, and
         * any entries passed to the solver or set by the reconstructor.

    """

    def __init__(self, flux, **kwargs):
        self.f = flux

    def pre_run(self, **kwargs):
        self.dx = self.grid.x[1:] - self.grid.x[:-1]

    def flux(self, qm, qp, f, **kwargs):
        self.f(qm, qp, self.dx, f, **kwargs)

        if __debug__:
            self.debug(qm=qm, qp=qp, f=f, **kwargs)


######################################################################

class LFFlux(Flux):
    """Lax-Friedrichs flux.

       This flux uses the Lax-Friedrichs numerical flux associated
       with a given flux and is implemented in C (clfflux).

       Implementing the flux in Cython is strongly recommended.

       **Arguments:**

       * *flux*     - flux function (callable)
       * *alpha*    - maximum wave speed

       The (non-numerical) flux function *flux* is called as ``flux(q,
       f, **kwargs)`` where:

       * ``q[i,:]`` is the state vector of q at the cell boundaries;
       * ``f`` is the resulting flux; and
       * ``kwargs`` contains:

         * ``n``: the current step,
         * ``t``: the current time, and
         * any entries passed to the solver or set by the reconstructor.

    """

    def __init__(self, flux, alpha):
        self.f = flux
        self.alpha = alpha


    def allocate(self):
        N = self.grid.size
        p = self.system.p

        self.fl = np.zeros((N+1,p))
        self.fr = np.zeros((N+1,p))
        self.fm = np.zeros((N+1,p))
        self.fp = np.zeros((N+1,p))


    def pre_run(self, **kwargs):
        self.dx = self.grid.x[1:] - self.grid.x[:-1]

        pyblaw.clfflux.init_lf_flux(self.alpha, 0, self.dx, self.fl, self.fr)


    def flux(self, qm, qp, f, **kwargs):

        self.f(qm, self.fm, **kwargs)
        self.f(qp, self.fp, **kwargs)

        pyblaw.clfflux.lf_flux(qm, qp, self.fm, self.fp, f)

        if __debug__:
            self.debug(qm=qm, qp=qp, f=f, **kwargs)

        return kwargs
