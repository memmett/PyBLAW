"""PyBLAW abstract Flux class.

   XXX: add some concrete fluxes

   XXX: add a 'linear flux' as a c extension

"""

import pyblaw.base
import pyblaw.grid
import pyblaw.system

import scipy.linalg
import pyblaw.clinearflux


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

    def flux(self, ql, qr, f):
        """Return net fluxes for each cell given the left and right
           reconstructions ql and qr, and store the result in f."""

        raise NotImplementedError


######################################################################

class LinearLFFlux(Flux):
    """Linear Lax-Friedrichs flux.

       This flux uses the Lax-Friedrichs numerical flux associated
       with the linear flux A, and is implemented in C.

       Arguments:

         * *A* - linear flux matrix

    """

    def __init__(self, A):
        self.A = A
        self.alpha = max(abs(scipy.linalg.eigvals(self.A)))

    def pre_run(self, **kwargs):
        self.dx = self.grid.x[1:] - self.grid.x[:-1]
        pyblaw.clinearflux.init_linear_lf_flux(self.A, self.alpha, self.dx)

    def flux(self, ql, qr, f):

        pyblaw.clinearflux.linear_lf_flux(ql, qr, f)
