"""PyBLAW Flux class.

   $Id: flux.py,v 1.5 2009/10/02 02:49:45 memmett Exp $

"""

import pyblaw.base
import pyblaw.grid
import pyblaw.system

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
