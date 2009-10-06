"""PyBLAW Reconstructor class.

   $Id: reconstructor.py,v 1.3 2009/10/04 23:10:05 memmett Exp $

"""

import pyblaw.base
import pyblaw.grid
import pyblaw.system

class Reconstructor(pyblaw.base.Base):
    """Abstract reconstructor.

       Given the cell averages q, reconstruct the unkown at various
       points.

       Instance variables:

         * *n* - number of quadrature points per cell

       Instance variables pulled from elsewhere:

         * *grid*   - pyblaw.grid.Grid
         * *system* - pyblaw.system.System

       Methods that should be overridden:

         * *allocate*    - allocate memory etc
         * *reconstruct* - reconstruct

    """

    grid   = None
    system = None

    n = 0                               # XXX: move this to system?

    def set_grid(self, grid):
        self.grid = grid

    def set_system(self, system):
        self.system = system

    def reconstruct(self, q, ql, qr, qq):
        """Reconstruct q and store the result in ql (left), qr
        (right), and qq (quadrature)."""

        raise NotImplementedError
