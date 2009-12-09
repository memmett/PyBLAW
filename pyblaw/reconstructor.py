"""PyBLAW abstract Reconstructor class.

"""

import os

import pyblaw.base


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
