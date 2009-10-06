"""PyBLAW Source class.

   $Id: source.py,v 1.4 2009/10/02 02:49:45 memmett Exp $

"""

import pyblaw.base
import pyblaw.grid
import pyblaw.system

class Source(pyblaw.base.Base):
    """Abstract source.

       Compute the source for each cell given the cell averages q.

       Instance variables pulled from elsewhere:

         * *grid*   - pyblaw.grid.Grid
         * *system* - pyblaw.system.System
         * *reconstructor* - pyblaw.reconstructor.Reconstructor

       Methods that should be overridden:

         * *allocate* - allocate memory etc
         * *source*   - compute sources

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

    def source(self, q, s):
        """Return source for each cell at given the reconstruction qs,
           and store the result in s."""

        raise NotImplementedError
