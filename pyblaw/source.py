"""PyBLAW abstract Source and concrete LinearQuad3Source classes.

"""

import pyblaw.base
import pyblaw.grid
import pyblaw.system

import pyblaw.clinearsource


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

    def source(self, qq, s):
        """Return source for each cell at given the reconstruction qq,
           and store the result in s."""

        raise NotImplementedError

######################################################################

class LinearQuad3Source(Source):
    """Linear 3-point quadrature source.

       This source uses a 3-point Gaussian quadrature to evaluate the
       linear source B, and is implemented in C (clinearsource).

       Arguments:

         * *B* - linear source matrix

    """

    def __init__(self, B):
        self.B = B

    def pre_run(self, **kwargs):
        pyblaw.clinearsource.init_linear_q3_source(self.B)

    def source(self, qq, s):
        pyblaw.clinearsource.linear_q3_source(qq, s)
