"""PyBLAW abstract Source and concrete LinearQuad3Source classes.

"""

import numpy as np

import pyblaw.base
import pyblaw.grid
import pyblaw.system

import pyblaw.cquad3source


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

    def source(self, qm, qp, qq, s):
        """Return next source for each cell given the left (-), right
           (+), and quadrature reconstructions *qm*, *qp* and *qq*;
           and store the result in *s*."""

        raise NotImplementedError


######################################################################

class SimpleSource(Source):
    """Simple numerical source.

       This source uses a user supplied numerical source.

       Arguments:

         * *source* - source function (callable)

       The source function is called as ``source(qm, qp, qq, dx, s)``.

       Implementing the source function in Cython (or similar) is
       strongly recommended.

    """

    def __init__(self, source):
        self.s = source

    def pre_run(self, **kwargs):
        self.dx = self.grid.x[1:] - self.grid.x[:-1]

    def source(self, qm, qp, qq, s):
        self.s(qm, qp, qq, self.dx, s)


######################################################################

# class GaussianQuad3Source(Source):
#     """Gaussian 3-point quadrature source.

#        This source uses a 3-point Gaussian quadrature to evaluate the
#        source *s*, and is implemented in C (cquad3source).

#        Arguments:

#          * *s* - source function (callable)

#        The source function *s* is called as ``s(q, f)`` where ``q`` is
#        a NumPy XXX

#        Implementing the source *s* in Cython is strongly recommended.

#        XXX: this is crap.

#     """

#     def __init__(self, s):
#         self.s = s

#     def allocate(self):
#         self.sq = np.zeros((self.grid.N, self.reconstructor.n, self.system.p))

# #    def pre_run(self, **kwargs):
#         #if self.reconstructor.n != 3:
#         #    raise ValueError, "reconstructor must reconstruct at 3 quadrature points"

#         #pyblaw.cquad3source.init_quad3_source()

#     def source(self, qq, s):
#         self.s(qq, self.sq)

#         pyblaw.cquad3source.quad3_source(self.sq, s)
