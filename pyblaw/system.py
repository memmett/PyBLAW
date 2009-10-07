"""PyBLAW abstract System class and a concrete linear system.

"""

import pyblaw.base


class System(pyblaw.base.Base):
    """Abstract system.

       Define initial conditions, abstract mass, and other parameters.

       Instance variables:

         * *n*          - number of quadrature points per cell (for source)
         * *p*          - number of unknowns
         * *parameters* - parameters (dictionary)

       Instrance variables pulled from elsewhere:

         * *grid* - pyblaw.grid.Grid

       Methods that should be overridden:

         * *allocate*           - allocate memory
         * *initial_conditions* - set initial condtions at time t
         * *mass*               - compute 'mass' of system

    """

    n = 0                               # number of quadrature points
    p = 0                               # number of unknowns
    parameters = {}                     # parameters

    def __init__(self, parameters={}, **kwargs):
        self.parameters = parameters

    def set_grid(self, grid):
        self.grid = grid

    def initial_conditions(self, t, q):
        """Initialise q."""
        raise NotImplementedError

    def mass(self, q):
        """Compute 'mass' of q."""
        raise NotImplementedError


######################################################################

class LinearSystem(pyblaw.system.System):
    """Linear system.

       Define a linear system of hyperbolic equations of the form

         q_t + A q_x = B q.

       Arguments:

         * *A*  - linear flux matrix
         * *B*  - linear source matrix
         * *q0* - initial condition (callable)
         * *n*  - number of quadrature points
         * *parameters* - parameters (dictionary)

       The initial condition function is called as q0(x, t), and
       should return a vector.

       The first component is taken to be the 'mass'.

    """

    def __init__(self, A, B, q0, n=3, parameters={}):

        self.p = A.shape[0]
        self.A = A
        self.B = B

        self.q0 = q0

        self.n = n

        self.parameters = parameters

    def initial_conditions(self, t, q):

        for m in xrange(self.p):
            q[:,m] = self.grid.average(lambda x: self.q0(x, t)[m])


    def mass(self, q):

        return np.dot(q[:,0], self.grid.sizes())
