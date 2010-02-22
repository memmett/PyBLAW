"""PyBLAW abstract System class and a concrete linear system.

"""

import numpy as np

import pyblaw.base


class System(pyblaw.base.Base):
    """Abstract system.

       Define initial conditions, abstract mass, and other parameters.

       Instance variables:

       * *p*          - number of unknowns
       * *parameters* - parameters (dictionary)

       Instrance variables pulled from elsewhere:

       * *grid* - pyblaw.grid.Grid

       Methods that should be overridden:

       * *allocate*           - allocate memory
       * *initial_conditions* - set initial condtions at time t
       * *mass*               - compute 'mass' of system

    """

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

class SimpleSystem(System):
    """Simple system.

       The class implements a simple system that holds a dictionary of
       parameters and a callable *q0* for the initial conditions.

       Arguments:

       * *q0* - initial condition (callable)
       * *parameters* - parameters (dictionary)

       The initial condition function is called as q0(x, t), and
       should return a vector.

       The first component of the solution is taken to be the 'mass'
       of the system.

    """

    def __init__(self, q0, parameters={}):
        pyblaw.system.System(parameters)

        self.q0 = q0
        self.p = len(q0(0.0, 0.0))

    def initial_conditions(self, t, q):
        for m in xrange(self.p):
            q[:,m] = self.grid.average(lambda x: self.q0(x, t)[m])

    def mass(self, q):
        return np.dot(q[:,0], self.grid.sizes())
