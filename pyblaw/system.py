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

    def set_step(self, n, t):
        self.n = n
        self.t = t

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

       Define a simple system without any parameters.

       Arguments:

         * *p* - number of components
         * *q0* - initial condition (callable)
         * *m* - 'mass' component
         * *parameters* - parameters (dictionary)

       The initial condition function is called as q0(x, t), and
       should return a vector.

       The *mass* component (indexed starting at 1) is taken to be the
       mass.

    """

    def __init__(self, p, q0, mass=1, **kwargs):

        pyblaw.system.System(kwargs)

        self.p = p
        self.q0 = q0
        self.m = mass - 1

    def initial_conditions(self, t, q):

        for m in xrange(self.p):
            q[:,m] = self.grid.average(lambda x: self.q0(x, t)[m])

    def mass(self, q):

        return np.dot(q[:,self.m], self.grid.sizes())
