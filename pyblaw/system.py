"""PyBLAW system class.

   $Id: system.py,v 1.5 2009/10/02 02:49:45 memmett Exp $

"""

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
