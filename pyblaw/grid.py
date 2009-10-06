"""PyBLAW Grid class.

   XXX: remove pyweno dependency

"""

import pyblaw.base
import pyweno.grid

class Grid(pyblaw.base.Base, pyweno.grid.Grid):
    """Unstructured spatial grid (discretisation).

       See pyweno.grid.Grid

    """
    pass
