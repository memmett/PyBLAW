"""PyBLAW Grid class.

   $Id: grid.py,v 1.3 2009/09/03 16:50:09 memmett Exp $

   """

import pyblaw.base
import pyweno.grid

class Grid(pyblaw.base.Base, pyweno.grid.Grid):
    """Unstructured spatial grid (discretisation).

       See pyweno.grid.Grid

    """
    pass
