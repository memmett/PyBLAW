"""PyBLAW Dumper class.

   $Id: dumper.py,v 1.3 2009/10/04 23:10:05 memmett Exp $

"""

class Dumper(object):
    """Abstract dumper.

       Dump the cell averages q to a file.

       Instance variables:

         * *x* - cell centers
         * *t* - times

       Methods that should be overridden:

         * *init_dump* - init and create dump file etc
         * *dump*      - dump solution q

    """

    x = []
    t = []

    def set_system(self, system):
        self.system = system

    def set_dims(self, x, t):
        self.x = x
        self.t = t

    def init_dump(self):
        """Initialise dumper instance, create dump file, etc."""

        raise NotImplementedError

    def dump(self, q):
        """Dump q."""

        raise NotImplementedError
