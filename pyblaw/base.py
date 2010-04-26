"""PyBLAW Base class.

"""

class Base(object):
    """Base class.

       **Instance variables**

       * *trace* - trace level

       **Methods that can be overridden**

       * *allocate* - allocate memory etc
       * *pre_run*  - pre run initialisation
       * *debug*    - debug

       **Methods**

    """

    M = 0                               # number of cells in this process
    trace = 0                           # trace level
    debug = {}                          # debug information

    def set_trace(self, trace_level):
        self.trace = trace_level

    def debug(self, **kwargs):
        """Perform any debugging checks (assertions) or display
        debugging information (this is called by the various PyBLAW
        classes after, eg, computing the flux)."""
        pass

    def allocate(self):
        """Allocate storage space etc."""
        pass

    def set_solver(self, solver):
        self.solver = solver

    def pre_run(self, **kwargs):
        """Perform any last minute initialisations (this is called by
        the solver after the initial condtions have been set)."""
        pass
