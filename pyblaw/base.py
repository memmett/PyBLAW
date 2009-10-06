"""PyBLAW Base class.

"""

class Base(object):
    """Base class.

       Instance variables:

         * *trace*  - trace level

       Methods that should be overridden:

         * *allocate* - allocate memory etc
         * *pre_run*  - pre run initialisation

    """

    trace = 0                           # trace level

    def set_trace(self, trace_level):
        self.trace = trace_level

    def allocate(self):
        pass

    def pre_run(self, **kwargs):
        pass
