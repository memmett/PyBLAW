"""PyBLAW Base class.

"""

class Base(object):
    """Base class.

       Instance variables:

       * *trace* - trace level

       Methods that can be overridden:

       * *allocate* - allocate memory etc
       * *pre_run*  - pre run initialisation
       * *debug*    - debug

    """

    trace = 0                           # trace level
    debug = {}                          # debug information

    def set_trace(self, trace_level):
        self.trace = trace_level

    def debug(self):
        pass

    def allocate(self):
        pass

    def pre_run(self, **kwargs):
        pass
