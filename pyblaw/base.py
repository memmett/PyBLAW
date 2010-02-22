"""PyBLAW Base class.

"""

class Base(object):
    """Base class.

       Instance variables:

       * *trace* - trace level
       * *debug* - debug information (dictionary)

       Methods that should be overridden:

       * *allocate* - allocate memory etc
       * *pre_run*  - pre run initialisation

    """

    trace = 0                           # trace level
    debug = {}                          # debug information

    def set_trace(self, trace_level):
        self.trace = trace_level

    def set_debug(self, debug):
        self.debug.update(debug)

    def allocate(self):
        pass

    def pre_run(self, **kwargs):
        pass
