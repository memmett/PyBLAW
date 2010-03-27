"""Setup environment to use Cython on POSIX operating systems."""

import os
import numpy
import pyximport


if os.name == 'posix':
    if os.environ.has_key('CFLAGS'):
        os.environ['CFLAGS'] = os.environ['CFLAGS'] + ' -I' + numpy.get_include()
    else:
        os.environ['CFLAGS'] = ' -I' + numpy.get_include()

    pyximport.install()
