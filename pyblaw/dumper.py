"""PyBLAW abstract Dumper class and MAT dumper.

"""

import numpy as np
import scipy.io as sio


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


######################################################################

class MATDumper(Dumper):
    """MAT dumper (using SciPy).

       Dump the cell averages q to a MAT file.  The matrices created
       within the MAT file are:

         * dims.xdim - cell centres
         * dims.tdim - dump times
         * parameters.X - parameters
         * data.q - cell averages of solution q

       The parameters are taken from the system (pyblaw.system.System).

       The H5Dumper in pyblaw.h5dumper is more efficient.

       Arguments:

         * *output* - output file name

    """

    def __init__(self, output='output.mat'):

        self.output = output

    def init_dump(self):

        mat = {}

        # x and t dimensions
        mat['dims.xdim'] = self.x
        mat['dims.tdim'] = self.t

        # parameters
        for key in self.system.parameters:
            mat[key] = self.system.parameters[key]

        # data
        mat['data.q'] = np.zeros((len(self.t), len(self.x), self.system.p))

        # done
        sio.savemat(self.output, mat)

        self.last = 0


    def dump(self, q):
        """Dump solution to MAT data file."""

        mat = sio.loadmat(self.output, struct_as_record=True)
        mat['data.q'][self.last,:,:] = q[:,:]
        sio.savemat(self.output, mat)

        self.last = self.last + 1
