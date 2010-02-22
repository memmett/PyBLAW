"""PyBLAW HDF5 dumper.

"""

import h5py

import pyblaw.dumper

######################################################################

class H5PYDumper(pyblaw.dumper.Dumper):
    """HDF5 dumper (using the h5py Python package).

       Dump the cell averages q to an HDF5 file.  The hierarchy
       created within the HDF5 file is:

       * /dims/xdim - cell centres
       * /dims/tdim - dump times
       * /parameters/X - parameters
       * /data/q - cell averages of solution q

       The parameters are taken from the system (pyblaw.system.System).

       Arguments:

       * *output* - output file name

    """

    def __init__(self, output='output.h5'):

        self.output = output

    def init_dump(self):

        # initialise hdf
        hdf = h5py.File(self.output, "w")

        # x and t dimensions
        sgrp = hdf.create_group("dims")
        sgrp.create_dataset("xdim", data=self.x)
        sgrp.create_dataset("tdim", data=self.t)

        # parameters
        sgrp = hdf.create_group("parameters")
        for key, value in self.system.parameters.iteritems():
            sgrp.attrs[key] = value

        # data sets (solution q)
        sgrp = hdf.create_group("data")
        dset = sgrp.create_dataset("q", (len(self.t), len(self.x), self.system.p))

        # done
        hdf.close()

        self.last = 0


    def dump(self, q):
        """Dump solution to HDF5 data file."""

        hdf = h5py.File(self.output, "a")
        dset = hdf["data/q"]
        dset[self.last,:,:] = q[:,:]
        hdf.close()

        self.last = self.last + 1
