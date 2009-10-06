"""Shallow-water solver.

   This shallow-water solver serves as short example of how to use the
   PyBLAW framework.

   It solves the depth-averaged shallow-water equations, which are

     * height:   h_t + (u h)_x = 0
     * momentum: (u h)_t + ( u^2 h + 1/2 h^2 )_x = - drag u^2

   where h is the height and u is the velocity of the fluid, by using
   a high order polynomial reconstruction of q.

   Throughout, q is:

     * q[i,0] - average height in cell C_i
     * q[i,1] - average momentum in cell C_i

   NOTE: This is a pure python implementation and is **VERY** slow.
   To speed this up, one should

     * cache the grid and recontruction matrices;
     * avoid calculating the same thing twice; and
     * use a hybrid approach (ie, write the flux and source functions in C).

"""

import math
import cProfile as profile
import pstats

import h5py as h5
import numpy as np
import scipy.sparse

import pyblaw.system
import pyblaw.flux
import pyblaw.source
import pyblaw.reconstructor
import pyblaw.evolver
import pyblaw.dumper
import pyblaw.solver
import pyweno.stencil

######################################################################
# system
#

class ShallowWaterSystem(pyblaw.system.System):

    p = 2                               # number of components
    drag = 0.0                          # drag coeff

    def __init__(self, drag):
        self.drag = drag

    def h0(self, x, t):
        if x < -t or x > t:
            return 1.0

        return math.cos(math.pi/2.0 * x/t) + 1.0

    def initial_conditions(self, t, q):
        q[:,0] = self.grid.average(lambda x: self.h0(x, t))
        q[:,1] = self.grid.average(lambda x: 0.0)

    def mass(self, q):
        return np.dot(q[:,0], self.grid.sizes())


######################################################################
# flux
#

class ShallowWaterFlux(pyblaw.flux.Flux):

    def pre_run(self, **kwargs):
        self.dx = self.grid.x[1:] - self.grid.x[:-1]

    def f_q(self, q):
        """f"""
        if q[0] > 0.0:
            return np.array([ q[1], q[1]*q[1]/q[0] + 0.5 * q[0]*q[0] ])

        return np.zeros(self.system.p)

    def flux_lf(self, ql, qr):
        """Lax-Friedrichs flux."""

        # alpha is taken to be 2.0 here...
        return 0.5 * (self.f_q(ql) + self.f_q(qr) - 2.0 * (qr - ql) )

    def flux(self, ql, qr, f):
        """Net flux (override base class)."""

        N  = self.grid.N
        dx = self.dx

        for i in xrange(N-1):
            fl = self.flux_lf(ql[i],   qr[i])
            fr = self.flux_lf(ql[i+1], qr[i+1])
            f[i] = - (fr - fl) / dx[i]


######################################################################
# source
#

class ShallowWaterSource(pyblaw.source.Source):

    def source(self, qq, s):
        # XXX: add drag
        pass


######################################################################
# reconstructor
#

class ShallowWaterReconstructor(pyblaw.reconstructor.Reconstructor):

    n = 3                               # number of quadrature points

    def __init__(self, order):
        self.k = order


    def allocate(self):

        N = self.grid.N
        p = self.system.p
        k = self.k
        n = self.n

        print "building reconstructors..."

        # compute reconstruction coeffs
        r = (k / 2) + (k % 2)
        stencil = pyweno.stencil.Stencil(order=k, shift=r, quad=n, grid=self.grid)

        # build reconstructor matrix
        BNDRYL = scipy.sparse.lil_matrix((N,N))
        BNDRYR = scipy.sparse.lil_matrix((N,N))
        QUAD   = scipy.sparse.lil_matrix((N*n,N))

        for i in xrange(2*k):
            BNDRYL[i,i] = 1.0
            BNDRYR[i,i] = 1.0
            for l in xrange(n):
                QUAD[i+l,i] = 1.0

        for i in xrange(2*k, N-2*k):
            BNDRYL[i,i-1-r:i-1-r+k] = stencil.c_p[i-1,:]
            BNDRYR[i,i-r:i-r+k]     = stencil.c_m[i,:]
            for l in xrange(n):
                QUAD[i*n+l,i-r:i-r+k] = stencil.c_q[i,l,:]

        for i in xrange(N-2*k, N):
            BNDRYL[i,i] = 1.0
            BNDRYR[i,i] = 1.0
            for l in xrange(n):
                QUAD[i+l,i] = 1.0

        self.BNDRYL = BNDRYL.tocsr()
        self.BNDRYR = BNDRYR.tocsr()
        self.QUAD   = QUAD.tocsr()

        print "building reconstructors... done."


    def reconstruct(self, q, ql, qr, qq):

        p = self.system.p

        # XXX: use weno instead of central polynomial...

        for m in xrange(p):
            ql[:,m] = self.BNDRYL * q[:,m]
            qr[:,m] = self.BNDRYR * q[:,m]
            qq[:,m] = self.QUAD * q[:,m]


######################################################################
# dumper
#

class ShallowWaterDumper(pyblaw.dumper.Dumper):

    def __init__(self, output='output.hdf5'):
        self.output = output

    def init_dump(self):

        # initialise hdf
        hdf = h5.File(self.output, "w")

        # x and t dimensions
        sgrp = hdf.create_group("dims")
        sgrp.create_dataset("xdim", data=self.x)
        sgrp.create_dataset("tdim", data=self.t)

        # parameters
        sgrp = hdf.create_group("parameters")
        sgrp.attrs['drag'] = self.system.drag

        # data sets (solution q)
        sgrp = hdf.create_group("data")
        dset = sgrp.create_dataset("q", (len(self.t), len(self.x), self.system.p))

        # done
        hdf.close()

        self.last = 0

    def dump(self, q):

        hdf = h5.File(self.output, "a")
        dset = hdf["data/q"]
        dset[self.last,:,:] = q[:,:]
        hdf.close()

        self.last = self.last + 1

######################################################################
# solver
#

class ShallowWaterSolver(pyblaw.solver.Solver):

    def __init__(self, drag=0.001, order=8, cell_boundaries=None, times=None):

        grid          = pyblaw.grid.Grid(boundaries=cell_boundaries)
        system        = ShallowWaterSystem(drag)
        reconstructor = ShallowWaterReconstructor(order)
        flux          = ShallowWaterFlux()
        source        = ShallowWaterSource()
        evolver       = pyblaw.evolver.SSPERK3()
        dumper        = ShallowWaterDumper()

        pyblaw.solver.Solver.__init__(self,
                                      grid=grid,
                                      system=system,
                                      reconstructor=reconstructor,
                                      flux=flux,
                                      source=source,
                                      evolver=evolver,
                                      dumper=dumper,
                                      times=times)

        # check cfl condition (max eigenvalue is taken to be 2...)
        if 2.0 * max(self.dt) >= 0.5 * min(self.dx):
            print ('WARNING: cfl condition not satisfied (2 dt = %.4e >= %.4e = 0.5 dx)'
                   % (2.0*max(self.dt), 0.5*min(self.dx)))


######################################################################
# giv'r!
#

k = 8
x = np.linspace(-400.0, 400.0, 1600+1)
t = np.linspace(100.0,  110.0, 100+1)

solver = ShallowWaterSolver(drag=0.001,
                            order=k,
                            cell_boundaries=x,
                            times=t)

profile.run("solver.run()", 'shallow_water.prof')
p = pstats.Stats('shallow_water.prof')
p.strip_dirs().sort_stats('time', 'cum').print_stats(10)
