"""Linear system solver.

   This linear solver serves as short example of how to use the PyBLAW
   framework.

   It solves a system of hyperbolic equations of the form

     q_t + A q_x = B q

   where A and B are constant (and possibly singular) matrices.

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
import scipy.linalg
import scipy.sparse

import pyblaw.system
import pyblaw.flux
import pyblaw.source
import pyblaw.reconstructor
import pyblaw.evolver
import pyblaw.dumper
import pyblaw.solver
import pyweno.weno

######################################################################
# system
#

class ExampleSystem(pyblaw.system.System):
    """Example system.

       To use:

         1. set p, A, B for your system;
         2. set the initial conditions q1, q2, and q3 (1st, 2nd, and
            3rd components of respectively); and
         3. tweak the 'mass' if you want.

    """

    p = 3
    A = np.matrix('2 1 0; 1 -2.1 0; 0 0 0')
    B = np.matrix('0 0 0.8; 0 0 0.5; 0.1 0.1 0')

    def q1(self, x, t):
        if t <= 0.0:
            return 0.0

        if x < -t or x > t:
            return 0.0

        return math.cos(math.pi/2.0 * x/t)

    def q2(self, x, t):
        return 0.0

    def q3(self, x, t):
        return 0.0

    def initial_conditions(self, t, q):
        q[:,0] = self.grid.average(lambda x: self.q1(x, t))
        q[:,1] = self.grid.average(lambda x: self.q2(x, t))
        q[:,2] = self.grid.average(lambda x: self.q3(x, t))

    def mass(self, q):
        return np.dot(q[:,0], self.grid.sizes())


######################################################################
# flux
#

class LinearFlux(pyblaw.flux.Flux):
    """Linear flux.

       The net flux is computed by taking the Lax-Friedrichs flux
       associated with the linear flux A.

       To use: don't do anything.

    """

    def __init__(self, A):
        self.A = A

    def pre_run(self, **kwargs):
        self.dx = self.grid.x[1:] - self.grid.x[:-1]
        self.alpha = max(abs(scipy.linalg.eigvals(A)))

    def f_q(self, q):
        """Flux (linear)."""
        return np.dot(self.A, q)

    def flux_lf(self, ql, qr):
        """Lax-Friedrichs flux."""
        return 0.5 * (self.f_q(ql) + self.f_q(qr) - self.alpha * (qr - ql) )

    def flux(self, ql, qr, f):
        """Net flux."""

        N  = self.grid.N
        dx = self.dx

        for i in xrange(N-1):
            fl = self.flux_lf(ql[i],   qr[i])
            fr = self.flux_lf(ql[i+1], qr[i+1])
            f[i] = - (fr - fl) / dx[i]


######################################################################
# source
#

class LinearSource(pyblaw.source.Source):
    """Linear source.

       The net source is computed by 3-point Gaussian quadrature (see
       LinearReconstructor).

       To use: don't do anything.

    """

    def __init__(self, B):
        self.B = B

    def pre_run(self, **kwargs):

        if self.reconstructor.n != 3:
            raise NotImplementedError, "only 3-point quadrature supported"

        self.w1 = 5.0/9.0
        self.w2 = 8.0/9.0
        self.w3 = 5.0/9.0

    def source(self, qq, s):

        N = self.grid.N
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3

        for i in xrange(N):
            s[i] = np.dot(self.B, w1 * qq[i*3+0,:] + w2 * qq[i*3+1,:] + w3 * qq[i*3+2,:])


######################################################################
# reconstructor
#

class WENOReconstructor(pyblaw.reconstructor.Reconstructor):
    """WENO reconstructor.

       The unknown q is recontructed at the cell boundaries and
       quadrature points using a high order polynomial interpolation.

       To use: XXX.

       XXX: quadrature is not WENO.

    """

    n = 3                               # number of quadrature points

    def __init__(self, order):
        self.k = order


    def allocate(self):

        N = self.grid.N
        p = self.system.p
        k = self.k
        n = self.n

        print "building reconstructors..."


        # weno
        self.weno = pyweno.weno.WENO(grid=self.grid, order=k/2)

        # quad
        r = (k / 2) + (k % 2)
        stencil = pyweno.stencil.Stencil(order=k, shift=r, quad=n, grid=self.grid)

        QUAD    = scipy.sparse.lil_matrix((N*n,N))

        for i in xrange(2*k):
            for l in xrange(n):
                QUAD[i+l,i] = 1.0

        for i in xrange(2*k, N-2*k):
            for l in xrange(n):
                QUAD[i*n+l,i-r:i-r+k] = stencil.c_q[i,l,:]

        for i in xrange(N-2*k, N):
            for l in xrange(n):
                QUAD[i+l,i] = 1.0

        self.QUAD   = QUAD.tocsr()

        print "building reconstructors... done."


    def reconstruct(self, q, ql, qr, qq):

        p = self.system.p

        # boundary points
        (v_p, v_m) = self.weno.reconstruct(q)

        # XXX: set ql and qr

        # quadrature points
        for m in xrange(p):
            qq[:,m] = self.QUAD * q[:,m]


######################################################################
# dumper
#

class HDF5Dumper(pyblaw.dumper.Dumper):
    """HDF5 dumper.

       The solution is dumped to an HDF5 file using h5py.

       To use: tweak if you need to...

    """

    def __init__(self, output='output.hdf5', parameters=None):
        self.output = output
        self.parameters = parameters


    def init_dump(self):

        # initialise hdf
        hdf = h5.File(self.output, 'w')

        # x and t dimensions
        sgrp = hdf.create_group('dims')
        sgrp.create_dataset('xdim', data=self.x)
        sgrp.create_dataset('tdim', data=self.t)

        # parameters
        sgrp = hdf.create_group('parameters')
        for key, value in self.parameters.iteritems():
            sgrp.attrs[key] = str(value)

        # data sets (solution q)
        sgrp = hdf.create_group('data')
        dset = sgrp.create_dataset('q', (len(self.t), len(self.x), self.system.p))

        # done
        hdf.close()

        self.last = 0


    def dump(self, q):

        hdf = h5.File(self.output, 'a')
        dset = hdf['data/q']
        dset[self.last,:,:] = q[:,:]
        hdf.close()

        self.last = self.last + 1

######################################################################
# solver
#

class ExampleLinearSolver(pyblaw.solver.Solver):
    """Example solver for a linear system.

       To use: tweak if you need to...

    """

    def __init__(self, A=None, B=None, order=8, cell_boundaries=None, times=None):

        grid          = pyblaw.grid.Grid(boundaries=cell_boundaries)
        system        = ExampleLinearSystem()
        reconstructor = PolynomialReconstructor(order)
        flux          = LinearFlux(system.A)
        source        = LinearSource(system.B)
        evolver       = pyblaw.evolver.SSPERK3()
        dumper        = HDF5Dumper('output.hdf5', parameters={'A': system.A, 'B': system.B})

        pyblaw.solver.Solver.__init__(self,
                                      grid=grid,
                                      system=system,
                                      reconstructor=reconstructor,
                                      flux=flux,
                                      source=source,
                                      evolver=evolver,
                                      dumper=dumper,
                                      times=times)


    def pre_run(self, **kwargs):

        if self.flux.alpha * max(self.dt) >= 0.5 * min(self.dx):
            print ("WARNING: cfl condition not satisfied (alpha dt = %.4e >= %.4e = 0.5 dx)"
                   % (2.0*max(self.dt), 0.5*min(self.dx)))

        print "running..."


######################################################################
# giv'r!
#

k = 8
x = np.linspace(-400.0, 400.0, 1600+1)
t = np.linspace(100.0,  110.0, 100+1)

solver = SimpleSolver(order=k,
                      cell_boundaries=x,
                      times=t)

profile.run('solver.run()', 'linear.prof')
p = pstats.Stats('linear.prof')
p.strip_dirs().sort_stats('time', 'cum').print_stats(10)
