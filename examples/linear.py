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
import pyweno.stencil


def initial_condition(x, t):

    if x < -t or x > t:
        return np.zeros(3)

    return np.array([ math.cos(0.5*math.pi*x/t), 0.0, 0.0 ])



######################################################################
# reconstructor
#

# XXX: move this to pyblaw

class PolynomialReconstructor(pyblaw.reconstructor.Reconstructor):
    """Polynomial reconstructor.

       The unknown q is recontructed at the cell boundaries and
       quadrature points using a high order polynomial interpolation.

       To use: don't do anything (or, adjust matrices to handle your
               boundary conditions appropriately).

       NOTE: If your solution has a discontinuity, this recontructor
             will do a very poor job of recontructing.  Try using a
             WENO recontructor instead.

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

        # compute reconstruction coeffs
        r = (k / 2) + (k % 2)
        stencil = pyweno.stencil.Stencil(order=k, shift=r, quad=n, grid=self.grid)

        # build reconstructor matrix
        BNDRYL = scipy.sparse.lil_matrix((N,N))
        BNDRYR = scipy.sparse.lil_matrix((N,N))
        QUAD   = scipy.sparse.lil_matrix((N*n,N))

        # boundary condition at left of domain
        for i in xrange(2*k):
            BNDRYL[i,i] = 1.0
            BNDRYR[i,i] = 1.0
            for l in xrange(n):
                QUAD[i+l,i] = 1.0

        # middle of domain: don't do anything here
        for i in xrange(2*k, N-2*k):
            BNDRYL[i,i-1-r:i-1-r+k] = stencil.c_p[i-1,:]
            BNDRYR[i,i-r:i-r+k]     = stencil.c_m[i,:]
            for l in xrange(n):
                QUAD[i*n+l,i-r:i-r+k] = stencil.c_q[i,l,:]

        # boundary condition at right of domain
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
# solver
#

# XXX: move this to pyblaw

class ExampleLinearSolver(pyblaw.solver.Solver):
    """Example solver for a linear system.

       To use: tweak if you need to...

    """

    A = np.matrix('2.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 -2.0')
    B = np.matrix('0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 -0.1 0.1')

    def __init__(self, order=8, cell_boundaries=None, times=None):

        grid          = pyblaw.grid.Grid(boundaries=cell_boundaries)
        system        = pyblaw.system.LinearSystem(self.A, self.B, initial_condition)
        reconstructor = PolynomialReconstructor(order)
        flux          = pyblaw.flux.LinearLFFlux(system.A)
        source        = pyblaw.source.LinearQuad3Source(system.B)
        evolver       = pyblaw.evolver.SSPERK3()
        dumper        = pyblaw.dumper.H5PYDumper('output.h5')

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

solver = ExampleLinearSolver(order=k,
                             cell_boundaries=x,
                             times=t)

profile.run('solver.run()', 'linear.prof')
p = pstats.Stats('linear.prof')
p.strip_dirs().sort_stats('time', 'cum').print_stats(10)
