"""Shallow-water solver.

   This shallow-water solver serves as short example of how to use the
   PyBLAW framework.

   It solves the depth-averaged shallow-water equations, over a
   non-flat bed, which are

     * height:   h_t + (u h)_x = 0
     * momentum: (h u)_t + ( h u^2 + 1/2 h^2 )_x = - h b_x

   where h is the depth of the fluid, u is the velocity of the fluid,
   and b is the bed topography.

   Throughout, q is:

     * q[i,0] - average depth in cell C_i
     * q[i,1] - average momentum in cell C_i
     * q[i,2] - average bed height in cell C_i (fixed)
     * q[i,3] - average bed height squared in cell C_i (fixed)

"""

import math
import os

import numpy as np
import pyweno.grid
import pyweno.weno
import pyblaw.wenosolver
import pyblaw.system

# import cython flux and source functions
import pyximport; pyximport.install()
import cshallowwater

# initial conditions (slightly perturbed still water from LeVeque)
epsilon = 0.2
def h0(x, t):
    if abs(x - 1.15) < 0.05:
        return 1.0 + epsilon  - b0(x,t)

    return 1.0 - b0(x,t)

def b0(x,t):
    if abs(x - 1.5) < 0.1:
        return 0.25 * ( math.cos( math.pi*(x - 1.5)/0.1 ) + 1.0 )

    return 0.0

def q0(x, t):
    return np.array([h0(x,t), 0.0, b0(x,t)])

# well-balanced reconstructor
class ShallowWaterReconstructor(pyblaw.reconstructor.Reconstructor):

    def __init__(self, order, cache):
        self.k = order
        self.cache = cache

    def pre_run(self, **kwargs):
        self.weno = pyweno.weno.WENO(order=self.k, cache=self.cache)

    def reconstruct(self, q, qm, qp, qq):
        weno = self.weno

        # flux and source reconstructions at boundaries
        weno.smoothness(q[:,0])

        weno.weights('left')
        for m in (0, 1, 2):
            weno.reconstruct(q[:,m], 'left', qp[:,m], False)

        weno.weights('right')
        for m in (0, 1, 2):
            weno.reconstruct(q[:,m], 'right', qm[:,m], False)

        # source reconstructions at quadrature points (see cshallowwater.pyx)
        weno.weights('gauss_quad3')
        weno.reconstruct(q[:,0], 'gauss_quad3', qq[:,:,0], False)
        weno.reconstruct(q[:,2], 'gauss_quad3', qq[:,:,1], False)

        weno.weights('d|gauss_quad3')
        weno.reconstruct(q[:,2], 'd|gauss_quad3', qq[:,:,2], False)

        qm[1:,:] = qm[:-1,:]            # XXX, tweak PyWENO so we don't have to do this


# define a solver to use our reconstructor and build/load a cache
class ShallowWaterSolver(pyblaw.solver.Solver):

    def __init__(self,
                 times=None,
                 dump_times=None,
                 cache='shallow_water_cache.mat', output='shallow_water.mat'):

        self.k      = 3
        self.cache  = cache
        self.output = output

        system        = pyblaw.system.SimpleSystem(q0)
        flux          = pyblaw.flux.SimpleFlux(cshallowwater.f)
        source        = pyblaw.source.SimpleSource(cshallowwater.s)
        reconstructor = ShallowWaterReconstructor(self.k, self.cache)
        evolver       = pyblaw.evolver.SSPERK3()
        dumper        = pyblaw.dumper.MATDumper(output)

        pyblaw.solver.Solver.__init__(self,
                                      system=system,
                                      reconstructor=reconstructor,
                                      flux=flux,
                                      source=source,
                                      evolver=evolver,
                                      dumper=dumper,
                                      times=times,
                                      dump_times=dump_times)

    def load_cache(self):
        if not os.access(self.cache, os.F_OK):
            return False

        self.grid = pyweno.grid.Grid(cache=self.cache, format='mat')
        self.weno = pyweno.weno.WENO(order=self.k, cache=self.cache, format='mat')

        return True

    def build_cache(self, x=None):
        grid = pyweno.grid.Grid(x)

        weno = pyweno.weno.WENO(grid=grid, order=self.k)
        weno.precompute_reconstruction('left')
        weno.precompute_reconstruction('right')
        weno.precompute_reconstruction('gauss_quad3')
        weno.precompute_reconstruction('d|gauss_quad3')
        weno.cache(self.cache)

        self.grid = grid
        self.weno = weno

# the solver
solver = ShallowWaterSolver(
    times=np.linspace(0.0, 1.0, 300+1),
    dump_times=np.linspace(0.0, 1.0, 150+1)
    )

# build/load grid and cache
if not solver.load_cache():
    solver.build_cache(np.linspace(0.0, 3.0, 300+1))

# giv'r!
solver.run()
