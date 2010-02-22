"""PyBLAW WENOCLAW reconstructor and solver class.

"""

import os
import numpy as np

import pyblaw.flux
import pyblaw.source
import pyblaw.reconstructor
import pyblaw.evolver
import pyblaw.solver
import pyweno.grid
import pyweno.weno


######################################################################

class WENOCLAWReconstructor(pyblaw.reconstructor.Reconstructor):
    """WENO CLAW Reconstructor.

       **Arguments:**

       * *order* - WENO reconstruction order k
       * *cache* - MAT cache file name

       The pyweno.weno.WENO object is loaded from the cache, which
       must be pre-built.

       The smoothness is based on the first component only.

    """

    def __init__(self, order=3, cache='cache.mat'):
        self.k = order
        self.cache = cache


    def pre_run(self, **kwargs):

        self.weno = pyweno.weno.WENO(order=self.k, cache=self.cache)


    def reconstruct(self, q, qm, qp, qq):

        p = q.shape[1]

        self.weno.smoothness(q[:,0])

        self.weno.weights('left')
        for m in range(p):
            self.weno.reconstruct(q[:,m], 'left', qp[:,m], False)

        self.weno.weights('right')
        for m in range(p):
            self.weno.reconstruct(q[:,m], 'right', qm[:,m], False)

        qp[-1,:] = qm[-1,:]
        qm[1:,:] = qm[:-1,:]
        qm[0,:]  = qp[0,:]


######################################################################

class WENOCLAWLFSolver(pyblaw.solver.Solver):
    """WENO conservation law solver using a Lax-Friedrichs flux.

       **Arguments:**

       * *flux* - a dictionary containing two entries: *f* and *alpha* (see below)
       * *order* - WENO reconstruction order k
       * *system* - a PyBLAW system
       * *evolver* - a PyBLAW evolver or None (defaults to pyblaw.evolver.SSPERK3)
       * *dumper* - a PyBLAW dumper or None (defaults to pyblaw.dumper.MATDumper)
       * *times* - NumPy array of times
       * *cache* - MAT cache file name (defaults to 'cache.mat')
       * *output* - MAT output file name (defaults to 'output.mat')

       The entries of the *flux* dictionary are:

       * *f* - a callable ``f(q, t, f)`` that computes the flux given the
         vector of states *q* and stores the results in the vector *f*
       * *alpha* - maximum wave speed for the Lax-Friedrichs flux
       * *virtual* - number of cells on each side of domain to zero
          during the LF process
       * *bc* - XXX: a callable ``bc(t, f)`` that, if defined, optionally
          updates/overwrites the net flux *f*
    """

    def __init__(self,
                 flux={},
                 order=3,
                 system=None, evolver=None, dumper=None,
                 cache='cache.mat', output='output.mat',
                 **kwargs):

        self.f       = flux
        self.k       = order
        self.cache   = cache
        self.output  = output

        if evolver is None:
            evolver = pyblaw.evolver.SSPERK3()

        if dumper is None:
            dumper = pyblaw.dumper.MATDumper(output)

        reconstructor = WENOCLAWReconstructor(order=self.k, cache=self.cache)

        if 'bc' in flux:
            bc = flux['bc']
        else:
            bc = None

        if 'virtual' in flux:
            virtual = flux['virtual']
        else:
            virtual = 0

        flux = pyblaw.flux.LFFlux(self.f['f'], self.f['alpha'], virtual, bc)

        pyblaw.solver.Solver.__init__(self,
                                      system=system,
                                      reconstructor=reconstructor,
                                      flux=flux,
                                      evolver=evolver,
                                      dumper=dumper,
                                      **kwargs)


    ####################################################################
    # cache related
    #

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
        weno.cache(self.cache)

        self.grid = grid
        self.weno = weno
