"""PyBLAW WENOCLAW reconstructor and solver classes.

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

       **Arguments**

       * *order*  - WENO reconstruction order
       * *cache*  - cache file name
       * *format* - cache file format (defaults to 'mat')

       The pyweno.weno.WENO object is loaded from the cache, which
       must be pre-built.

       The WENO smoothness indicators and weights are computed for
       each component.

    """

    def __init__(self, order=3, cache='cache.mat', format='mat'):
        self.k = order
        self.cache = cache
        self.format = format


    def pre_run(self, **kwargs):

        self.weno = pyweno.weno.WENO(order=self.k, cache=self.cache, format=self.format)


    def reconstruct(self, q, qm, qp, qq, **kwargs):

        p = q.shape[1]

        for m in range(p):
            self.weno.smoothness(q[:,m])
            self.weno.reconstruct(q[:,m], 'left', qp[:,m], compute_weights=True)
            self.weno.reconstruct(q[:,m], 'right', qm[:,m], compute_weights=True)

        qp[-1,:] = qm[-1,:]
        qm[1:,:] = qm[:-1,:]
        qm[0,:]  = qp[0,:]

        if __debug__:
            self.debug(q=q, qp=qp, qm=qm, qq=qq, **kwargs)


######################################################################

class WENOCLAWLFSolver(pyblaw.solver.Solver):
    """WENO conservation law solver using a Lax-Friedrichs flux.

       **Arguments**

       * *flux*    - flux dictionary (see below)
       * *order*   - WENO reconstruction order
       * *system*  - system
       * *evolver* - evolver or None (defaults to pyblaw.evolver.SSPERK3)
       * *dumper*  - dumper or None (defaults to pyblaw.dumper.MATDumper)
       * *times*   - times
       * *cache*   - cache file name (defaults to 'cache.mat')
       * *format*  - cache file format (defaults to 'mat')
       * *output*  - output file name (defaults to 'output.mat')

       The entries of the *flux* dictionary are:

       * *flux*    - a callable (see pyblaw.flux.LFFlux)
       * *alpha*   - maximum wave speed for the LF flux

    """

    def __init__(self,
                 flux={},
                 order=3,
                 system=None, evolver=None, dumper=None,
                 cache='cache.mat', format='mat',
                 output='output.mat',
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

        flux = pyblaw.flux.LFFlux(self.f['flux'], self.f['alpha'])

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
