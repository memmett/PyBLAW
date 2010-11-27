"""PyBLAW WENOCLAW reconstructor and solver classes.

"""

import os
import numpy as np

import pyblaw.flux
import pyblaw.source
import pyblaw.reconstructor
import pyblaw.evolver
import pyblaw.solver
import pyblaw.dumper
import pyblaw.h5dumper                  # moving this creates problems with cython
import pyweno.grid
import pyweno.weno


######################################################################

class WENOCLAWReconstructor(pyblaw.reconstructor.Reconstructor):
    """WENO CLAW Reconstructor.

       **Arguments**

       * *order*  - WENO reconstruction order
       * *cache*  - cache file name

       The pyweno.weno.WENO object is loaded from the cache, which
       must be pre-built.

       The WENO smoothness indicators and weights are computed for
       each component.

    """

    def __init__(self, order=3, cache='cache.h5'):
        self.k = order
        self.cache = cache


    def pre_run(self, **kwargs):

        self.weno = pyweno.weno.WENO(order=self.k, cache=self.cache)


    def reconstruct(self, q, qm, qp, qq, **kwargs):

        p = q.shape[1]

        for m in range(p):
            self.weno.smoothness(q[:,m])
            self.weno.reconstruct(q[:,m], 'left', qp[:,m], compute_weights=True)
            self.weno.reconstruct(q[:,m], 'right', qm[:,m], compute_weights=True)

        qp[-1,:] = qm[-2,:]
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
       * *cache*   - cache file name (defaults to 'cache.h5')
       * *output*  - output file name (defaults to 'output.h5')
       * *format*  - output file format

       The entries of the *flux* dictionary are:

       * *flux*    - a callable (see pyblaw.flux.LFFlux)
       * *alpha*   - maximum wave speed for the LF flux

    """

    def __init__(self,
                 flux={},
                 order=3,
                 system=None, evolver=None, dumper=None,
                 cache='cache.h5',
                 output='output.h5', format = 'h5py',
                 **kwargs):

        self.f       = flux
        self.k       = order
        self.cache   = cache
        self.output  = output
        self.format  = format

        if evolver is None:
            evolver = pyblaw.evolver.SSPERK3()

        if dumper is None:
            if self.format == 'mat':
                dumper = pyblaw.dumper.MATDumper(output)
            elif self.format == 'h5py':
                dumper = pyblaw.h5dumper.H5PYDumper(output)

        reconstructor = WENOCLAWReconstructor(order=self.k,
                                              cache=self.cache)

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

        self.grid = pyweno.grid.Grid(cache=self.cache)
        self.weno = pyweno.weno.WENO(order=self.k, cache=self.cache)

        return True

    def build_cache(self, x=None):
        grid = pyweno.grid.Grid(x)

        weno = pyweno.weno.WENO(grid=grid, order=self.k)
        weno.precompute_reconstruction('left')
        weno.precompute_reconstruction('right')
        weno.cache(self.cache)

        self.grid = grid
        self.weno = weno


######################################################################

class PeriodicWENOCLAWReconstructor(pyblaw.reconstructor.Reconstructor):
    """Periodic WENO CLAW Reconstructor.

       **Arguments**

       * *order*  - WENO reconstruction order
       * *cache*  - cache file name

       The pyweno.weno.WENO object is loaded from the cache, which
       must be pre-built.

       The WENO smoothness indicators and weights are computed for
       each component.

    """

    def __init__(self, order=3, cache='cache.h5'):
        self.k = order
        self.cache = cache


    def pre_run(self, **kwargs):

        self.weno = pyweno.weno.WENO(order=self.k, cache=self.cache)
        self.grid = self.weno.grid      # set our grid to the ghost grid

        N = self.grid.N
        p = self.system.p

        self.wqm = np.zeros((N+1, p))
        self.wqp = np.zeros((N+1, p))
        self.wq  = np.zeros((N, p))


    def reconstruct(self, q, qm, qp, qq, **kwargs):

        p = q.shape[1]
        k = self.k
        N = self.grid.N

        wqm = self.wqm
        wqp = self.wqp
        wq  = self.wq

        # copy and set ghost cells
        wq[k:-k,:] = q[:,:]
        wq[:k,:]   = q[-k:,:]
        wq[-k:,:]  = q[:k,:]

         # reconstruct using ghost cells
        for m in range(p):
            self.weno.smoothness(wq[:,m])
            self.weno.reconstruct(wq[:,m], 'left', wqp[:,m], imin=k-1, imax=N-k, compute_weights=True)
            self.weno.reconstruct(wq[:,m], 'right', wqm[:,m], imin=k-1, imax=N-k, compute_weights=True)

        qm[:,:] = wqm[k-1:N-k,:]
        qp[:,:]  = wqp[k:N-k+1,:]


        if __debug__:
            self.debug(q=q, qp=qp, qm=qm, qq=qq, **kwargs)


######################################################################

class PeriodicWENOCLAWLFSolver(pyblaw.solver.Solver):
    """Periodic WENO conservation law solver using a Lax-Friedrichs
       flux.

       **Arguments**

       * *flux*    - flux dictionary (see below)
       * *order*   - WENO reconstruction order
       * *system*  - system
       * *evolver* - evolver or None (defaults to pyblaw.evolver.SSPERK3)
       * *dumper*  - dumper or None (defaults to pyblaw.dumper.MATDumper)
       * *times*   - times
       * *cache*   - cache file name (defaults to 'cache.h5')
       * *output*  - output file name (defaults to 'output.h5')
       * *format*  - cache file format (defaults to 'h5py')

       The entries of the *flux* dictionary are:

       * *flux*    - a callable (see pyblaw.flux.LFFlux)
       * *alpha*   - maximum wave speed for the LF flux

    """

    def __init__(self,
                 flux={},
                 order=3,
                 system=None, evolver=None, dumper=None,
                 cache='cache.h5', format='h5py',
                 output='output.h5',
                 **kwargs):

        self.f       = flux
        self.k       = order
        self.cache   = cache
        self.output  = output
        self.format  = format

        if evolver is None:
            evolver = pyblaw.evolver.SSPERK3()

        if dumper is None:
            if self.format == 'mat':
                dumper = pyblaw.dumper.MATDumper(output)
            elif self.format == 'h5py':
                dumper = pyblaw.h5dumper.H5PYDumper(output)

        reconstructor = PeriodicWENOCLAWReconstructor(order=self.k,
                                                      cache=self.cache)

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

        self.ghost_grid = pyweno.grid.Grid(cache=self.cache)
        self.weno = pyweno.weno.WENO(order=self.k, cache=self.cache)

        # remove ghost cells and create new grid
        k = self.k
        y = self.ghost_grid.x
        x = y[k:-k]
        self.grid = pyblaw.grid.Grid(x)

        return True

    def build_cache(self, x=None):

        k = self.k

        # create ghost cells
        y = np.zeros(x.size + 2*k)
        y[k:-k] = x[:]
        dx = x[1:] - x[:-1]

        for i in range(k+1):
            y[k-i] = y[k-i+1] - dx[-i]
            y[-k+i-1] = y[-k+i-2] + dx[i]

        # create grid for weno (includes ghost cells)
        ghost_grid = pyweno.grid.Grid(y)

        weno = pyweno.weno.WENO(grid=ghost_grid, order=self.k)
        weno.precompute_reconstruction('left')
        weno.precompute_reconstruction('right')
        weno.cache(self.cache)

        self.ghost_grid = ghost_grid
        self.grid = pyblaw.grid.Grid(x)
        self.weno = weno
