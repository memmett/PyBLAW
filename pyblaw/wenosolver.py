"""WENO conservation law solvers."""

import os
import numpy as np

import pyblaw.flux
import pyblaw.reconstructor
import pyblaw.evolver
import pyblaw.solver
import pyweno.grid
import pyweno.weno


class WENOCLAWLFSolver(pyblaw.solver.Solver):
    """WENO conservation law solver using an Lax-Friedrichs flux.

       XXX
    """

    def __init__(self,
                 f=None, alpha=None,
                 order=3,
                 system=None, evolver=None, dumper=None,
                 times=None,
                 cache='cache.mat', output='output.mat'):

        self.f       = f
        self.alpha   = alpha
        self.k       = order
        self.cache   = cache
        self.output  = output

        if evolver is None:
            evolver = pyblaw.evolver.SSPERK3()

        if dumper is None:
            dumper = pyblaw.dumper.MATDumper(output)

        reconstructor = pyblaw.reconstructor.WENOReconstructor(self.k, 0, self.cache)
        flux          = pyblaw.flux.LFFlux(self.f, self.alpha)

        pyblaw.solver.Solver.__init__(self,
                                      system=system,
                                      reconstructor=reconstructor,
                                      flux=flux,
                                      evolver=evolver,
                                      dumper=dumper,
                                      times=times)


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
