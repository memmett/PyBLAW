"""Linear system solver.

   This linear solver serves as short example of how to use the PyBLAW
   framework.

   It solves a system of hyperbolic equations of the form

     q_t + A q_x = B q

   where A and B are constant (and possibly singular) matrices.

"""

import math

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
# initial condition and solver
#

def q0(x, t):
    """Initial condition.

       XXX
    """

    if x < -t or x > t:
        return np.zeros(3)

    return np.array([ math.cos(0.5*math.pi*x/t), 0.0, 0.0 ])


class LinearSolver(pyblaw.solver.Solver):
    """Solver for a linear system.

    """

    def __init__(self, A, B, boundaries, times, order=4):

        p = A.shape[0]

        grid          = pyblaw.grid.Grid(boundaries=boundaries)
        system        = pyblaw.system.SimpleSystem(p, q0, parameters={'A': A, 'B': B})
        reconstructor = pyblaw.reconstructor.WENOReconstructor(order, 3, 'cache.h5')
        flux          = pyblaw.flux.LinearLFFlux(A)
        source        = pyblaw.source.LinearQuad3Source(B)
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

k = 3

A = np.matrix('2.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 -2.0')
B = np.matrix('0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 -0.1 0.1')

x = np.linspace(-400.0, 400.0, 1600+1)
t = np.linspace(100.0,  110.0, 100+1)

solver = LinearSolver(A, B, x, t, order=k)

solver.run()
