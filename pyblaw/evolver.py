"""PyBLAW abstract Evolver class and a few concrete evolvers.

"""

import numpy as np

import pyblaw.base


class Evolver(pyblaw.base.Base):
    """Abstract evolver (time-stepper).

       Evolve the cell averages q given at time t^n to time t^{n+1}.

       Instance variables:

       * *t*             - times (time grid)
       * *dt*            - time step sizes

       * *grid*          - pyblaw.grid.Grid
       * *system*        - pyblaw.system.System
       * *reconstructor* - pyblaw.reconstructor.Reconstructor
       * *flux*          - pyblaw.flux.Flux
       * *source*        - pyblaw.grid.Source

       Methods that should be overridden:

       * *allocate* - allocates memory etc
       * *evolve*   - evolve q

    """

    t  = []                             # times
    dt = []                             # time steps

    grid   = None
    system = None
    flux   = None

    def set_grid(self, grid):
        self.grid = grid

    def set_system(self, system):
        self.system = system

    def set_reconstructor(self, reconstructor):
        self.reconstructor = reconstructor

    def set_flux(self, flux):
        self.flux = flux

    def set_source(self, source):
        self.source = source

    def set_times(self, times):
        """Set times at which the solution is computed to *times* and
        compute time step sizes."""
        self.t  = times
        self.dt = self.t[1:] - self.t[:-1]

    def evolve(self, q, qn, **kwargs):
        """Evolve q and store the result in qn."""

        raise NotImplementedError

    def evolve_homogeneous(self, q, qn, **kwargs):
        """Evolve q (no source) and store the result in qn."""

        raise NotImplementedError


######################################################################

class FE(Evolver):
    """Forward-Euler evolver."""

    def allocate(self):

        N = self.grid.N
        p = self.system.p
        n = self.reconstructor.n

        self.f  = np.zeros((N,p))
        self.ql = np.zeros((N+1,p))
        self.qr = np.zeros((N+1,p))
        self.qq = np.zeros((N,n,p))
        self.s  = np.zeros((N,p))


    def evolve(self, q, n, qn):

        t  = self.t[n]
        dt = self.dt[n]

        f  = self.f
        ql = self.ql
        qr = self.qr
        qq = self.qq
        s  = self.s

        self.reconstructor.reconstruct(q, ql, qr, qq)
        self.flux.flux(ql, qr, t, f)
        self.source.source(ql, qr, qq, t, s)
        qn[:,:] = q[:,:] + dt * (f[:,:] + s[:,:])

    def evolve_homogeneous(self, q, n, qn):

        t  = self.t[n]
        dt = self.dt[n]

        f  = self.f
        ql = self.ql
        qr = self.qr
        qq = self.qq
        s  = self.s

        self.reconstructor.reconstruct(q, ql, qr, qq)
        self.flux.flux(ql, qr, t, f)
        qn[:,:] = q[:,:] + dt * f[:,:]


######################################################################

class SSPERK3(Evolver):
    """Strong stability-conserving explicit three-stage Runge-Kutta evolver."""

    def allocate(self):

        N = self.grid.N
        p = self.system.p
        n = self.reconstructor.n

        self.f  = np.zeros((N,p))
        self.ql = np.zeros((N+1,p))
        self.qr = np.zeros((N+1,p))
        self.qq = np.zeros((N,n,p))
        self.s  = np.zeros((N,p))

        self.q1 = np.zeros((N, p))
        self.q2 = np.zeros((N, p))


    def evolve(self, q, n, qn):

        q1 = self.q1
        q2 = self.q2

        t  = self.t[n]
        dt = self.dt[n]

        f  = self.f
        ql = self.ql
        qr = self.qr
        qq = self.qq
        s  = self.s

        # q1
        self.reconstructor.reconstruct(q, ql, qr, qq)
        self.flux.flux(ql, qr, t, f)
        self.source.source(ql, qr, qq, t, s)
        q1[:,:] = q[:,:] + dt * (f[:,:] + s[:,:])

        # q2
        self.reconstructor.reconstruct(q1, ql, qr, qq)
        self.flux.flux(ql, qr, t, f)
        self.source.source(ql, qr, qq, t, s)
        q2[:,:] = 3.0/4.0 * q[:,:] + 1.0/4.0 * q1[:,:] + 1.0/4.0 * dt * (f[:,:] + s[:,:])

        # qn
        self.reconstructor.reconstruct(q2, ql, qr, qq)
        self.flux.flux(ql, qr, t, f)
        self.source.source(ql, qr, qq, t, s)
        qn[:,:] = 1.0/3.0 * q[:,:] + 2.0/3.0 * q2[:,:] + 2.0/3.0 * dt * (f[:,:] + s[:,:])

    def evolve_homogeneous(self, q, n, qn):

        q1 = self.q1
        q2 = self.q2

        t  = self.t[n]
        dt = self.dt[n]

        f  = self.f
        ql = self.ql
        qr = self.qr
        qq = self.qq
        s  = self.s

        # q1
        self.reconstructor.reconstruct(q, ql, qr, qq)
        self.flux.flux(ql, qr, t, f)
        q1[:,:] = q[:,:] + dt * f[:,:]

        # q2
        self.reconstructor.reconstruct(q1, ql, qr, qq)
        self.flux.flux(ql, qr, t, f)
        q2[:,:] = 3.0/4.0 * q[:,:] + 1.0/4.0 * q1[:,:] + 1.0/4.0 * dt * f[:,:]

        # qn
        self.reconstructor.reconstruct(q2, ql, qr, qq)
        self.flux.flux(ql, qr, t, f)
        qn[:,:] = 1.0/3.0 * q[:,:] + 2.0/3.0 * q2[:,:] + 2.0/3.0 * dt * f[:,:]
