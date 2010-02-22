"""PyBLAW abstract Evolver class and a few concrete evolvers.

"""

import numpy as np
import pyblaw.base


######################################################################

class Evolver(pyblaw.base.Base):
    """Abstract evolver (time-stepper).

       Evolve the cell averages q given at time t^n to time t^{n+1}.

       **Instance variables**

       * *t*             - times
       * *dt*            - time step sizes

       * *grid*          - grid
       * *system*        - system
       * *reconstructor* - reconstructor
       * *flux*          - flux
       * *source*        - source

       **Methods that should be overridden**

       * *allocate* - allocates memory etc
       * *evolve*   - evolve q

       **Methods**

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


    def allocate(self):
        """Allocate storage space for reconstruction, flux, and source."""

        N = self.grid.N
        p = self.system.p
        n = self.reconstructor.n

        self.f  = np.zeros((N,p))
        self.ql = np.zeros((N+1,p))
        self.qr = np.zeros((N+1,p))
        self.qq = np.zeros((N,n,p))
        self.s  = np.zeros((N,p))

    def reconstruct_and_compute_flux_and_source(self, q, **kwargs):
        """Helper function to reconstruct and compute the flux and
        source while updating the keyword argument dictionary."""

        f  = self.f
        ql = self.ql
        qr = self.qr
        qq = self.qq
        s  = self.s

        r = self.reconstructor.reconstruct(q, ql, qr, qq, **kwargs)
        if isinstance(r, dict):
            kwargs.update(r)

        r = self.flux.flux(ql, qr, f, **kwargs)
        if isinstance(r, dict):
            kwargs.update(r)

        r = self.source.source(ql, qr, qq, s, **kwargs)
        if isinstance(r, dict):
            kwargs.update(r)

        return kwargs


    def reconstruct_and_compute_flux(self, q, **kwargs):
        """Helper function to reconstruct and compute the flux while
        updating the keyword argument dictionary."""

        f  = self.f
        ql = self.ql
        qr = self.qr
        qq = self.qq

        r = self.reconstructor.reconstruct(q, ql, qr, qq, **kwargs)
        if isinstance(r, dict):
            kwargs.update(r)

        r = self.flux.flux(ql, qr, f, **kwargs)
        if isinstance(r, dict):
            kwargs.update(r)

        return kwargs


######################################################################

class FE(Evolver):
    """Forward-Euler evolver."""

    def evolve(self, q, qn, **kwargs):

        f  = self.f
        s  = self.s
        dt = self.dt[kwargs['n']]

        # qn
        kwargs = self.reconstruct_and_compute_flux_and_source(q, **kwargs)
        qn[:,:] = q[:,:] + dt * (f[:,:] + s[:,:])

        # done
        if __debug__:
            self.debug()

        return kwargs

    def evolve_homogeneous(self, q, qn, **kwargs):

        f  = self.f
        dt = self.dt[kwargs['n']]

        # qn
        kwargs = self.reconstruct_and_compute_flux(q, **kwargs)
        qn[:,:] = q[:,:] + dt * f[:,:]

        # done
        if __debug__:
            self.debug()

        return kwargs


######################################################################

class SSPERK3(Evolver):
    """Strong stability-conserving explicit three-stage Runge-Kutta evolver."""

    def allocate(self):

        Evolver.allocate(self)

        N = self.grid.N
        p = self.system.p

        self.q1 = np.zeros((N, p))
        self.q2 = np.zeros((N, p))


    def evolve(self, q, qn, **kwargs):

        f  = self.f
        s  = self.s
        q1 = self.q1
        q2 = self.q2
        dt = self.dt[kwargs['n']]

        # q1
        kwargs = self.reconstruct_and_compute_flux_and_source(q, **kwargs)
        q1[:,:] = q[:,:] + dt * (f[:,:] + s[:,:])

        # q2
        kwargs = self.reconstruct_and_compute_flux_and_source(q1, **kwargs)
        q2[:,:] = 3.0/4.0 * q[:,:] + 1.0/4.0 * q1[:,:] + 1.0/4.0 * dt * (f[:,:] + s[:,:])

        # qn
        kwargs = self.reconstruct_and_compute_flux_and_source(q2, **kwargs)
        qn[:,:] = 1.0/3.0 * q[:,:] + 2.0/3.0 * q2[:,:] + 2.0/3.0 * dt * (f[:,:] + s[:,:])

        # done
        if __debug__:
            self.debug()

        return kwargs

    def evolve_homogeneous(self, q, qn, **kwargs):

        f  = self.f
        s  = self.s
        q1 = self.q1
        q2 = self.q2
        dt = self.dt[kwargs['n']]

        # q1
        kwargs = self.reconstruct_and_compute_flux(q, **kwargs)
        q1[:,:] = q[:,:] + dt * f[:,:]

        # q2
        kwargs = self.reconstruct_and_compute_flux(q1, **kwargs)
        q2[:,:] = 3.0/4.0 * q[:,:] + 1.0/4.0 * q1[:,:] + 1.0/4.0 * dt * f[:,:]

        # qn
        kwargs = self.reconstruct_and_compute_flux(q2, **kwargs)
        qn[:,:] = 1.0/3.0 * q[:,:] + 2.0/3.0 * q2[:,:] + 2.0/3.0 * dt * f[:,:]

        # done
        if __debug__:
            self.debug()

        return kwargs
