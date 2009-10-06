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


######################################################################

class FE(Evolver):
    """Forward-Euler evolver."""

    def allocate(self):

        M = self.reconstructor.M
        N = self.grid.N
        p = self.system.p

        self.f  = np.zeros((N,p))
        self.ql = np.zeros((N,p))
        self.qr = np.zeros((N,p))
        self.qq = np.zeros((N*n,p))
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
        self.flux.flux(ql, qr, f)
        self.source.source(qq, s)
        qn[:,:] = q[:,:] + dt * (f[:,:] + s[:,:])


######################################################################

class ERK2(Evolver):
    """Explicit two-stage Runge-Kutta evolver."""

    def allocate(self):

        M = self.reconstructor.M
        N = self.grid.N
        p = self.system.p

        self.f  = np.zeros((N,p))
        self.ql = np.zeros((N,p))
        self.qr = np.zeros((N,p))
        self.qq = np.zeros((N*n,p))
        self.s  = np.zeros((N,p))

        self.qk = np.zeros((N, p))
        self.k1 = np.zeros((N, p))
        self.k2 = np.zeros((N, p))

    def evolve(self, q, n, qn):

        qk = self.qk
        k1 = self.q1
        k2 = self.q2

        t  = self.t[n]
        dt = self.dt[n]

        f  = self.f
        ql = self.ql
        qr = self.qr
        qq = self.qq
        s  = self.s

        # q1
        self.reconstructor.reconstruct(q, ql, qr, qq)
        self.flux.flux(ql, qr, f)
        self.source.source(qq, s)
        k1[:,:] = f[:,:] + s[:,:]

        # k2
        qk[:,:] = q[:,:] + dt * k1[:,:]
        self.reconstructor.reconstruct(qk, ql, qr, qq)
        self.flux.flux(ql, qr, f)
        self.source.source(qq, s)
        k2[:,:] = f[:,:] + s[:,:]

        # qn
        qn[:,:] = q[:,:] + 0.5 * dt * (k1[:,:] + k2[:,:])


######################################################################

class SSPERK3(Evolver):
    """Strong stability-conserving explicit three-stage Runge-Kutta evolver."""

    def allocate(self):

        N = self.grid.N
        p = self.system.p
        n = self.reconstructor.n

        self.f  = np.zeros((N,p))
        self.ql = np.zeros((N,p))
        self.qr = np.zeros((N,p))
        self.qq = np.zeros((N*n,p))
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
        self.flux.flux(ql, qr, f)
        self.source.source(qq, s)
        q1[:,:] = q[:,:] + dt * (f[:,:] + s[:,:])

        # q2
        self.reconstructor.reconstruct(q1, ql, qr, qq)
        self.flux.flux(ql, qr, f)
        self.source.source(qq, s)
        q2[:,:] = 3.0/4.0 * q[:,:] + 1.0/4.0 * q1[:,:] + 1.0/4.0 * dt * (f[:,:] + s[:,:])

        # qn
        self.reconstructor.reconstruct(q2, ql, qr, qq)
        self.flux.flux(ql, qr, f)
        self.source.source(qq, s)
        qn[:,:] = 1.0/3.0 * q[:,:] + 2.0/3.0 * q2[:,:] + 2.0/3.0 * dt * (f[:,:] + s[:,:])






# ######################################################################

# class ERK4(Evolver):
#     """Explicit four-stage Runge-Kutta time-stepper."""

#     # XXX: this is broken

#     def allocate(self):

#         M = self.reconstructor.M
#         N = self.grid.N
#         p = self.system.p

#         self.f  = np.zeros((N,p))
#         self.qs = np.zeros((M,p))
#         self.s  = np.zeros((N,p))

#         self.qk = np.zeros((N, p))
#         self.k1 = np.zeros((N, p))
#         self.k2 = np.zeros((N, p))
#         self.k3 = np.zeros((N, p))
#         self.k4 = np.zeros((N, p))


#     def evolve(self, q, n, qn):

#         qk = self.qk
#         k1 = self.k1
#         k2 = self.k2
#         k3 = self.k3
#         k4 = self.k4

#         t  = self.t[n]
#         dt = self.dt[n]

#         f  = self.f
#         qs = self.qs
#         s  = self.s

#         # k1
#         qk[:,:] = q[:,:]
#         self.reconstructor.reconstruct(qk, qs)
#         self.flux.flux(qs, f)
#         self.source.source(qs, s)
#         k1[:,:] = f[:,:] + s[:,:]

#         # k2
#         qk[:,:] = q[:,:] + 0.5 * dt * k1[:,:]
#         self.reconstructor.reconstruct(qk, qs)
#         self.flux.flux(qs, f)
#         self.source.source(qs, s)
#         k2[:,:] = f[:,:] + s[:,:]

#         # k3
#         qk[:,:] = q[:,:] + 0.5 * dt * k2[:,:]
#         self.reconstructor.reconstruct(qk, qs)
#         self.flux.flux(qs, f)
#         self.source.source(qs, s)
#         k3[:,:] = f[:,:] + s[:,:]

#         # k4
#         qk[:,:] = q[:,:] + dt * k3[:,:]
#         self.reconstructor.reconstruct(qk, qs)
#         self.flux.flux(qs, f)
#         self.source.source(qs, s)
#         k4[:,:] = f[:,:] + s[:,:]

#         # qn
#         qn[:,:] = q[:,:] + 1.0/6.0 * dt * ( k1[:,:] + 2.0 * k2[:,:] + 2.0 * k3[:,:] + k4[:,:] )


# ######################################################################

# class EAM4(Evolver):
#     """Explicit four-step (predictor-corrector) Adams-Moulton time-stepper."""

#     # XXX: this is broken

#     def allocate(self):

#         N = self.grid.N
#         p = self.system.p

#         self.f = np.zeros((N,p))
#         self.s = np.zeros((N,p))

#         self.qp = np.zeros((N,p))
#         self.fn = np.zeros((5,N,p))

#     def evolve(self, q, n, qn):

#         qp = self.qp
#         fn = self.fn
#         f  = self.f

#         t0 = self.t[n]
#         t1 = self.t[n+1]
#         dt = self.dt[n]

#         flux = self.flux.flux

#         if n > 3:

#             flux(q, t0, fn[4,:,:])

#             # predictor
#             qp[:,:] = q[:,:] + dt * ( 1901.0/720.0 * fn[4,:,:] - 1387.0/360.0 * fn[3,:,:]
#                                       + 109.0/30.0 * fn[2,:,:] - 637.0/360.0 * fn[1,:,:] + 251.0/720.0 * fn[0,:,:] )

#             # corrector
#             flux(qp, t1, f)
#             qn[:,:] = q[:,:] + dt * ( 251.0/720.0 * f[:,:] + 646.0/720.0 * fn[3,:,:]
#                                       - 264.0/720.0 * fn[2,:,:] + 106.0/720.0 * fn[1,:,:] - 19.0/720.0 * fn[0,:,:] )

#             # rotate fn
#             fn[0,:,:] = fn[1,:,:]
#             fn[1,:,:] = fn[2,:,:]
#             fn[2,:,:] = fn[3,:,:]
#             fn[3,:,:] = fn[4,:,:]

#         elif n == 3:

#             flux(q, t0, fn[3,:,:])

#             # predictor
#             qp[:,:] = q[:,:] + dt * ( 55.0/24.0 * fn[3,:,:] - 59.0/24.0 * fn[2,:,:] + 37.0/24.0 * fn[1,:,:] - 3.0/8.0 * fn[0,:,:] )

#             # corrector
#             flux(qp, t1, f)
#             qn[:,:] = q[:,:] + dt * ( 3.0/8.0 * f[:,:] + 19.0/24.0 * fn[2,:,:] - 5.0/24.0 * fn[1,:,:] + 1.0/24.0 * fn[0,:,:] )

#         elif n == 2:

#             flux(q, t0, fn[2,:,:])

#             # predictor
#             qp[:,:] = q[:,:] + dt * ( 23.0/12.0 * fn[2,:,:] - 4.0/3.0 * fn[1,:,:] + 5.0/12.0 * fn[0,:,:] )

#             # corrector
#             flux(qp, t1, f)
#             qn[:,:] = q[:,:] + dt * ( 5.0/12.0 * f[:,:] + 2.0/3.0 * fn[1,:,:] - 1.0/12.0 * fn[0,:,:] )

#         elif n == 1:

#             flux(q, t0, fn[1,:,:])

#             # predictor
#             qp[:,:] = q[:,:] + dt * ( 3.0/2.0 * fn[1,:,:] - 1.0/2.0 * fn[0,:,:] )

#             # corrector
#             flux(qp, t1, f)
#             qn[:,:] = q[:,:] + dt * ( 1.0/2.0 * f[:,:] + 1.0/2.0 * fn[0,:,:] )

#         else:

#             flux(q, t0, fn[0,:,:])

#             # predictor
#             qp[:,:] = q[:,:] + dt * fn[0,:,:]

#             # corrector
#             flux(qp, t1, f)
#             qn[:,:] = q[:,:] + dt * f[:,:]
