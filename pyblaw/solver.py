"""PyBLAW generic Solver class.

"""

import os
import numpy as np

import pyblaw.base
import pyblaw.dumper
import pyblaw.evolver

try:
    from mpi4py import MPI as mpi
except:
    pass


######################################################################

class Solver(pyblaw.base.Base):
    """Solver for 1D systems of balance laws.

       Numerically approximates the solution of a system of hyperbolic
       balance laws of the form

         :math:`q_t + f(q)_x = s`.

       The PyBLAW System, Flux, Source, and Evolver classes should be
       extended to define the system, compute the flux, compute the
       source terms, and evolve the system.

       The constructor takes care of connecting the PyBLAW classes
       that you have extended (as mentioned in the previous paragraph)
       and calling their *allocate* and *pre_run* methods.

       **Keyword arguments**

       * *grid*           - pyblaw.grid.Grid
       * *system*         - pyblaw.system.System
       * *reconstructor*  - pyblaw.system.Reconstructor
       * *flux*           - pyblaw.flux.Flux
       * *source*         - pyblaw.source.Source
       * *evolver*        - pyblaw.evolver.Evolver
       * *dumper*         - pyblaw.dumper.Dumper
       * *dump_times*     - dump times
       * *times*          - times

       **Instance variables**

       * *t*             - times
       * *dt*            - time steps
       * *t_dump*        - dump times

       **Instance variables pulled from elsewhere**

       * *N*             - number of cells
       * *x*             - cell boundaries
       * *dx*            - cell sizes
       * *p*             - number of unknowns

       **Methods**

    """

    t      = []                         # times
    dt     = []                         # time steps
    t_dump = []                         # dump times

    grid    = None                      # pyblaw.grid.Grid
    system  = None                      # pyblaw.system.System
    flux    = None                      # pyblaw.flux.Flux
    source  = None                      # pyblaw.source.Source
    evolver = None                      # pyblaw.evolver.Evolver
    dumper  = None                      # pyblaw.dumper.Dumper


    def __init__(self,
                 grid=None, system=None, reconstructor=None, flux=None, evolver=None, source=None,
                 dumper=None, dump_times=None,
                 times=[],
                 **kwargs):

        # init times
        self.t  = times
        self.dt = times[1:] - times[:-1]

        if dump_times is None:
            self.t_dump = np.linspace(self.t[0], self.t[-1], 10+1)
        else:
            self.t_dump = dump_times.copy()

        # are we running through MPI?
        try:
            self.comm = mpi.COMM_WORLD
            self.workers = comm.Get_size() - 1
            self.rank = comm.Get_rank()
            self.mpi = True

        except:
            self.workers = 1
            self.rank = 0
            self.mpi = False


        # init self
        self.grid           = grid
        self.system         = system
        self.reconstructor  = reconstructor
        self.flux           = flux
        self.evolver        = evolver
        self.source         = source
        self.dumper         = dumper

        self.initialised = False


    ####################################################################
    # init
    #

    def initialise_and_allocate(self):
        """Initialise the solver.

           Call all allocate methods, set the initial condtions
           (defined by *system.initial_conditions*), call pre-run
           hooks.

        """

        self.N  = self.grid.size
        self.x  = self.grid.boundaries()
        self.dx = self.grid.sizes()
        self.p  = self.system.p

        #
        # initialise ``slices`` and ``gathers``:
        #
        # * ``slices`` - dictionary (keyed by the rank of the worker) of index
        #                arrays that describe which slice of the solution
        #                (initial condition) is distributed to each worker.
        #                this includes the ghost cells.
        #
        # * ``gathers`` - list of lengths of the above slices (excluding the
        #                 ghost cells) that will be used to gather the
        #                 solutions.  the first entry corresponds to the
        #                 marshaller, which doesn't send anything during the
        #                 gather operation.
        #

        N = self.N
        workers = self.workers
        ghosts  = self.reconstructor.ghosts

        slices  = {}
        gathers = [ 0 ]

        if workers == 1:
            slices[1] = range(N)
            gathers.append(N)

        else:
            slices[1] = range(0, N/workers+ghosts)
            gathers.append(N/workers)

            for i in range(2,workers):
                slices[i] = range((i-1)*(N/workers)-ghosts, i*(N/workers)+ghosts)
                gathers.append(N/workers)

            slices[workers] = range(N-(N/workers + ghosts + N%workers), N)
            gathers.append(N/workers + N%workers)

        self.slices = slices
        self.gathers = gathers
        self.ghosts = ghosts

        #
        # initialise ``right``, ``left``, ``right_worker`` and ``left_worker``:
        #
        # * ``right`` - index of the right-most non-ghosts cell in the worker's
        #               solution.
        #
        # * ``left`` - index of the left-most non-ghosts cell in the worker's
        #              solution.
        #

        rank = self.rank

        right = N
        left  = 0

        right_worker = mpi.PROC_NULL
        left_worker  = mpi.PROC_NULL

        if rank != 0 and workers > 1:
            self.M = M = len(slices[rank])

            if rank == 1:
                right = M - ghosts
                left  = 0

                right_worker = rank + 1

            elif rank == workers:
                right = M
                left  = ghosts

                left_worker = rank - 1

            else:
                right = M - ghosts
                left  = ghosts

                right_worker = rank + 1
                left_worker  = rank - 1
        else:
            self.M = N

        self.left = left
        self.right = right
        self.left_worker = left_worker
        self.right_worker = right_worker


        # link everything up
        self.system.set_grid(self.grid)
        self.system.set_solver(self)
        self.system.M = self.M
        self.reconstructor.set_grid(self.grid)
        self.reconstructor.set_system(self.system)
        self.reconstructor.set_solver(self)
        self.reconstructor.M = self.M
        self.flux.set_grid(self.grid)
        self.flux.set_system(self.system)
        self.flux.set_reconstructor(self.reconstructor)
        self.flux.set_solver(self)
        self.flux.M = self.M
        if self.source is not None:
            self.source.set_grid(self.grid)
            self.source.set_system(self.system)
            self.source.set_reconstructor(self.reconstructor)
            self.source.set_solver(self)
            self.source.M = self.M
        self.evolver.set_grid(self.grid)
        self.evolver.set_system(self.system)
        self.evolver.set_reconstructor(self.reconstructor)
        self.evolver.set_flux(self.flux)
        self.evolver.set_source(self.source)
        self.evolver.set_solver(self)
        self.evolver.set_times(self.t)
        self.evolver.M = self.M
        self.dumper.set_dims(self.grid.centers(), self.t_dump)
        self.dumper.set_system(self.system)

        # allocate
        self.system.allocate()
        self.reconstructor.allocate()
        self.flux.allocate()
        if self.source is not None:
            self.source.allocate()
        self.evolver.allocate()
        self.allocate()

        self.q  = np.zeros((self.M, self.p))
        self.qn = np.zeros((self.M, self.p))

        # apply initial conditions and distribute
        if self.rank == 0:

            # XXX: support loading initial conditions from file...
            self.system.initial_conditions(self.t[0], self.q)

            if self.mpi:
                for i in range(1,self.workers+1):
                    self.comm.Send(self.q[self.slices[i]], dest=i)

        elif self.mpi:
            self.comm.Recv(self.q, source=0)

        # run pre-run hooks
        pre_run_args = {'t0': self.t[0], 'q0': self.q}
        self.system.pre_run(**pre_run_args)
        self.reconstructor.pre_run(**pre_run_args)
        self.flux.pre_run(**pre_run_args)
        if self.source is not None:
            self.source.pre_run(**pre_run_args)
        self.evolver.pre_run(**pre_run_args)
        self.pre_run(**pre_run_args)

        # init the dumper
        if self.rank == 0:
            self.dumper.init_dump()

        # done
        self.initialised = True


    ####################################################################
    # cache related
    #

    def load_cache(self, **kwargs):
        """Load grid etc from a cache.

           This method must set self.grid at least.
        """
        raise NotImplementedError, 'load_cache not implemented'


    def build_cache(self, **kwargs):
        """Pre-compute grid etc and cache.

           This method must set self.grid at least.
        """
        raise NotImplementedError, 'build_cache not implemented'


    ####################################################################
    # run
    #

    def run(self, **kwargs):
        """Run the solver.

           If we are in debugging mode then:

           1. If the trace level is non-zero, XXX
           2. If the trace level is positive, XXX

           The keyword arguments ``kwargs`` are passed on to the
           evolver, reconstructor, flux, and source methods.

        """

        #### allocate and init

        if not self.initialised:
            self.initialise_and_allocate()

        q = self.q
        qn = self.qn

        if kwargs is None:
            kwargs = {}

        #### giv'r!
        for n, t in enumerate(self.t[0:-1]):

            # debug: time step header
            if __debug__ and self.rank == 0:
                if abs(self.trace) > 0:
                    if abs(self.trace) > 1:
                        print "="*69

                    print "n = %d, t = %11.5f, mass = %11.5f" % (n, t, self.system.mass(q))

            # dump solution if necessary
            if self.rank == 0:
                if t >= self.t_dump[0] and self.rank == 0:
                    print "data dump at t = %11.2f, mass = %11.5f" % (t, self.system.mass(q))

                    if self.mpi:
                        if self.rank != 0:
                            recv = None
                            send = q[self.left:self.right]
                        else:
                            recv = q
                            send = None

                        self.comm.Gatherv(sendbuf=[send, mpi.DOUBLE],
                                          recvbuf=[recv, (self.gathers, None), mpi.DOUBLE],
                                          root=0)

                    self.dumper.dump(q)
                    self.t_dump = self.t_dump[1:]

            # evolve
            kwargs.update({'n': n, 't': t})

            if self.source is not None:
                self.evolver.evolve(q, qn, **kwargs)
            else:
                self.evolver.evolve_homogeneous(q, qn, **kwargs)
            q[:,:] = qn[:,:]

            if self.mpi:
                # send non-ghost cells to right_worker's left ghost cells
                self.comm.Sendrecv(q[self.right-self.ghosts:self.right], self.right_worker, 0,
                                   q[:self.left],                        self.left_worker,  0)

                # send non-ghost cells to left_worker's right ghost cells
                self.comm.Sendrecv(q[self.left:self.left+self.ghosts],   self.left_worker,  0,
                                   q[self.right:],            self.right_worker, 0)


            # debug: break?
            if __debug__:
                if self.trace > 0 and n == self.trace:
                    raise ValueError, 'trace stop'


        # last dump if necessary
        if len(self.t_dump) > 0:
            print "data dump at t = %11.2f, mass = %11.5f" % (self.t[-1], self.system.mass(q))
            self.dumper.dump(q)

