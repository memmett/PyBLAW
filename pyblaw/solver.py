"""PyBLAW generic Solver class.

"""

import numpy as np

import pyblaw.base
import pyblaw.dumper
import pyblaw.evolver


class Solver(pyblaw.base.Base):
    """Solver for 1D systems of balance laws.

       Numerically approximates the solution of a system of hyperbolic
       balances laws of the form

         :math:`q_t + f(q)_x = s`.

       The PyBLAW System, Flux, Source, and Evolver classes should be
       extended to define the system, compute the flux, compute the
       source terms, and evolve the system.

       The constructor takes care of connecting the PyBLAW classes
       that you have extended (as mentioned in the previous paragraph)
       and calling their *allocate* methods.

       Keyword arguments:

         * *grid*           - pyblaw.grid.Grid
         * *system*         - pyblaw.system.System
         * *reconstructor*  - pyblaw.system.Reconstructor
         * *flux*           - pyblaw.flux.Flux
         * *source*         - pyblaw.source.Source
         * *evolver*        - pyblaw.evolver.Evolver
         * *dumper*         - pyblaw.dumper.Dumper
         * *dump_times*     - dump times
         * *times*          - times

       Instance variables:

         * *t*             - times
         * *dt*            - time steps
         * *t_dump*        - dump times
         * *trace*         - trace level

       Instance variables pulled from elsewhere:

         * *N*             - number of cells
         * *x*             - cell boundaries
         * *dx*            - cell sizes
         * *p*             - number of unknowns

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

        self.t  = times
        self.dt = times[1:] - times[:-1]

        self.x  = grid.boundaries()
        self.dx = grid.sizes()
        self.N  = grid.size
        self.p  = system.p

        if dump_times is None:
            self.t_dump = np.linspace(self.t[0], self.t[-1], 10+1)
        else:
            self.t_dump = dump_times.copy()

        self.grid           = grid
        self.system         = system
        self.reconstructor  = reconstructor
        self.flux           = flux
        self.evolver        = evolver
        self.source         = source
        self.dumper         = dumper

        self.system.set_grid(self.grid)
        self.reconstructor.set_grid(self.grid)
        self.reconstructor.set_system(self.system)
        self.flux.set_grid(self.grid)
        self.flux.set_system(self.system)
        self.flux.set_reconstructor(self.reconstructor)
        self.source.set_grid(self.grid)
        self.source.set_system(self.system)
        self.source.set_reconstructor(self.reconstructor)
        self.evolver.set_grid(self.grid)
        self.evolver.set_system(self.system)
        self.evolver.set_reconstructor(self.reconstructor)
        self.evolver.set_flux(self.flux)
        self.evolver.set_source(self.source)
        self.evolver.set_times(self.t)
        self.dumper.set_dims(self.grid.centers(), self.t_dump)
        self.dumper.set_system(self.system)

        self.system.allocate()
        self.reconstructor.allocate()
        self.flux.allocate()
        self.source.allocate()
        self.evolver.allocate()
        self.allocate()


    ####################################################################
    # run
    #

    def run(self):
        """Run the solver.

           This method runs the solver.  First, it allocates the cell
           averages q and sets the initial condtions defined by
           *system.initial_conditions*.  Next, it calls the pre run
           hooks *pre_run* (last minute initialisations).  Finally, it
           cycles throuth the *times* and evolves the system using the
           *evolver*, dumping the cell averages at *dump_times* using
           the *dumper*.

        """

        #### allocate and init

        N = self.grid.N
        p = self.system.p
        q  = np.zeros((N, p))
        qn = np.zeros((N, p))

        self.dumper.init_dump()

        t0 = self.t[0]
        self.system.initial_conditions(t0, q)

        self.grid.pre_run(t0=t0, q0=q)
        self.system.pre_run(t0=t0, q0=q)
        self.reconstructor.pre_run(t0=t0, q0=q)
        self.flux.pre_run(t0=t0, q0=q)
        self.source.pre_run(t0=t0, q0=q)
        self.evolver.pre_run(t0=t0, q0=q)
        self.pre_run(t0=t0, q0=q)

        #### giv'r!

        for n, t in enumerate(self.t[0:-1]):

            # dump solution if necessary

            if t >= self.t_dump[0]:
                print "data dump at t = %11.2f, mass = %11.5f" % (t, self.system.mass(q))
                self.dumper.dump(q)
                self.t_dump = self.t_dump[1:]

            # evolve
            if __debug__:
                if abs(self.trace) > 0:
                    if abs(self.trace) > 1:
                        print "="*150

                    print "n = %d, t = %11.5f, mass = %11.5f" % (n, t, self.system.mass(q))

            self.evolver.evolve(q, n, qn)
            q[:,:] = qn[:,:]

            if __debug__:
                if self.trace > 0 and n == self.trace:
                    raise ValueError, 'trace stop'

        #### done

        if len(self.t_dump) > 0:
            print "data dump at t = %11.2f, mass = %11.5f" % (self.t[-1], self.system.mass(q))
            self.dumper.dump(q)

