"""Multi-class traffic flow solver.

   This traffic flow solver recreates the results of G.C.K. Wong and
   S.C. Wong in 'A multi-class traffic flow model- an extension of the
   LWR model with heterogeneous drivers', Trans. Res. Part A, 36
   (2002) pp. 827-841.

   Throughout this solver, k[i,m] is the average vehicle density of
   class m in the cell C_i.  There are M vehicle classes and free flow
   speeds u_f[m].  The isotropic speed-density relationship is

   .. math:: U_m (k_1, \ldots , k_M) = u_{f,m} e^{ -0.5 k^2/k_0^2 }

   where k = sum(k_m) and k_0 is some parameter.

   The unit convention used throughout this solver is:

   * space: kilometers (km)
   * time: hours (h)
   * speed: kilometers per hour (km/h)

   In the Wong & Wong paper:

   * There are nine vehicle classes.

   * The parameter k_0 is taken to be 50 veh/km.

   * The spatial grid is: 0 to 2 km, in steps of 5 m; and the temporal
     grid is: 0 to 1.5 h, in steps of 1.5 x 10^{-3} min.

   * The highway is initially empty.

   * Traffic enters the highway (x=0) according to a peak demand
     pattern (see Fig. 4 of Wong & Wong).

   * The exit (x=2km) is blocked from t=1.125h and t=1.175h.

   Here:

   * The spatial grid is: 0 to 2 km, in steps of 50 m; and the temporal
     grid is: 0 to 1.5 h, in steps of ~.72 s.

   * The highway is initially empty except for the first 25m in which
     we place vehicles in each class according to Wong & Wong Fig. 2.
     We this in order to avoid some transient oscillations at the
     entrance of the highway.

   * At the highway exit, the speeds are taken to be the free flow
     speeds (and do not depend on the density at the exit) when the
     exit is not blocked.

"""

import numpy
import math

import pyblaw.flux
import pyblaw.wenoclaw
import pyblaw.system

import pyblaw.version
import pyweno.version

if __debug__:
    import matplotlib.pyplot as plt

import pyblaw.cython
import ctraffic


##
## parameters
##

order = 3                               # reconstruction order
M     = 9                               # number of driver classes

parameters = { 'pyblaw_version': pyblaw.version.version(),
               'pyblaw_git_version': pyblaw.version.git_version(),
               'pyweno_version': pyweno.version.version(),
               'pyweno_git_version': pyweno.version.git_version(),
               }


##
## initial conditions (empty highway, except first 25m)
##

k00 = numpy.zeros(M)
for m in range(M):
    p = 0.2 - 0.04 * ( abs(5-(m+1))%5 ) # fraction of vehicles in
                                        #   class m, see Wong & Wong Fig. 2
    k00[m] = 10.0 * p

def k0(x, t):
    """Initial condition callable (called by PyBLAW).

       In the first 25m, return k00 (defined above).  Otherwise,
       return zeros.

       See pyblaw.system.SimpleSystem for more information.

    """

    if x < 0.025:
        return k00

    return numpy.zeros(M)


##
## the solver (WENO conservation law solver, this is were it all comes
##             together).
##

#times = numpy.linspace(0.0, 1.5, int(1.5*60.0/0.0015)+1) # Wong & Wong time step (1.5e-3 min ~ 0.09 s)
times = numpy.linspace(0.0, 1.5, int(1.5/0.0002)+1)      # our time step (~ 0.72 s)
dump_times = numpy.linspace(0.0, 1.5, 200+1)

solver = pyblaw.wenoclaw.WENOCLAWLFSolver(
    order=order,
    times=times, dump_times=dump_times,
    flux={'flux': ctraffic.flux, 'alpha': 140.0},
    system=pyblaw.system.SimpleSystem(k0, parameters),
    cache='gridk%d.mat' % (order),
    output='traffic.mat'
    )

# build/load grid and cache
if not solver.load_cache():
    #cell_boundaries = numpy.linspace(0.0, 2.0, int(2.0*1000.0/5)+1)  # Wong & Wong cell boundaries (5m)
    cell_boundaries = numpy.linspace(0.0, 2.0, int(2.0*1000.0/50)+1) # our cell boundaries (50m)
    solver.build_cache(cell_boundaries)

# debug reconstructor... (run as 'python -O traffic.py' to disable
# debugging)
def debug(**kwargs):
    t = kwargs['t']
    q  = kwargs['q']
    qm = kwargs['qm']
    qp = kwargs['qp']

    if t > 0.04:
        plt.clf()

        x = solver.grid.centers()
        plt.plot(x, q[:,0], '-k')

        x = solver.grid.boundaries()
        plt.plot(x, qm[:,0], 'or')
        plt.plot(x, qp[:,0], 'ob')

        plt.draw()

        while not plt.waitforbuttonpress():
            pass

solver.reconstructor.debug = debug

##
## giv'r!
##

ctraffic.init()
solver.run()
