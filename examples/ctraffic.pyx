"""Multi-class traffic flux and boundary condition functions."""

import numpy as np
cimport numpy as np
cimport cython

# use libm's exp function
cdef extern from "math.h":
    double exp(double)

# initialise u_f[m]
cdef double u_f[9]
def init():
    for i in range(9):
        u_f[i] = 60.0 + 60.0/8.0 * i


##
## flux function
##

@cython.boundscheck(False)
def flux(np.ndarray[np.double_t, ndim=2] k,
         np.ndarray[np.double_t, ndim=2] f,
         **kwargs):
    """Multi-class traffic flow flux function (called by PyBLAW).

       This is called by the pyblaw.flux.LFFlux class, which is in
       turn used by the pyblaw.wenoclaw.WENOLFSolver class to compute
       conservative fluxes.
    """
    cdef int N = k.shape[0]
    cdef int M = k.shape[1]
    cdef unsigned int i
    cdef unsigned int m
    cdef double K
    cdef double expK
    cdef double t = kwargs['t']

    ## fluxes

    # cycle through each grid cell
    for i in range(1,N-1):

        # compute total density in this cell
        K = 0.0
        for m in range(M):
            K = K + k[i,m]
        expK = exp( - K * K / 5000.0 )

        # compute flux for each vehicle class
        for m in range(M):
            f[i,m] = u_f[m] * expK * k[i,m]

    ## boundary conditions

    # highway entrance (peak demand pattern, see Fig. 4)
    K = 10.0
    if t < 0.25:
        K = 10.0
    elif t < 0.5:
        K = 10.0 + 40.0/0.25 * (t - 0.25)
    elif t < 1.0:
        K = 50.0
    elif t < 1.25:
        K = 50.0 - 40.0/0.25 * (t - 1.0)

    expK = exp( - K * K / 5000.0 )

    for m in range(M):
        # fraction of vehicles in class m, see Wong & Wong Fig. 2
        if m+1 < 5:
            p = 0.2 - 0.04 * ( 5-(m+1) )
        else:
            p = 0.2 - 0.04 * ( (m+1)-5 )

        f[0,m] = u_f[m] * K * p * expK


    # highway exit (blocked from t=1.125h and t=1.175h, free-flow
    # otherwise)

    if t < 1.125:
        for m in range(M):
            f[N-1,m] = u_f[m] * k[N-1,m]
    elif t <= 1.175:
        for m in range(M):
            f[N-1,m] = 0.0
    else:
        for m in range(M):
            f[N-1,m] = u_f[m] * k[N-1,m]
