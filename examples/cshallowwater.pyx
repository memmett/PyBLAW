"""Shallow-water flux and source."""

import numpy as np
cimport numpy as np
cimport cython

# shallow water flux function
cdef void flux(double q[4], double f[4]):

    if q[0] > 0.0:
        f[0] = q[1]
        f[1] = q[1]*q[1]/q[0] + 0.5 * q[0]*q[0]
    else:
        f[0] = 0.0
        f[1] = 0.0

    #f[2] = 0.0
    #f[3] = 0.0

# shallow water numerical flux that well-balances still water
cdef void swflux(double qm[4], double qp[4], double f[4]):
    cdef double fm[4]
    cdef double fp[4]
    cdef unsigned int j

    flux(qm, fm)
    flux(qp, fp)

    f[0] = 0.5 * ( fm[0] + fp[0] - 2.0 * ( (qp[0] + qp[2]) - (qm[0] + qm[2]) ) )
    f[1] = 0.5 * ( fm[1] + fp[1] - 2.0 * ( qp[1] - qm[1] ) )

    #f[2] = 0.0
    #f[3] = 0.0

# shallow water net numerical flux
@cython.boundscheck(False)
def f(np.ndarray[np.double_t, ndim=2] qm,
      np.ndarray[np.double_t, ndim=2] qp,
      np.ndarray[np.double_t, ndim=1] dx,
      np.ndarray[np.double_t, ndim=2] flux):

    # components are: 0: height, 1: momentum

    cdef unsigned int N
    cdef unsigned int i, j
    cdef double fl[4], fr[4]

    N = qm.shape[0]

    swflux(&qm[8,0], &qp[8,0], fr)
    for i in range(8, N-8):
        for j in range(4):
            fl[j] = fr[j]

        swflux(&qm[i+1,0], &qp[i+1,0], fr)

        for j in range(2):
            flux[i,j] = ( fl[j] - fr[j] ) / dx[i]

        flux[i,2] = 0.0
        flux[i,3] = 0.0


# shallow water source that well-balances still water
cdef double w3[3]
w3[0] = 0.5*5.0/9.0
w3[1] = 0.5*8.0/9.0
w3[2] = 0.5*5.0/9.0

@cython.boundscheck(False)
def s(np.ndarray[np.double_t, ndim=2] qm,
      np.ndarray[np.double_t, ndim=2] qp,
      np.ndarray[np.double_t, ndim=3] qq,
      np.ndarray[np.double_t, ndim=1] dx,
      np.ndarray[np.double_t, ndim=2] source):

    # qm, qp components are: 0: height, 1: momentum, 2: bed, 3: bed^2
    # qq components are: 0: height, 1: bed, 2: d/dx bed, 3: d/dx bed^2

    cdef int N = qq.shape[0]
    cdef unsigned int i
    cdef unsigned int j

    cdef double s1, s2, t1, t2
    cdef double st1q[3], st2q[3]

    for i in range(8, N-8):
        source[i,0] = 0.0
        source[i,2] = 0.0
        source[i,3] = 0.0

        s1 = - (qp[i,0] + qp[i,2]) - (qm[i+1,0] + qm[i+1,2])
        s2 = 1.0

        t1 = 0.5 * ( qm[i+1,2] + qp[i+1,2] - qm[i,2] - qp[i,2] )
        t2 = 0.5 * ( qm[i+1,3] + qp[i+1,3] - qm[i,3] - qp[i,3] )

        for j in range(3):
            st1q[j] = ( -(qq[i,j,0] + qq[i,j,1]) - 0.5 * s1 ) * qq[i,j,2]
            st2q[j] = ( 0.5 - 0.5 * s2 ) * qq[i,j,3]

        source[i,1] = 0.5 * ( s1 * t1  + s2 * t2 ) / dx[i]
        for j in range(3):
            source[i,1] += 0.5 * w3[j] * ( st1q[j] + st2q[j] )
