"""Traffic flow flux."""

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def f(np.ndarray[np.double_t, ndim=2] q, np.ndarray[np.double_t, ndim=2] flux):
    cdef int N = q.shape[0]
    cdef unsigned int i

    for i in range(N):
        flux[i,0] = 0.0
        flux[i,1] = 0.0
        flux[i,2] = 0.0
