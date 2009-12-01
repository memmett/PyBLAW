"""Plot the first component of the solutions contained in
   'output.mat' for all dump times."""

import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('output.mat')
q   = mat['data.q']
(M, N, p) = q.shape

for i in range(M):
    plt.plot(q[i,:,0], 'o-')

plt.show()
