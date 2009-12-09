"""Plot the first component of the solutions contained in
   'output.mat' for all dump times."""

import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('output.mat')
q = mat['data.q']
x = mat['dims.xdim']

(M, N, p) = q.shape

for i in (0,4,9):
    plt.plot(x, q[i,:,0]+q[i,:,2], 'o-k')
    plt.plot(x, q[i,:,1], 'o-r')
    plt.plot(x, q[i,:,2], 'o-b')

plt.show()
