import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('legend', fontsize='small')

import matplotlib.pyplot as plt

mat = sio.loadmat('shallow_water.mat')
q = mat['data.q']
x = mat['dims.xdim']

(M, N, p) = q.shape

plt.plot(x, q[0,:,0]+q[0,:,2], '-.k')     # initial height
plt.plot(x, q[0,:,2], '-b')               # bed

plt.plot(x, q[M-2,:,0]+q[M-1,:,2], '-k')  # second last height
plt.plot(x, q[M-2,:,1], '-r')             # second last momentum

plt.title('shallow-water')
plt.xlabel('x')
plt.ylabel('height/momentum')
plt.legend(['initial height', 'bed', 'height', 'momentum'])

plt.savefig('shallow_water.png', format='png')
