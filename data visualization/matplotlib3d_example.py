from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x = [1,2,3,4,5,6,7,8,9,10]
y = [4,8,7,2,6,9,8,1,2,4]
z = [9,4,1,2,7,5,3,4,5,8]

#ax1.plot_wireframe(x, y, z)

x2 = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
y2 = [-4,-8,-7,-2,-6,-9,-8,-1,-2,-4]
z2 = [9,4,1,2,7,5,3,4,5,-8]

#ax1.scatter(x, y, z, c='b', marker='o')
#ax1.scatter(x2, y2, z2, c='r', marker='o')

x3 = [1,2,3,4,5,6,7,8,9,10]
y3 = [4,8,7,2,6,9,8,1,2,4]
z3 = np.zeros(10)
# if use numbers instead of 'zeros', the bars will be 'floating in the air'

dx = np.ones(10)
dy = np.ones(10)
dz = [1,2,3,4,5,6,7,8,9,10]

#ax1.bar3d(x3, y3, z3, dx, dy, dz)

x4, y4, z4 = axes3d.get_test_data()
ax1.plot_wireframe(x4, y4, z4, rstride=5, cstride=5)

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()