# https://www.youtube.com/watch?v=afITiFR6vfw&index=19&list=PLQVvvaa0QuDfefDfXb9Yf0la1fPDKluPF
import random
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()

def create_plots():
	xs = []
	ys = []

	for i in range(10):
		x = i
		y = random.randrange(10)

		xs.append(x)
		ys.append(y)

	return xs, ys

# add_subplot syntax
'''
# height, width, plot
#ax1 = fig.add_subplot(211)
#ax2 = fig.add_subplot(212)

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)
'''

# 6x1 (6 tall, 1 wide) | (0,0) - start point
#ax1 = plt.subplot2grid((6,1), (0,0), rowspan=2, colspan=1)
#ax2 = plt.subplot2grid((6,1), (2,0), rowspan=2, colspan=1)
#ax3 = plt.subplot2grid((6,1), (4,0), rowspan=2, colspan=1)

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((6,1), (1,0), rowspan=4, colspan=1)
ax3 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1)


x, y = create_plots()
ax1.plot(x, y)

x, y = create_plots()
ax2.plot(x, y)

x, y = create_plots()
ax3.plot(x, y)

plt.show()