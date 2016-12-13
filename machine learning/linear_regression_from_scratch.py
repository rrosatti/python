'''
	y = mx + b
	m = Slope or Gradient (how steep the line is)
	b = the Y intercept (where the line crosses the Y axis)
	
	m =  Change in Y / Change in X	
'''

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# Defining some simple data
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
	m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
		  ((mean(xs)**2) - mean(xs**2)) )

	b = mean(ys) - m*mean(xs)

	return m, b

def predict_y(x):
	return (m*x)+b

m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m*x)+b for x in xs]

p_x = 8
p_y = predict_y(p_x)

plt.scatter(xs, ys)
plt.scatter(p_x, p_y, color='g')
plt.plot(xs, regression_line)
plt.show()


