from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

m = Basemap(projection='mill',
			llcrnrlat = 25,
			llcrnrlon = -130,
			urcrnrlat = 50,
			urcrnrlon = -60,
			resolution = 'l')

m.drawcoastlines()
m.drawcountries()
m.drawstates()

xs = []
ys = []

NYClat, NYClon = 40.7127, -74.0059
LAlat, LAlon = 34.05, -118.25

# convert latitude and longitude
xpt, ypt = m(NYClon, NYClat)
xpt2, ypt2 = m(LAlon, LAlat)

xs.append(xpt)
xs.append(xpt2)
ys.append(ypt)
ys.append(ypt2)

# c* = cyan/star
m.plot(xpt, ypt, 'c*', markersize=15)
# c^ = blue/triangle
m.plot(xpt2, ypt2, 'b^', markersize=15)

# (line between two points)
m.plot(xs, ys, color='r', linewidth=3, label='Flight 98')
m.drawgreatcircle(NYClon, NYClat, LAlon, LAlat, color='y', linewidth=3, label='Arc')

plt.legend(loc=4)
plt.title('Plotting coords')
plt.show()

