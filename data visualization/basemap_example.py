from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from PIL import Image

m = Basemap(projection='mill',
			llcrnrlat = -90,
			llcrnrlon = -180,
			urcrnrlat = 90,
			urcrnrlon = 180,
			resolution = 'l')

m = Basemap('mill')
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(color='b')
#m.drawcounties(color='darkred')
m.fillcontinents()

#m.etopo()
#m.bluemarble()

plt.title('Basemap Example')
plt.show()
