import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
from matplotlib import style
import numpy as np
import urllib
import datetime as dt

#style.use('ggplot')
style.use('fivethirtyeight')
#style.use('dark_background')

# fmt - format
def bytespdate2num(fmt, encoding='utf-8'):
	strconverter = mdates.strpdate2num(fmt)
	def bytesconverter(b):
		s = b.decode(encoding)
		return strconverter(s)
	return bytesconverter

def graph_data(stock):

	fig = plt.figure()
	# first tuple is the shape of the grid
	# the second tuple is the start point of this plot
	ax1 = plt.subplot2grid((1,1), (0,0))

	# in order to test it using unix time, change the parameter '10y' to '10d' 
	stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=10y/csv'
	source_code = urllib.request.urlopen(stock_price_url).read().decode()
	stock_data = []
	split_source = source_code.split('\n')

	for line in split_source:
		split_line = line.split(',')
		if len(split_line) == 6:
			if 'values' not in line:
				stock_data.append(line)

	
	date, close_price, high_price, low_price, open_price, volume = np.loadtxt(stock_data, 
																			delimiter=',',
																			unpack=True,
																			converters={0: bytespdate2num('%Y%m%d')})
	
	x = 0
	y = len(date)
	ohlc = []

	while x < y:
		append_me = date[x], open_price[x], high_price[x], low_price[x], close_price[x], volume[x]
		ohlc.append(append_me)
		x+=1

	candlestick_ohlc(ax1, ohlc, width=0.6, colorup='#77d879', colordown='#db3f3f')

	for label in ax1.xaxis.get_ticklabels():
		label.set_rotation(45)

	# converting the date using DateFormatter
	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
	# set a custom locator (x-axis "labels")
	ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
	ax1.grid(True)

	bbox_props = dict(boxstyle='round', fc='w', ec='k', lw=1)

	ax1.annotate(str(close_price[-1]), (date[-1], close_price[-1]), 
						xytext=(date[-1]+4, close_price[-1]), bbox=bbox_props)

	# Adding annotation and text 
	'''
	# adding an annotation
	# axis fraction - the text will stay at the same place 'whenever you go' (because of the fractions set in xytext)
	ax1.annotate('Big News!', (date[11], high_price[11]), xytext=(0.8, 0.9), 
								textcoords='axes fraction', 
								arrowprops=dict(facecolor='grey', color='grey'))


	font_dict = {'family': 'serif', 'color': 'darkred', 'size': 15}
	# first two parameters are the 'coords' where the text will be placed
	ax1.text(date[10], close_price[1], stock+' Prices', fontdict=font_dict	)
	'''

	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title(stock)
	#plt.legend()
	# wspace and hspace correspond to padding between figures
	plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.88, wspace=0.2, hspace=0)
	plt.show()

# TESLA
#graph_data('TSLA')
#graph_data('EBAY')
graph_data('TWTR')