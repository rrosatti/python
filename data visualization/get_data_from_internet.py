import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
from matplotlib import style
import numpy as np
import urllib
import datetime as dt

#style.use('ggplot')
#style.use('fivethirtyeight')
style.use('dark_background')

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
																			# %Y = full year 2015
																			# %y = partial year 15
																			# %m = number month
																			# %d = number day
																			# %H = hours
																			# %M = minutes
																			# %S = seconds 
																			# Ex: 12-20-2016
																			# 	  %m-%d-%Y 
																			converters={0: bytespdate2num('%Y%m%d')})
	
	# converting unix time
	'''
	date, close_price, high_price, low_price, open_price, volume = np.loadtxt(stock_data, 
																			delimiter=',',
																			unpack=True)
	dateconv = np.vectorize(dt.datetime.fromtimestamp)
	date = dateconv(date)
	'''

	# customization stuff
	'''
	# '-' = line	
	ax1.plot_date(date, close_price, '-', label='Price')
	
	# fill between 0 and the close price	
	#ax1.fill_between(date, close_price, 0, alpha=0.3, label='Price')
	
	# fill between 12 and the close price
	#ax1.fill_between(date, close_price, 12, alpha=0.3, label='Price')
	
	# fill between teh first value and the close price
	#ax1.fill_between(date, close_price, close_price[0], alpha=0.3, label='Price')

	# using some logic
	ax1.fill_between(date, close_price, close_price[0], where=(close_price > close_price[0]), facecolor='g', alpha=0.3)
	ax1.fill_between(date, close_price, close_price[0], where=(close_price < close_price[0]), facecolor='r', alpha=0.3)


	for label in ax1.xaxis.get_ticklabels():
		# 45 degress of the rotation
		label.set_rotation(45)

	# add a line grid 'behind' the graph
	ax1.grid(True)# color='g', linestyle='-')
	#ax1.xaxis.label.set_color('c')
	#ax1.yaxis.label.set_color('r')
	# set specific numbers for y-axis
	#ax1.set_yticks([0,25,50,75])

	# spines are like the "frame" of the graph
	ax1.spines['left'].set_color('c')
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.spines['left'].set_linewidth(5)

	# change the color of the x-axis (labels)
	ax1.tick_params(axis='x', colors='#10151e')
	'''

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

	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title(stock)
	plt.legend()
	# wspace and hspace correspond to padding between figures
	plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.88, wspace=0.2, hspace=0)
	plt.show()

# TESLA
#graph_data('TSLA')
#graph_data('EBAY')
graph_data('TWTR')