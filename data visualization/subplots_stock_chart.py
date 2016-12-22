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

MA1 = 10
MA2 = 30

def moving_average(values, window):
	weights = np.repeat(1.0, window)/window
	smas = np.convolve(values, weights, 'valid')
	return smas

def high_minus_low(highs, lows):
	return highs-lows

# fmt - format
def bytespdate2num(fmt, encoding='utf-8'):
	strconverter = mdates.strpdate2num(fmt)
	def bytesconverter(b):
		s = b.decode(encoding)
		return strconverter(s)
	return bytesconverter

def graph_data(stock):

	fig = plt.figure(facecolor='#f0f0f0')
	# first tuple is the shape of the grid
	# the second tuple is the start point of this plot
	ax1 = plt.subplot2grid((6,1), (0,0), rowspan=1, colspan=1)
	plt.title(stock)
	plt.ylabel('H-L')
	ax2 = plt.subplot2grid((6,1), (1,0), rowspan=4, colspan=1, sharex=ax1)
	plt.ylabel('Price')
	ax2v = ax2.twinx()
	ax3 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
	plt.ylabel('M-AVGs')

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

	ma1 = moving_average(close_price, MA1)
	ma2 = moving_average(close_price, MA2)
	start = len(date[MA2-1:])

	h_1 = list(map(high_minus_low, high_price, low_price))

	ax1.plot_date(date[-start:], h_1[-start:], '-', label='H-L')
	# nbins = it will dictate how many labels we have
	ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='lower'))

	candlestick_ohlc(ax2, ohlc[-start:], width=0.6, colorup='#77d879', colordown='#db3f3f')

	ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune='upper'))
	ax2.grid(True)

	bbox_props = dict(boxstyle='round', fc='w', ec='k', lw=1)

	ax2.annotate(str(close_price[-1]), (date[-1], close_price[-1]), 
						xytext=(date[-1]+4, close_price[-1]), bbox=bbox_props)

	# 'False data', because we can put a label when using 'fill_between'
	ax2v.plot([], [], color='#0079a3', alpha=0.4, label='Volume')
	ax2v.fill_between(date[-start:], 0, volume[-start:], facecolor='#0079a3', alpha=0.4)
	ax2v.axes.yaxis.set_ticklabels([])
	ax2v.grid(False)
	# set the limits of this yaxis
	ax2v.set_ylim(0, 3*volume.max())

	ax3.plot(date[-start:], ma1[-start:], linewidth=1, label=(str(MA1) + 'MA'))
	ax3.plot(date[-start:], ma2[-start:], linewidth=1, label=(str(MA2) + 'MA'))
	ax3.fill_between(date[-start:], ma2[-start:], ma1[-start:], 
						where=(ma1[-start:] < ma2[-start:]), 
						facecolor='r', edgecolor='r', alpha=0.5)

	ax3.fill_between(date[-start:], ma2[-start:], ma1[-start:], 
						where=(ma1[-start:] > ma2[-start:]), 
						facecolor='g', edgecolor='g', alpha=0.5)

	# converting the date using DateFormatter
	ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
	# set a custom locator (x-axis "labels")
	ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))
	ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='upper'))
	
	for label in ax3.xaxis.get_ticklabels():
		label.set_rotation(45)

	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.setp(ax2.get_xticklabels(), visible=False)
	# wspace and hspace correspond to padding between figures
	plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.88, wspace=0.2, hspace=0)
	
	ax1.legend()
	leg = ax1.legend(loc=9, ncol=2, prop={'size':11})
	leg.get_frame().set_alpha(0.4)
	
	ax2v.legend()
	leg = ax2v.legend(loc=9, ncol=2, prop={'size':11})
	leg.get_frame().set_alpha(0.4)
	
	ax3.legend()
	leg = ax3.legend(loc=9, ncol=2, prop={'size':11})
	leg.get_frame().set_alpha(0.4)

	plt.show()
	fig.savefig('C:/Users/rodri/OneDrive/Documentos/python/data/stockchart.png', facecolor=fig.get_facecolor())

# TESLA
#graph_data('TSLA')
#graph_data('EBAY')
#graph_data('TWTR')
graph_data('GOOG')