import quandl, json, pickle, requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from bs4 import BeautifulSoup

style.use('fivethirtyeight')

data_path = 'C:/Users/rodri/OneDrive/Documentos/python/data/'

# get my quandl api key
with open('C:/Users/rodri/OneDrive/Documentos/python/quandl_api.json') as f:
	quandl_key = json.load(f)['key']

def set_quandl_key():
	quandl.ApiConfig.api_key = quandl_key

# get the list of usa states (abbreviation)
def state_list():
	url = 'https://simple.wikipedia.org/wiki/List_of_U.S._states'
	soup = BeautifulSoup(requests.get(url).text, 'html5lib')
	table = soup.find('table', {'class': 'wikitable sortable'})
	table_body = table.find('tbody')
	rows = table_body.find_all('tr')

	data = []
	for row in rows:
		cols = row.find_all('td')
		cols = [e.text.strip() for e in cols]
		data.append([e for e in cols if e]) # get rid of empty elements

	states_abbv = [d[0] for d in data if d] # get only the first columns (abbreviation)

	return states_abbv
	
def save_pickle_data(data, pickle_file):
	with open(data_path+pickle_file, 'wb') as f:
		pickle.dump(data, f)

def get_pickle_data(pickle_file):
	with open(data_path+pickle_file, 'rb') as f:
		data = pickle.load(f)
	return data

# join all the dataframes (states) into one
def grab_initial_state_data():

	states = state_list()
	# initialize an 'empty' data frame that will be used to join with the others data frames
	main_df = pd.DataFrame()

	set_quandl_key()
	# fiddy_states[0][0][1:]
	# [0] - the table we want is the first data frame in the list
	# [0][0] - the column we want is the first one
	# [0][0][1:] - here we are getting rid of the first row ('Abbreviation')
	for abbv in states:
		query = 'FMAC/HPI_'+abbv
		# get the data of each usa state
		df = quandl.get(query)
		df.rename(columns={'Value': abbv}, inplace=True)
		print(abbv)
		# calculate percent change
		df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0

		if main_df.empty:
			main_df = df
		else:
			main_df = main_df.join(df)

	# run it only once
	save_pickle_data(main_df, 'fiddy_states.pickle')

# add new data
def HPI_Benchmark():
	df = quandl.get('FMAC/HPI_USA')
	df.rename(columns={'Value': 'United States'}, inplace=True)
	df['United States'] = (df['United States'] - df['United States'][0]) / df['United States'][0] * 100.0
	return df

def mortgage_30years():
	set_quandl_key()
	df = quandl.get('FMAC/MORTG', trim_start='1975-01-01')
	#df.rename(columns={'Value': 'United States'}, inplace=True)
	df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100.0
	df = df.resample('1D').mean()
	df = df.resample('M').mean()
	df.columns = ['M30']
	return df	

def sp500_data():
	set_quandl_key()
	df = quandl.get('YAHOO/INDEX_GSPC', trim_start='1975-01-01')
	df['Adjusted Close'] = (df['Adjusted Close'] - df['Adjusted Close'][0]) / df['Adjusted Close'][0] * 100.0
	df = df.resample('M').mean()
	df.rename(columns={'Adjusted Close': 'sp500'}, inplace=True)
	df = df['sp500']
	return df

def us_gdp_data():
	set_quandl_key()
	df = quandl.get('BCB/4385', trim_start='1975-01-01')
	df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100.0
	df = df.resample('M').mean()
	df.rename(columns={'Value': 'GDP'}, inplace=True)
	df = df['GDP']
	return df

def us_unemployment():
	df = quandl.get('ECPI/JOB_G', trim_start='1975-01-01')
	df['Unemployment Rate'] = (df['Unemployment Rate'] - df['Unemployment Rate'][0]) / df['Unemployment Rate'][0] * 100.0
	df = df.resample('1D').mean()
	df = df.resample('M').mean()
	return df

'''
# run it only once in order to 'pickle' the data
#grab_initial_state_data()


HPI_data = get_pickle_data('fiddy_states.pickle')
#print(HPI_data.head())
benchmark = HPI_Benchmark()

# plot data

fig = plt.figure()
# subplots on a figure
# 2 tall 1 wide
ax1 = plt.subplot2grid((2,1), (0,0))
ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)

# correlation
#HPI_State_Correlation = HPI_data.corr()
# information of each column (mean, std, min, mean, etc..)
#print(HPI_State_Correlation.describe())

# resample data (A - anually)
#NY1yr = HPI_data['NY'].resample('A', how='mean')
#print(NY1yr.head())

# NY - 12 months/year
# rolling - apply a function in a 'group of time' Ex: 12 months
HPI_data['NY12MA'] = pd.rolling_mean(HPI_data['NY'], 12)
HPI_data['NY12STD'] = pd.rolling_std(HPI_data['NY'], 12)

#HPI_data['NY'].plot(ax=ax1, label='Monthly NY HPI')
#NY1yr.plot(ax=ax1, label='Yearly NY HPI')
HPI_data[['NY', 'NY12MA']].plot(ax=ax1)
HPI_data['NY12STD'].plot(ax=ax2)

# 4 = bottom
plt.legend(loc=4)
plt.show()
'''

# joining year mortgage rate
m30 = mortgage_30years()
HPI_data = get_pickle_data('fiddy_states.pickle')
benchmark = HPI_Benchmark()
sp500 = sp500_data()
US_GDP = us_gdp_data()
US_unemployment = us_unemployment()

HPI = HPI_data.join([benchmark, m30, sp500, US_GDP, US_unemployment])
HPI.dropna(inplace=True)

# HPI correlation
print(HPI.corr())

# run it only once
#save_pickle_data(HPI, 'US_HPI.pickle')