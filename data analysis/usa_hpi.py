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

	quandl.ApiConfig.api_key = quandl_key
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


# run it only once in order to 'pickle' the data
#grab_initial_state_data()


HPI_data = get_pickle_data('fiddy_states.pickle')
#print(HPI_data.head())
benchmark = HPI_Benchmark()

# plot data

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))
'''
HPI_data.plot(ax=ax1)
benchmark.plot(ax=ax1, color='k', linewidth=10)

plt.legend().remove()
plt.show()
'''

# correlation
HPI_State_Correlation = HPI_data.corr()
# information of each column (mean, std, min, mean, etc..)
print(HPI_State_Correlation.describe())

# resample data (A - anually)
NY1yr = HPI_data['NY'].resample('A', how='mean')
print(NY1yr.head())

HPI_data['NY'].plot(ax=ax1, label='Monthly NY HPI')
NY1yr.plot(ax=ax1, label='Yearly NY HPI')

# 4 = bottom
plt.legend(loc=4)
plt.show()