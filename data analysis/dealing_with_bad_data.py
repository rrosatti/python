import quandl, json, pickle, requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from bs4 import BeautifulSoup

style.use('fivethirtyeight')

bridge_height = {'meters': [10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}

df = pd.DataFrame(bridge_height)
# standard deviation is a good way to find 'erroneous' data
df['STD'] = pd.rolling_std(df['meters'], 2)

df_std = df.describe()['meters']['std']
# get only the data where the std is lower than the average std
df = df[ (df['STD'] < df_std) ]

df['meters'].plot()
plt.show()