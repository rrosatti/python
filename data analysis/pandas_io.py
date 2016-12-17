import pandas as pd

# the data used here is from https://www.quandl.com/data/ZILL/Z10128_MLP-Zillow-Home-Value-Index-ZIP-Median-List-Price-New-York
data_path = 'C:/Users/rodri/OneDrive/Documentos/python/data/'
df = pd.read_csv(data_path+'ZILL-Z10128_MLP.csv')
df.set_index('Date', inplace=True)

# saving as a new .csv file
df.to_csv(data_path+'housing_ny.csv')

# reading file specifying the index column
df = pd.read_csv(data_path+'housing_ny.csv', index_col=0)

# rename columns # HPI - House Price Index
df.rename(columns={'Value': 'NY_HPI'}, inplace=True)
df.columns = ['NY_HPI'] 

# convert to other formats (HTML, JSON, Excel, etc)
df.to_html(data_path+'example.html')

print(df.head())