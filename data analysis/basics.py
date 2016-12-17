import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')

web_stats = {'Day': [1,2,3,4,5,6], 'Visitors': [43,53,34,45,64,34], 'Bounce_Rate': [65,72,62,64,54,66]}

df = pd.DataFrame(web_stats)

# set index. Inplace makes sure the changes will be applied to df itself rather than returning a new data frame
df.set_index('Day', inplace=True)
print(df)

# printing one column
print(df['Visitors'])
print(df.Visitors)

# printing more than one column
print(df[['Visitors', 'Bounce_Rate']])

# converting to a list
print(df.Visitors.tolist())

# converting multiple columns as a list
print(np.array(df[['Visitors', 'Bounce_Rate']]))

# from numpy array to data frame
df2 = pd.DataFrame(np.array(df[['Visitors', 'Bounce_Rate']]))
print(df2)