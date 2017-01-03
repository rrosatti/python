# This example was taken from https://www.springboard.com/blog/data-mining-python-tutorial/
# 
# With this problem we want to see if we can estimate the relationship between house price and the square footage of the house. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from matplotlib import rcParams
import statsmodels.api as sm
from statsmodels.formula.api import ols

data_path = 'C:/Users/rodri/OneDrive/Documentos/python/data/'

df = pd.read_csv(data_path+'kc_house_data.csv')

# Checking if there is null values in the data frame
#print(df.isnull().any())

# Checking the data types of the data frame
#print(df.dtypes)


# Two important things to do in order to check if the data is reasonable and not corrupted.
# 1 - Use df.describe() to look at all the variables
# 2 - Use plt.pyplot.hist() to plot histograms of the variables
#print(df.describe())

fig = plt.figure(figsize=(12,6))
sqft = fig.add_subplot(121)
cost = fig.add_subplot(122)

sqft.hist(df.sqft_living, bins=80)
sqft.set_xlabel('Ft^2')
sqft.set_title('Histogram of House Square Footage')

cost.hist(df.price, bins=80)
cost.set_xlabel('Price ($)')
cost.set_title('Histogram of Housing Prices')

plt.show()

# What we found after ploting: both variables have a distribution that is right-skewed


# Linear Regression
# Using OLS we can produce a linear regression with only two variables.
# The formula used is that:
#
# Reg = ols('Dependent variable ~ independent variable(s), dataframe).fit()
#
# .summary - show the summary report

m = ols('price ~ sqft_living', df).fit()
print(m.summary())

# The most relevant information in this summary: R-squared, t-statistics, standard error and the coefficients of correlation
# There is an extremely significant  relationship between square footage and housing prices, because there is an extremely
# high t-value of 144.920, and a P>|t| of 0%, which means that this relationship has a near-zero chance of being due to statistical
# variation or chance

# Testing with more than one independent variable
#m2 = ols('price ~ sqft_living + bedrooms + grade + condition', df).fit()
#print(m2.summary())

# Adding a few more independent variables made we went from being able to explain about 49.3% of the variation
# in the model to 55.5%
# In this way we can provide a model that fits the data better

# Using seaborn to visualize the regression summary
# It shows the regression line as well as distribution plots for each variable
sns.jointplot(x='sqft_living', y='price', data=df, kind='reg', fit_reg=True, size=7)
plt.show()