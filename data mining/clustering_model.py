# This example was taken from https://www.springboard.com/blog/data-mining-python-tutorial/
# 
# With this problem we want to create natural groupings for a set of data objects that might not
# be explicitly stated in the data itself
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster

data_path = 'C:/Users/rodri/OneDrive/Documentos/python/data/'

df = pd.read_csv(data_path+'faithful.csv')

df.columns = ['eruptions', 'waiting']

plt.scatter(df.eruptions, df.waiting)
plt.title('Old Faithful Data Scatterplot')
plt.xlabel('Length of eruption (minutes)')
plt.ylabel('Time between eruptions (minutes)')
plt.show()


# Building the cluster model
# We can easily find two clusters in the plot, but it doesn't label 
# any observation as belonging to either group

# read the dataframe as numpy array in order for sci-kit to be able to read the data
faith = np.array(df)

# number of clusters (in this case is two because it is clearly visible)
k = 2
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(faith)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Creating a visualization

for i in range(k):
	# select only data observations with cluster label == i
	ds = faith[np.where(labels==i)]
	# plot the data observations
	plt.plot(ds[:,0],ds[:,1],'o',markersize=7)
	# plot the centroids
	lines = plt.plot(centroids[i,0], centroids[i,1],'kx')
	# make the centroid x's bigger
	plt.setp(lines, ms=15.0)
	plt.setp(lines, mew=4.0)
plt.show()