import pandas as pd
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn import cluster
from sklearn.decomposition import PCA

from imp import reload

"""
PLAN: To understand features better, run pca, clustering algorithm, starting with k-means,
and see where to go from there to more sophisticated stuff.
"""

def plot_kmeans(kmeans, features_reduced):
	""" Function mostly from 
	 http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html """

	# Plots decision boundary
	offset = 0.5
	x_min, x_max = features_reduced[:, 0].min() - offset, features_reduced[:, 0].max() + offset
	y_min, y_max = features_reduced[:, 1].min() - offset, features_reduced[:, 1].max() + offset
	mesh_size = 0.02
	xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_size), np.arange(y_min, y_max, mesh_size))

	# Obtain labels for each point in mesh.
	Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	fig1 = plt.figure()
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
	           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	           cmap=plt.cm.Paired,
	           aspect='auto', origin='lower')

	plt.plot(features_reduced[:, 0], features_reduced[:, 1], 'k.', markersize=2)
	# Mark centroids
	centroids = kmeans.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
	            marker='x', s=169, linewidths=4,
	            color='r', zorder=10)
	plt.title('Clusters')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	fig1.show() 


def basic_test():
	""" Basic clustering analysis. """ 

	# Pick one of 'nov_00', 'dec_00', 'jan_01', 'feb_01', 'all', or 'all_periodic'
	data_period= 'nov_00'
	print('Data set:', data_period)

	features_file = 'data/' + 'features_array_' + data_period + '.npy'
	print('Loading features:', features_file)
	features=np.load(features_file)

	# Remove 'Age', 'Residence Area', and 'Visits data', fact that it's discrete doesn't play well with k-means.
	# TODO: Find an algorithm that deals with this better.
	features = features[:, 3:]
	features = np.log(features) # TODO: deal with outliers

	features_mean = np.mean(features, axis=0)
	features_std = np.var(features, axis=0)

	features_normalised = (features - features_mean)/features_std

	# pca
	pca = PCA(n_components=2)
	pca.fit(features_normalised)

	features_reduced = pca.transform(features_normalised)

	print('Percentage of variance explained by components:\n', pca.explained_variance_ratio_)

	# k-means - TODO: improve, not instructive as things stand.
	kmeans = cluster.KMeans(n_clusters=3)
	kmeans.fit(features_reduced)

	plot_kmeans(kmeans, features_reduced)

if __name__ == "__main__":
    basic_test()


