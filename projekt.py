import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import rand_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import FeatureAgglomeration
from sklearn.neighbors import KNeighborsClassifier

#data: LABEL|784pixels; x on image: x = i*28 + j,  i,j = [0,27]


#importing data from csv:
file = "fashion-mnist_train.csv"
with open(file) as f:
    lines = (line for line in f if not line.startswith('label')) #importing everything except for the first line
    data = np.loadtxt(lines, delimiter=',') 


#reducing data dimensionality
labels = data[:, 0]
X = data[:,1:]
pca = PCA(n_components=2) #=10
Xnew = pca.fit_transform(X)
x = Xnew [:, 0]
y = Xnew[:, 1]
#case for 2-dimensional reduction
D = pca.explained_variance_ratio_
E = ['1', '2'] #,'3',...,'10'
plt.bar(E,D)
print(D)
plt.show()
#2-dim data
df = pd.DataFrame(dict(x=x, y=y, label=labels))
groups = df.groupby('label')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.x.to_numpy(), group.y.to_numpy(), marker='.', linestyle='', ms=5, label=name)
ax.legend()



#CLUSTERING
#DEFINING FUNCTIONS


#-------------------------------------------------------------------
#too much memory is needed here
#Unable to allocate 13.4 GiB for an array with shape (1799910001,) and data type float64
#clustering = AgglomerativeClustering(compute_distances=True, n_clusters=10, linkage='average').fit(Xnew)
#plt.title("Hierarchical Clustering Dendrogram, reduced dimensions")
# plot the top three levels of the dendrogram
#plot_dendrogram(clustering, labels=clustering.labels_)
#plt.xlabel("Number of points in node (or index of point if no parenthesis).")
#plt.savefig("Hierarchical Agglomerative Clustering with dimensional reduction.png")
#plt.show()

#k-means

labels = data[:, 0]
X = data[:,1:]
#for i in range (1, 10):
pca = PCA(n_components=8)
X8new = pca.fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(X8new)
r = rand_score(labels, kmeans.labels_)
print(f'K-Means Clustering, 8-dimension, random state 0, rand index is: {r}\n')
#MiniBatchKMeans Clustering
miniBatch = MiniBatchKMeans(n_clusters=10, random_state=1).fit(X8new)
r = rand_score(labels, miniBatch.labels_)
print(f'MiniBatch K-Means Clustering, 8-dimension, random state 1, rand index is: {r}\n')

#print(kmeans.cluster_centers_)
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.savefig('2D data with KMeans cluster centres.png')
plt.show()
df_clustered = pd.DataFrame(dict(x=x, y=y, label=kmeans.labels_))
groups_clustered = df_clustered.groupby('label')
fig, ax = plt.subplots()
for name, group in groups_clustered:
    ax.plot(group.x.to_numpy(), group.y.to_numpy(), marker='.', linestyle='', ms=5, label=name)
ax.legend()
plt.savefig('8-dimensional KMeans clustering.png')
plt.show()





df_clustered_MINI = pd.DataFrame(dict(x=x, y=y, label=miniBatch.labels_))
groups_clustered_MINI = df_clustered_MINI.groupby('label')
fig, ax = plt.subplots()
for name, group in groups_clustered_MINI:
    ax.plot(group.x.to_numpy(), group.y.to_numpy(), marker='.', linestyle='', ms=5, label=name)
ax.legend()
plt.savefig('8-dimensional MiniBatch KMeans clustering.png')
plt.show()

bisect_means = BisectingKMeans(n_clusters=10, bisecting_strategy='largest_cluster').fit(X8new)
r = rand_score(labels, bisect_means.labels_)
print(f'BisectingKMeans K-Means Clustering, 8-dimension, random state 0, rand index is: {r}\n')

df_bisect_means = pd.DataFrame(dict(x=x, y=y, label=bisect_means.labels_))
groups_bisect_means = df_bisect_means.groupby('label')
fig, ax = plt.subplots()
for name, group in groups_bisect_means:
    ax.plot(group.x.to_numpy(), group.y.to_numpy(), marker='.', linestyle='', ms=5, label=name)
ax.legend()
plt.savefig('8-dimensional Bisecting KMeans clustering.png')
plt.show()



#Birch clustering

###=========================

brc = Birch(n_clusters=KMeans(n_clusters=10, random_state=0)) #10 -> memory-error
brc.fit(X8new)
r = rand_score(labels, brc.labels_)
print(f'Birch, 8-dimension, rand index is: {r}')

df_BirchClustered = pd.DataFrame(dict(x=x,y=y,label=brc.labels_))
groups_BirchClustered = df_BirchClustered.groupby('label')
fig, ax = plt.subplots()
for name, group in groups_BirchClustered:
    ax.plot(group.x.to_numpy(), group.y.to_numpy(), marker='.', linestyle='', ms=5, label=name)
ax.legend(loc='upper left')
plt.savefig('8-dimensional Birch + KMeans clustering.png')
plt.show()
#too much subclasters
#density-based Clustering: DBSCAN

DBSCANclustering = DBSCAN(eps=0.001).fit(X8new)
r = rand_score(labels, DBSCANclustering.labels_)
print(f'DBSCAN, 8-dimension, rand index is: {r}')

df_DBSCAN = pd.DataFrame(dict(x=x,y=y,label=DBSCANclustering.labels_))
DBSCANGroups = df_DBSCAN.groupby('label')
fig, ax = plt.subplots()
for name, group in DBSCANGroups:
    ax.plot(group.x.to_numpy(), group.y.to_numpy(), marker='.', linestyle='', ms=5, label=name)
ax.legend()
plt.savefig('8-dimensional DBSCAN clustering.png')
plt.show()

#-------takes infinity to process

#density-based Clustering
#opticsClustering = OPTICS().fit(X8new)
#r = rand_score(labels, opticsClustering.labels_)

#df_opticsClustering = pd.DataFrame(dict(x=x, y=y, label=opticsClustering.labels_))
#opticsClustered = df_opticsClustering.groupby('label')
#fig, ax = plt.subplots()
#for name, group in opticsClustered:
#    ax.plot(group.x.to_numpy(), group.y.to_numpy(), marker='.', linestyle='', ms=5, label=name)
#ax.legend()
#plt.show()







#specCluster = SpectralClustering(n_clusters=10).fit(X8new) #Unable to allocate 26.8 GiB for an array with shape (60000, 60000) and data type float64
#r = rand_score(labels, specCluster.labels_)

#splitting the dataset:
#the dataset is already split
file = "fashion-mnist_test.csv"
with open(file) as f:
    lines = (line for line in f if not line.startswith('label'))
    data2 = np.loadtxt(lines, delimiter=',')
labels = data[:, 0]
X = data[:,1:]
labels_test = data2[:,0]
X_test = data2[:,1:]

#classification
#training part
neigh = KNeighborsClassifier().fit(X, labels)
#testing part
accuracy = neigh.score(X_test, labels_test)
prediction = neigh.predict(X_test)
matr = confusion_matrix(labels_test, prediction)
print (accuracy)
print (matr)


pca = PCA(n_components=2)
Xtest_sh = pca.fit_transform(X_test)
x = Xtest_sh [:, 0]
y = Xtest_sh[:, 1]

#plotting predicted data
df_classifying = pd.DataFrame(dict(x=x,y=y,label=prediction))
classifyingGroup = df_classifying.groupby('label')
fig, ax = plt.subplots()
for name, group in classifyingGroup:
    ax.plot(group.x.to_numpy(), group.y.to_numpy(), marker='.', linestyle='', ms=5, label=name)
ax.legend()
plt.savefig('KNeighborsClassifier_test.png')
plt.show()

#plotting true data
df_true = pd.DataFrame(dict(x=x,y=y,label=labels_test))
classifyingGroup = df_true.groupby('label')
fig, ax = plt.subplots()
for name, group in classifyingGroup:
    ax.plot(group.x.to_numpy(), group.y.to_numpy(), marker='.', linestyle='', ms=5, label=name)
ax.legend()
plt.savefig('KNeighborsClassifier_TRUE.png')
plt.show()

