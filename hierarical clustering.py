#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#importing the dataset
dataset = pd.read_csv("Mall_customers.csv")
x = dataset.iloc[:, [3,4]].values


# using dendogrames to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method = "ward"))
plt.title("dendogram")
plt.xlabel("customer")
plt.ylabel("euclidean disances")
plt.show()


# fitting the hierarical clustering to the mall customers dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity= "euclidean", linkage= "ward")
y_hc = hc.fit_predict(x)


# Visualising the clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 30, color = "green", label = "carefull")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 30, color = "red", label = "moderate")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 30, color = "cyan", label = "target customers")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 30, color = "yellow", label = "careless")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 30, color = "magenta", label = "sensible")
plt.title("clusters of cliens")
plt.xlabel("anual income ($)")
plt.ylabel("spending score (1-100)")
plt.legend()
plt.show()