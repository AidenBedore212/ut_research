import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import phate
import scprep

mnist = datasets.load_digits()
keys = mnist.keys()
data = mnist.data
dataShape = mnist.data.shape


#Spliting the data into test and train sets. An 80/20 split
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target,
                                                             test_size=.2, random_state=0)

#-------------------------------------------
#Below are the multiple methods of dimensionality reduction
#-------------------------------------------

#Make an instance of PCA
#def pca_MNIST():
pca = PCA(n_components=2)
pca.fit(train_img)
#Apply the mapping to both sets
pca_train_img = pca.transform(train_img)
pca_test_img = pca.transform(test_img)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
targets = mnist.target
test_targets = test_lbl

def graph_PCA():
    print(pca_train_img.shape)
    fig = plt.figure(figsize=(13, 8))
    labels = ['pca_1', 'pca_2']
    pca_df = pd.DataFrame(pca_train_img, columns=labels)
    plt.scatter(pca_df.pca_1, pca_df.pca_2, c=train_lbl[0:1437], cmap='Set3', s=20)
    plt.title('My PCA graph')
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.set_label('')
    plt.show()

def getAccuracyPCA(n):
    neighbors = KNeighborsClassifier(n_neighbors=n)
    neighbors.fit(pca_train_img, train_lbl)
    score = neighbors.score(pca_test_img, test_targets) * 100
    return score

#TSNE
tsne = TSNE(n_components=2, verbose=0, perplexity=10, n_iter=1000, random_state=42)
tsne.fit_transform(train_img)
tsne_train_img = tsne.fit_transform(train_img)
tsne_test_img = tsne.fit_transform(test_img)

def graph_tnse():
    print(tsne_train_img.shape)
    fig = plt.figure(figsize=(13, 8))
    labels = ['tsne_x', 'tsne_y']
    tsne_df = pd.DataFrame(tsne_train_img, columns=labels)
    plt.scatter(tsne_df.tsne_x, tsne_df.tsne_y, c=train_lbl[0:1437], cmap='Set3', s=20)
    plt.title('My TSNE graph')
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.set_label('')
    plt.show()

def getAccuracyTSNE(n):
    neighbors = KNeighborsClassifier(n_neighbors=n)
    neighbors.fit(tsne_train_img, train_lbl)
    score = neighbors.score(tsne_test_img, test_targets) * 100
    return score


#PHATE
ph8 = phate.PHATE(knn=20)
ph8.fit(train_img)
ph8_train_img = ph8.transform(train_img)
ph8_test_img = ph8.transform(test_img)

def graph_phate():
    print(ph8_train_img.shape)
    fig = plt.figure(figsize=(13, 8))
    labels = ['phate_1', 'phate_2']
    ph8_df = pd.DataFrame(ph8_train_img, columns=labels)
    plt.scatter(ph8_df.phate_1, ph8_df.phate_2, c=train_lbl[0:1437], cmap='Set3', s=20)
    plt.title('My PHATE graph')
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.set_label('')
    plt.show()

#graph_PCA()
#graph_tnse()
#graph_phate()

#print(getAccuracyTSNE(5))
#print(getAccuracyPCA(5))
print("Goodbye World")