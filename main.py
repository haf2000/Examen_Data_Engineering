from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import umap
import matplotlib.pyplot as plt

# Define our dimensionality reduction UMAP function
def dim_red_umap(mat, p):
    '''
    Perform dimensionality reduction using UMAP

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    
    umap_model = umap.UMAP(n_components=p)
    red_mat = umap_model.fit_transform(embeddings)    
    return red_mat


# Define the Clustering function
def clust(mat, k):
    '''
    Perform clustering

    Input:
    -----
        mat : input list 
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''
    kmeans = KMeans(k)
    cluster_labels = kmeans.fit_predict(mat)
    return cluster_labels


# Import the Data
ng20 = fetch_20newsgroups(subset='all')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# We need to embed textual data
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)
print(embeddings)

# Perform dimentionality reduction
red_emb_umap = dim_red_umap(embeddings, 20)

# Perform Clustering Kmeans
pred_kmeans_umap = clust(red_emb_umap, k)

# evaluate clustering results
nmi_score_umap = normalized_mutual_info_score(pred_kmeans_umap,labels)
ari_score_umap = adjusted_rand_score(pred_kmeans_umap,labels)

print(f'NMI: {nmi_score_umap:.2f} \nARI: {ari_score_umap:.2f}')