from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

def acp_dim_red(mat, p):    
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    pca = PCA(n_components=p)
    red_mat = pca.fit_transform(mat)
    return red_mat

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
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# perform dimentionality reduction
red_acp_emb = acp_dim_red(embeddings, 20)

# perform clustering ACP
pred_acp_kmeans = clust(red_acp_emb, k)

# evaluate clustering results
nmi_score_acp = normalized_mutual_info_score(pred_acp_kmeans,labels)
ari_score_acp = adjusted_rand_score(pred_acp_kmeans,labels)

print(f'NMI: {nmi_score_acp:.2f} \nARI: {ari_score_acp:.2f}')

# Perform dimentionality reduction
red_emb_umap = dim_red_umap(embeddings, 20)

# Perform Clustering Kmeans
pred_kmeans_umap = clust(red_emb_umap, k)

# evaluate clustering results
nmi_score_umap = normalized_mutual_info_score(pred_kmeans_umap,labels)
ari_score_umap = adjusted_rand_score(pred_kmeans_umap,labels)

print(f'NMI: {nmi_score_umap:.2f} \nARI: {ari_score_umap:.2f}')
