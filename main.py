from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

'''
Variables:
---------

corpus : list of documents
embeddings : documents embeddings of size NxM (N : number of documents, M : embedding dimension)
red_emd : reduced embeddings matrix using dimentionality reduction
k : number of clusters
labels : documents labels
pred : list of clustering predicted clusters

''';

def tsne_red(mat, p):
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

    red_mat = mat[:,:p]
    tsne = TSNE(n_components=2,random_state=42)
    tsne_result = tsne.fit_transform(red_mat)

    return tsne_result

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

# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# perform dimentionality reduction
tsne_emb = tsne_red(embeddings, 20)

# perform clustering
pred_tsne = clust(tsne_emb, k)

# evaluate clustering results
nmi_score_tsne = normalized_mutual_info_score(pred_tsne,labels)
ari_score_tsne = adjusted_rand_score(pred_tsne,labels)

print(f'NMI: {nmi_score_tsne:.2f} \nARI: {ari_score_tsne:.2f}')
