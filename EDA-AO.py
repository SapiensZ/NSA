#!/usr/bin/env python
# coding: utf-8

# # Missing link prediction - Kaggle competition

# ### Libraries

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

#%%
import igraph
import networkx as nx


#%%
from tqdm import tqdm
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing

# ### Read data

#%%
with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]



#%%
nodes_train_set = [element[0] for element in training_set] + [element[1] for element in training_set]
nodes_test_set = [element[0] for element in testing_set] + [element[1] for element in testing_set]


# __Subset__

#%%
subset = False
#ratio = 0.001
#subset_indices = random.sample(range(len(training_set)),
                        #k=int(round(len(training_set)*ratio)))
if subset:
    training_set = [training_set[i] for i in [1,2]]
    testing_set = [testing_set[i] for i in [1,2]]



# # Features Engineering on graph 

# ### Functions to compute on graph

#%%
def st_jaccard_similarity(source, target, g):
    nr = len(set(g.neighbors(source)).intersection(set(g.neighbors(target))))
    dr = float(len(set(g.neighbors(source)).union(set(g.neighbors(target)))))
    if dr==0:
        ans=0
    else:
        ans = nr/dr
    return ans


#%%
def st_katz_similarity(source, target, g):
    i = IDs.index(source)
    j = IDs.index(target)
    
    def katz(g):
        katzDict = {}  # build a special dict that is like an adjacency list
        adjList = g.get_adjlist()

        for i, l in enumerate(adjList):
            katzDict[i] = l
        return katzDict
    katzDict = katz(g)
    l = 1
    maxl = 5
    neighbors = katzDict[i]
    score = 0
    beta = 0.005 

    while l <= maxl:
        numberOfPaths = neighbors.count(j)
        if numberOfPaths > 0:
            score += (beta**l)*numberOfPaths

        neighborsForNextLoop = []
        for k in neighbors:
            neighborsForNextLoop += katzDict[k]
        neighbors = neighborsForNextLoop
        l += 1
    return score


#%%
def st_count_nodes_in_paths(source, target, g):
    s=set(g.subcomponent(source, mode="out"))
    t=set(g.subcomponent(target, mode="in"))
    return(len(s.intersection(t)))


# __Definition of a directed and undirected graphs__

#%%
def dir_undir_graph_creation(dataset):
    G_dir = igraph.Graph(directed=True)
    edges = [(element[0], element[1]) for element in dataset if element[2] == "1"]
    nodes = nodes_train_set
    G_dir.add_vertices(nodes)
    G_dir.add_edges(edges)
    G_undir = G_dir.as_undirected()
    return G_dir, G_undir


# __For training_set__

#%%
features = dict()


#%%
l_st_in_degree_product = []
l_st_out_degree_product = []
l_st_in_out_degree_product = []
l_st_common_neighbors = []
l_st_jaccard_similarity = []
l_st_adamic_adar = []
l_st_friendtns = []
l_st_katz_similarity = []
l_st_closeness = []
l_st_closeness = []
l_st_count_nodes_in_paths = []
l_st_betweenness = []
l_st_shortest_paths_dijkstra = []
l_st_shortest_path_dijkstra_und = []
l_st_jacc_und = []

g, g_und = dir_undir_graph_creation(training_set)

for i in tqdm(range(len(training_set))):
    source = training_set[i][0]
    target = training_set[i][1]

    l_st_in_degree_product.append(g.degree(source, mode="in") * g.degree(target, mode="in"))
    l_st_out_degree_product.append(g.degree(source,mode="out") * g.degree(target, mode="out"))
    l_st_in_out_degree_product.append(g.degree(source) * g.degree(target))
    l_st_common_neighbors.append(len(set(g.neighbors(source)).intersection(g.neighbors(target))))
    l_st_jaccard_similarity.append(st_jaccard_similarity(source, target, g))
    l_st_adamic_adar.append(sum([1.0/math.log(g.degree(v)) 
                               for v in set(g.neighbors(source)).intersection(set(g.neighbors(target)))]))
    l_st_friendtns.append(round((1.0/(g.degree(source) + g.degree(target) - 1.0)),3))
    l_st_katz_similarity.append(st_katz_similarity(source, target, g))
    l_st_closeness.append(g.closeness(vertices=source) + g.closeness(vertices=target))
    l_st_count_nodes_in_paths.append(st_count_nodes_in_paths(source, target, g))
    l_st_betweenness.append(g.betweenness(vertices=source) + g.betweenness(vertices=target))
    
    if training_set[i][2] == "1":
        g.delete_edges((source, target))
        g_und.delete_edges((source, target))
    l_st_shortest_paths_dijkstra.append(min(100000, g.shortest_paths_dijkstra(source, target)[0][0]))
    short_path_und = min(100000, g.shortest_paths_dijkstra(source, target)[0][0])
    l_st_shortest_path_dijkstra_und.append(short_path_und)
                                
    if short_path_und > 2:
        jacc = 0
    else:
        jacc = g_und.similarity_jaccard(pairs=[(source, target)])[0]
    l_st_jacc_und.append(jacc)
    
    if training_set[i][2] == "1":
        g.add_edge(source, target)
        g_und.add_edge(source, target)


#%%

feat = np.array([
    l_st_in_degree_product,
    l_st_out_degree_product,
    l_st_in_out_degree_product,
    l_st_common_neighbors,
    l_st_jaccard_similarity,
    l_st_adamic_adar,
    l_st_friendtns,
    l_st_katz_similarity,
    l_st_closeness,
    l_st_count_nodes_in_paths,
    l_st_betweenness,
    l_st_shortest_paths_dijkstra,
    l_st_shortest_path_dijkstra_und,
    l_st_jacc_und
]).T


#%%
features['training'] = feat
m = features['training'].mean(axis=0)
std = features['training'].std(axis=0)
features['training'] = (features['training'] - m) / std


#%%
training_labels = np.array([int(element[2]) for element in training_set])


#%%
np.savetxt('training_graph_features.txt', features['training'])
np.savetxt("training_labels.txt", training_labels)


#%%
l_st_in_degree_product = []
l_st_out_degree_product = []
l_st_in_out_degree_product = []
l_st_common_neighbors = []
l_st_jaccard_similarity = []
l_st_adamic_adar = []
l_st_friendtns = []
l_st_katz_similarity = []
l_st_closeness = []
l_st_closeness = []
l_st_count_nodes_in_paths = []
l_st_betweenness = []
l_st_shortest_paths_dijkstra = []
l_st_shortest_path_dijkstra_und = []
l_st_jacc_und = []

for i in range(len(testing_set)):
    print(i)
    source = testing_set[i][0]
    target = testing_set[i][1]

    l_st_in_degree_product.append(g.degree(source, mode="in") * g.degree(target, mode="in"))
    l_st_out_degree_product.append(g.degree(source,mode="out") * g.degree(target, mode="out"))
    l_st_in_out_degree_product.append(g.degree(source) * g.degree(target))
    l_st_common_neighbors.append(len(set(g.neighbors(source)).intersection(g.neighbors(target))))
    l_st_jaccard_similarity.append(st_jaccard_similarity(source, target, g))
    l_st_adamic_adar.append(sum([1.0/math.log(g.degree(v)) 
                               for v in set(g.neighbors(source)).intersection(set(g.neighbors(target)))]))
    l_st_friendtns.append(round((1.0/(g.degree(source) + g.degree(target) - 1.0)),3))
    l_st_katz_similarity.append(st_katz_similarity(source, target, g))
    l_st_closeness.append(g.closeness(vertices=source) + g.closeness(vertices=target))
    l_st_count_nodes_in_paths.append(st_count_nodes_in_paths(source, target, g))
    l_st_betweenness.append(g.betweenness(vertices=source) + g.betweenness(vertices=target))
    

    l_st_shortest_paths_dijkstra.append(g.shortest_paths_dijkstra(source, target)[0][0])
    short_path_und = min(100000, g.shortest_paths_dijkstra(source, target)[0][0])
    l_st_shortest_path_dijkstra_und.append(short_path_und)
                                
    if short_path_und > 2:
        jacc = 0
    else:
        jacc = g_und.similarity_jaccard(pairs=[(source, target)])[0]
    l_st_jacc_und.append(jacc)

#%%
feat = np.array([
    l_st_in_degree_product,
    l_st_out_degree_product,
    l_st_in_out_degree_product,
    l_st_common_neighbors,
    l_st_jaccard_similarity,
    l_st_adamic_adar,
    l_st_friendtns,
    l_st_katz_similarity,
    l_st_closeness,
    l_st_count_nodes_in_paths,
    l_st_betweenness,
    l_st_shortest_paths_dijkstra,
    l_st_shortest_path_dijkstra_und,
    l_st_jacc_und
]).T

#%%
features['testing'] = feat
features['testing'] = (features['testing'] - m) / std

#%%
np.savetxt('testing_graph_features.txt', features['testing'])



#%%


# %%
