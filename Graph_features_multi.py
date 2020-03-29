#!/usr/bin/env python
# coding: utf-8

# In[1]:


from multiprocessing import Process


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import math


# In[3]:


import igraph
import networkx as nx


# In[4]:


from tqdm import tqdm
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing


# In[5]:


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


# In[6]:


nodes_train_set = [element[0] for element in training_set] + [element[1] for element in training_set]
nodes_test_set = [element[0] for element in testing_set] + [element[1] for element in testing_set]


# In[7]:


subset = True

if subset:
    training_set = [training_set[i] for i in [1,2,3,4]]
    testing_set = [testing_set[i] for i in [1,2,3,4]]


# In[8]:


def st_jaccard_similarity(source, target, g):
    nr = len(set(g.neighbors(source)).intersection(set(g.neighbors(target))))
    dr = float(len(set(g.neighbors(source)).union(set(g.neighbors(target)))))
    if dr==0:
        ans=0
    else:
        ans = nr/dr
    return ans


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


# In[9]:


g, g_und = dir_undir_graph_creation(training_set)


# In[10]:


def process_training_1():
    start = 0
    end = int(len(training_set)/2)
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []

    for i in range(len(training_set[start:end])):
        step = 'train_set 1 : ' + str(i+1) + '/' + str(len(training_set[start:end]))
        print(step)
        source = training_set[i][0]
        target = training_set[i][1]
        l_source_target.append((source, target))

        l_st_in_out_degree_product.append(g.degree(source) * g.degree(target))
        l_st_common_neighbors.append(len(set(g.neighbors(source)).intersection(g.neighbors(target))))
        l_st_jaccard_similarity.append(st_jaccard_similarity(source, target, g))
        l_st_count_nodes_in_paths.append(st_count_nodes_in_paths(source, target, g))
        
        if training_set[i][2] == "1":
            g.delete_edges((source, target))
        l_st_shortest_paths_dijkstra.append(min(100000, g.shortest_paths_dijkstra(source, target)[0][0]))
        l_st_shortest_paths_dijkstra_und.append(min(100000, g_und.shortest_paths_dijkstra(source, target)[0][0]))

        if training_set[i][2] == "1":
            g.add_edge(source, target)

    feat_training = np.array([
        l_st_in_out_degree_product,
        l_st_common_neighbors,
        l_st_jaccard_similarity,
        l_st_count_nodes_in_paths,
        l_st_shortest_paths_dijkstra,
        l_st_shortest_paths_dijkstra_und
    ]).T

    training_labels = np.array([int(element[2]) for element in training_set])
    np.savetxt('training_graph_features_0_307756.txt', feat_training)
    np.savetxt("training_labels_0_307756.txt", training_labels)
    writer = csv.writer(open("training_nodes_0_307756.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))
    
    return 'process 1 done'


# In[11]:


def process_training_2():
    start = 0
    end = int(len(training_set)/2)
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []

    for i in range(len(training_set[start:end])):
        step = 'train_set 2 : ' + str(i+1) + '/' + str(len(training_set[start:end]))
        print(step)
        source = training_set[i][0]
        target = training_set[i][1]
        l_source_target.append((source, target))

        l_st_in_out_degree_product.append(g.degree(source) * g.degree(target))
        l_st_common_neighbors.append(len(set(g.neighbors(source)).intersection(g.neighbors(target))))
        l_st_jaccard_similarity.append(st_jaccard_similarity(source, target, g))
        l_st_count_nodes_in_paths.append(st_count_nodes_in_paths(source, target, g))
        
        if training_set[i][2] == "1":
            g.delete_edges((source, target))
            g_und.delete_edges((source, target))
        l_st_shortest_paths_dijkstra.append(min(100000, g.shortest_paths_dijkstra(source, target)[0][0]))
        l_st_shortest_paths_dijkstra_und.append(min(100000, g_und.shortest_paths_dijkstra(source, target)[0][0]))

        if training_set[i][2] == "1":
            g.add_edge(source, target)
            g_und.add_edge(source, target)

    feat_training = np.array([
        l_st_in_out_degree_product,
        l_st_common_neighbors,
        l_st_jaccard_similarity,
        l_st_count_nodes_in_paths,
        l_st_shortest_paths_dijkstra,
        l_st_shortest_paths_dijkstra_und
    ]).T

    training_labels = np.array([int(element[2]) for element in training_set])
    np.savetxt('training_graph_features_307756_615512.txt', feat_training)
    np.savetxt("training_labels_307756_615512.txt", training_labels)
    writer = csv.writer(open("training_nodes_307756_615512.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))
            
    return 'process 2 done'


# In[12]:


def process_testing():
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []

    for i in range(len(testing_set)):
        step = 'test set : ' + str(i+1) + '/' + str(len(testing_set))
        print(step)
        source = testing_set[i][0]
        target = testing_set[i][1]

        l_source_target.append((source, target))

        l_st_in_out_degree_product.append(g.degree(source) * g.degree(target))
        l_st_common_neighbors.append(len(set(g.neighbors(source)).intersection(g.neighbors(target))))
        l_st_jaccard_similarity.append(st_jaccard_similarity(source, target, g))
        l_st_count_nodes_in_paths.append(st_count_nodes_in_paths(source, target, g))
        

        l_st_shortest_paths_dijkstra.append(g.shortest_paths_dijkstra(source, target)[0][0])
        l_st_shortest_paths_dijkstra_und.append(g_und.shortest_paths_dijkstra(source, target)[0][0])

    feat_testing = np.array([
        l_st_in_out_degree_product,
        l_st_common_neighbors,
        l_st_jaccard_similarity,
        l_st_count_nodes_in_paths,
        l_st_shortest_paths_dijkstra,
        l_st_shortest_paths_dijkstra_und
    ]).T

    np.savetxt('testing_graph_features_all.txt', feat_testing)
    writer = csv.writer(open("testing_nodes.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))

    return 'process 3 done'


# In[13]:


if __name__ == '__main__':
    processes = []
    for f in [process_training_1, process_training_2, process_testing]:
        p = Process(target=f)
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


# In[ ]:




