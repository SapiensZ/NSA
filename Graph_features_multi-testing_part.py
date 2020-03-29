#!/usr/bin/env python
# coding: utf-8

# In[1]:


from multiprocessing import Process


# In[2]:


import pandas as pd
import numpy as np
import random
import csv
import math


# In[3]:


import igraph


# In[4]:


from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing


# In[27]:


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


# In[28]:


nodes_train_set = [element[0] for element in training_set] + [element[1] for element in training_set]
nodes_test_set = [element[0] for element in testing_set] + [element[1] for element in testing_set]


# In[29]:


subset = True
ratio = 0.1
subset_indices = random.sample(range(len(training_set)),
                        k=int(round(len(training_set)*ratio)))

if subset:
    training_set = [training_set[i] for i in subset_indices]
    
step = int(len(training_set)/6)


# In[30]:


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


# In[31]:


g, g_und = dir_undir_graph_creation(training_set)


# In[39]:


def process_training_1():
    start = 0
    end = step
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    j = 0
    for i in range(start,end):
        log = 'train_set 1 : ' + str(j+1) + '/' + str(len(training_set[start:end]))
        j = j+1
        print(log)
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
    np.savetxt('training_graph_features_1.txt', feat_training)
    np.savetxt("training_labels_1.txt", training_labels)
    writer = csv.writer(open("training_nodes_1.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))
    
    return 'process 1 done'


# In[40]:


def process_training_2():
    start = step
    end = step * 2
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    j = 0
    for i in range(start,end):
        log = 'train_set 2 : ' + str(j+1) + '/' + str(len(training_set[start:end]))
        j = j+1
        print(log)
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
    np.savetxt('training_graph_features_2.txt', feat_training)
    np.savetxt("training_labels_2.txt", training_labels)
    writer = csv.writer(open("training_nodes_2.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))
            
    return 'process 2 done'


# In[41]:


def process_training_3():
    start = step * 2
    end = step * 3
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    j = 0
    for i in range(start,end):
        log = 'train_set 3 : ' + str(j+1) + '/' + str(len(training_set[start:end]))
        j = j+1
        print(log)
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
    np.savetxt('training_graph_features_3.txt', feat_training)
    np.savetxt("training_labels_3.txt", training_labels)
    writer = csv.writer(open("training_nodes_3.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))
            
    return 'process 3 done'


# In[42]:


def process_training_4():
    start = step * 3
    end = step * 4
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    j = 0
    for i in range(start,end):
        log = 'train_set 4 : ' + str(j+1) + '/' + str(len(training_set[start:end]))
        j = j+1
        print(log)
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
    np.savetxt('training_graph_features_4.txt', feat_training)
    np.savetxt("training_labels_4.txt", training_labels)
    writer = csv.writer(open("training_nodes_4.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))
            
    return 'process 4 done'


# In[43]:


def process_training_5():
    start = step * 4
    end = step * 5
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    j = 0
    for i in range(start,end):
        log = 'train_set 5 : ' + str(j+1) + '/' + str(len(training_set[start:end]))
        j = j+1
        print(log)
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
    np.savetxt('training_graph_features_5.txt', feat_training)
    np.savetxt("training_labels_5.txt", training_labels)
    writer = csv.writer(open("training_nodes_5.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))
            
    return 'process 5 done'


# In[44]:


def process_training_6():
    start = len(training_set) - step * 5
    end = len(training_set) 
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    
    j = 0
    for i in range(start,end):
        
        log = 'train_set 6 : ' + str(j+1) + '/' + str(len(training_set[start:end]))
        j = j+1
        print(log)
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
    np.savetxt('training_graph_features_6.txt', feat_training)
    np.savetxt("training_labels_6.txt", training_labels)
    writer = csv.writer(open("training_nodes_6.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))
            
    return 'process 6 done'


# In[ ]:





# In[ ]:





# In[45]:


step_test = int(len(testing_set)/6)
step_test


# In[ ]:





# In[48]:


def process_testing_1():
    start = 0
    end = step_test
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    
    j = 0
    for i in range(start,end):
        
        log = 'test_set 1 : ' + str(j+1) + '/' + str(len(testing_set[start:end]))
        j = j+1
        print(log)
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

    np.savetxt('testing_graph_1.txt', feat_testing)
    writer = csv.writer(open("testing_nodes_1.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))

    return 'process testing done'


# In[49]:


def process_testing_2():
    start = step_test
    end = step_test * 2
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    
    j = 0
    for i in range(start,end):
        
        log = 'test_set 2 : ' + str(j+1) + '/' + str(len(testing_set[start:end]))
        j = j+1
        print(log)
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

    np.savetxt('testing_graph_2.txt', feat_testing)
    writer = csv.writer(open("testing_nodes_2.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))

    return 'process testing done'


# In[50]:


def process_testing_3():
    start = step_test * 2
    end = step_test * 3
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    
    j = 0
    for i in range(start,end):
        
        log = 'test_set 3 : ' + str(j+1) + '/' + str(len(testing_set[start:end]))
        j = j+1
        print(log)
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

    np.savetxt('testing_graph_3.txt', feat_testing)
    writer = csv.writer(open("testing_nodes_3.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))

    return 'process testing done'


# In[51]:


def process_testing_4():
    start = step_test * 3
    end = step_test * 4
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    
    j = 0
    for i in range(start,end):
        
        log = 'test_set 4 : ' + str(j+1) + '/' + str(len(testing_set[start:end]))
        j = j+1
        print(log)
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

    np.savetxt('testing_graph_4.txt', feat_testing)
    writer = csv.writer(open("testing_nodes_4.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))

    return 'process testing done'


# In[52]:


def process_testing_5():
    start = step_test * 4
    end = step_test * 5
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    
    j = 0
    for i in range(start,end):
        
        log = 'test_set 5 : ' + str(j+1) + '/' + str(len(testing_set[start:end]))
        j = j+1
        print(log)
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

    np.savetxt('testing_graph_5.txt', feat_testing)
    writer = csv.writer(open("testing_nodes_5.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))

    return 'process testing done'


# In[53]:


def process_testing_6():
    start = step_test * 5
    end = len(testing_set)
    l_source_target = []
    l_st_in_out_degree_product = []
    l_st_common_neighbors = []
    l_st_jaccard_similarity = []
    l_st_count_nodes_in_paths = []
    l_st_shortest_paths_dijkstra = []
    l_st_shortest_paths_dijkstra_und = []
    
    j = 0
    for i in range(start,end):
        
        log = 'test_set 6 : ' + str(j+1) + '/' + str(len(testing_set[start:end]))
        j = j+1
        print(log)
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

    np.savetxt('testing_graph_6.txt', feat_testing)
    writer = csv.writer(open("testing_nod_6.csv","w"))
    for row in l_source_target:
         writer.writerow(str(row))

    return 'process testing done'


# In[17]:


if __name__ == '__main__':
    processes = []
    for f in [process_testing_1, process_testing_2, process_testing_3, process_testing_4, process_testing_5, process_testing_6]:
        p = Process(target=f)
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


# In[ ]:





# In[ ]:




