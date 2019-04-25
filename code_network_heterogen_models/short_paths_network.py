import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import pydicom
#import os
#import scipy.ndimage
import matplotlib.pyplot as plt
import networkx as nx

#########################
print('Assign parameter values')
#########################
n = 500; # number of nodes
p=0.5; #parameter value for SF net 
src_vertex = 1; #source node in the network
k = 8 # patameter of WS network
beta = 0.2 #rewiring parameter for WS net


#########################  import matrices 
#A=np.matrix([[1,1],[2,1]])
#A_lond = np.load('Londonmatr.txt)

######################### initialize graph
#G=nx.Graph()
#G=nx.erdos_renyi_graph(n, p, seed=None, directed=False)
#G=nx.scale_free_graph(n, alpha=0.5, beta=0.4, gamma=0.1);#, delta_in=0.2, delta_out=0, create_using=None, seed=None)
G=nx.watts_strogatz_graph(n, k, beta, seed=None)
#G=nx.from_numpy_matrix(A)



#####################
#draw graph with different degrees for nodes 
degree = nx.degree(G)
nx.draw(G, nodelist=degree.keys(), node_size=[v * 1 for v in degree.values()]) # to plot the graph: size of the node is degree of that node
plt.show() #visualize graph G 

######################### using different layouts 
pos=nx.circular_layout(G) 
#pos=nx.spring_layout(G)


######################### nodes from different distance from source
path1 = [];
path2 = [];
path3 = [];
path4 = [];


source_path_lengths = nx.single_source_shortest_path_length(G,src_vertex)
for (v, l) in source_path_lengths.iteritems():
		if l == 1:
			path1.extend([v]);
		elif l ==2:
			path2.extend([v]);
		elif l==3:
			path3.extend([v]);
		else:
			path4.extend([v]);
			
			
			
# draw source node in black	
nx.draw_networkx_nodes(G,pos,nodelist=[src_vertex], node_color='cyan', node_size=100, alpha=0.8)

#path1 = nx.single_source_shortest_path_length(G,source=src_vertex,cutoff=1) #the list of vertices at a distance of <=2 from source
nx.draw_networkx_nodes(G,pos,nodelist=path1, node_color='k', node_size=50, alpha=0.8)
print('path 1')
print path1
	   
#path2 = nx.single_source_shortest_path_length(G,source=src_vertex,cutoff=2) #the list of vertices at a distance of <=2 from source
nx.draw_networkx_nodes(G,pos,nodelist=path2, node_color='b', node_size=50, alpha=0.8)
print('path 2')
print path2

#path3 = nx.single_source_shortest_path_length(G,source=src_vertex,cutoff=3) #the list of vertices at a distance of <=2 from source
nx.draw_networkx_nodes(G,pos,nodelist=path3, node_color='r', node_size=50, alpha=0.8)
print('path 3')
print path3

#path4
nx.draw_networkx_nodes(G,pos,nodelist=path4, node_color='g', node_size=50, alpha=0.8)
print path4
plt.figure(1,figsize=(12,12)) 

# edges
nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
edges = G.edges()
nx.draw_networkx_edges(G,pos,
                       edgelist=edges,
                       width=1,alpha=0.5,edge_color='indigo')
plt.show() 				   







