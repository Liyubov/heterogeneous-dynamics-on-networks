# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:15:09 2019

@author: Aurel
"""
"""
import networkx as nx 
import matplotlib.pyplot as plt

fh="C:/Users/Aurel/Documents/PythonFilestoOpen/txtfile2.txt"
plt.subplot(121)
G = nx.read_edgelist(fh)
#print(G.edges(data = True))
#print(G.nodes)
print("Number of node: "+ str(G.number_of_nodes()))
pos = nx.spring_layout(G)
plt.figure(1, figsize=(11, 5))

plt.clf()
#print(G.edges(data = True))
#print(G.nodes)
#nx.draw(pos,node_size=0.6)
nx.draw_networkx(G, pos, node_size=0.6)
#nx.draw(G,node_size=0.6)

#plt.axes()


#plt.show()
plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/infect_dublin/fig1_ia-infect-dublin.png")
#nx.write_gml(G,"C:/Users/Aurel/Documents/PythonFilestoOpen/test3.gml")
"""


"""
OTHIS ONE SEEMS TO WORK
"""

from scipy.io import mmread

import networkx as nx 
import matplotlib.pyplot as plt


A = mmread("C:/Users/Aurel/Documents/PythonFilestoOpen/road-minnesota.mtx")
#A = mmread("C:/Users/Aurel/Documents/PythonFilestoOpen/ia-infect-dublin.mtx")
G = nx.from_scipy_sparse_matrix(A)
print("Number of node: "+ str(G.number_of_nodes()))
pos=nx.spring_layout(G)

nx.draw_networkx(G,pos, with_labels=False, node_size=0.6)
#nx.write_gml(G,"C:/Users/Aurel/Documents/PythonFilestoOpen/rM.gml")
