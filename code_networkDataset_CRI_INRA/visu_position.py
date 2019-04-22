# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:13:43 2019

@author: Aurel
"""

import networkx as nx 
import matplotlib.pyplot as plt

plt.subplots_adjust(left  = 0.1,right = 1.5, bottom = 0.1,wspace=0.5)
G = nx.Graph()
G.add_nodes_from([0],id="zero",time=(1))
G.add_nodes_from([1],id="one",time=(1,2))
G.add_nodes_from([2],id="two",time=(2))
G.add_nodes_from([3],id="three",time=(2))

plt.subplot(121)

pos = {0: (40, 20), 1: (20, 30), 2: (40, 30), 3: (30, 10)}

##############
##########

"""
the position is also taken as an attribute
"""
print(pos)

for i in pos:
    G.nodes[i]['pos']=pos.get(i)
   
#nx.set_node_attributes(G, 'test', pos)
print("GRAPH WITH POS AS ATTRIBUTES")
print(G.nodes.data())
print(G.node[0]['pos'][0])

######################
######################

#nx.draw_networkx(G, pos,with_labels=True) 
print(G.nodes.data())
A= nx.get_node_attributes(G,'time')

    #if A[i]==1:
print (A[1])
print (A[1])
G.add_edges_from([(0, 1), (1, 3),(2,3)])
nx.draw_networkx(G,pos)
plt.title("the whole network, all times together")
plt.xlabel("x")
plt.ylabel("y")

#plt.show()
#SG=G.subgraph( [n for n,attrdict in G.node.items() if attrdict 
#['time'] == 1 ] )
    
#SG=G.edge_subgraph((u,v) for u,v,d in G.edges[u,v][d] if (G.edges[u, v]['time']==1))

SG=G.subgraph(n for n,attrdict in G.node.items()
    for i in attrdict ['time']
        if attrdict['time'][i]==1)


#SG=G.edge_subgraph((u,v) for u,v,d in G.edges.data('time') if (d < 580))



print("subgraphe:")
print(SG.nodes.data())
plt.subplot(122)

nx.draw_networkx(SG,pos)
plt.title("time=1")
plt.xlabel("x")
plt.ylabel("y")

#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/test/time_attribute.png")

"""
import matplotlib.pyplot as plt

import networkx as nx

#setup labels on network edges and network nodes (using basic examples from networkx)

G=nx.cubical_graph()
pos=nx.spring_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,
                       nodelist=[0,1,2,3],
                       node_color='r',
                       node_size=500,
                   alpha=0.8)
nx.draw_networkx_nodes(G,pos,
                       nodelist=[4,5,6,7],
                       node_color='b',
                       node_size=500,
                   alpha=0.8)

# edges
nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
nx.draw_networkx_edges(G,pos,
                       edgelist=[(0,1),(1,2),(2,3),(3,0)],
                       width=8,alpha=0.5,edge_color='r')
nx.draw_networkx_edges(G,pos,
                       edgelist=[(4,5),(5,6),(6,7),(7,4)],
                       width=8,alpha=0.5,edge_color='b')


# some math labels
labels={}
labels[0]=r'$a$'
labels[1]=r'$b$'
labels[2]=r'$c$'
labels[3]=r'$d$'
labels[4]=r'$\alpha$'
labels[5]=r'$\beta$'
labels[6]=r'$\gamma$'
labels[7]=r'$\delta$'
nx.draw_networkx_labels(G,pos,labels,font_size=16)

plt.axis('off')
plt.savefig("labels_and_colors.png") # save as png
plt.show() # display
"""