"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import os
os.environ["PROJ_LIB"] = "C:\\Users\\Aurel\Anaconda3\\Library\\share"; #those lines are needed in order to avoid an error, maybe an easiest way to solve it though :/....

from mpl_toolkits.basemap import Basemap as Basemap

print("hello world")
"""
#import os
#os.environ["PROJ_LIB"] ="C:\\users\\aurel\\anaconda3\\lib\\site-packages"
"""

#from networkx.drawing.nx_agraph import to_agraph 
# define the graph as per your question
G=nx.MultiDiGraph([(1,2),(1,1),(1,2),(2,3),(3,4),(2,4), 
    (1,2),(1,2),(1,2),(2,3),(3,4),(2,4)])

# add graphviz layout options (see https://stackoverflow.com/a/39662097)
G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
G.graph['graph'] = {'scale': '3'}
#nx.write_dot(G,'multi.dot')
# adding attributes to edges in multigraphs is more complicated but see
# https://stackoverflow.com/a/26694158                    
G[1][1][0]['color']='red'

A = to_agraph(G) 
A.layout('dot')                                                                 
A.draw('multi.png') 

"""
"""
import networkx as nx
#from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_pydot import write_dot
#from networkx.drawing.nx_agraph import write_dot


G=nx.MultiGraph()
G.add_edge(1,2)
G.add_edge(1,2)

write_dot(G,'multi.dot')
"""

import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx
#from networkx.drawing.nx_agraph import to_agraph 
#import pygraphviz

import collections
import networkx as nx
#from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_pydot import write_dot
#from networkx.drawing.nx_agraph import write_dot


###########OPEN HOPITAL FILE###################
"""
hospital files for sociopattern :
http://www.sociopatterns.org/datasets/hospital-ward-dynamic-contact-network/ 
unzipped and 

put in pandas format with header

The path need to be change depending on where you put the file, same for the plt.savefig

at the end the file is preprocess to make a video: 
    the time is put in a list and at each time a subgraph is created
    at the end there is the code to make the video
    here the file will stop at a certain time because the dataset is toobig, but one just need to delete
    those two lines to make it run through the whole dataset
"""
col_H = ['time', 'Node1', 'Node2', 'ID1','ID2']

nodes = pd.read_csv("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/detailed_list_of_contacts_Hospital.dat_/s55Y6BgCeFB", delim_whitespace =True,names = col_H)

#print(nodes)

nt=nodes[['Node1','Node2','time']]

#print(nt)
ids1=nodes[['Node1','ID1']]
ids2=nodes[['Node2','ID2']]

G = nx.MultiDiGraph()

G=nx.from_pandas_edgelist(nt,'Node1','Node2','time',create_using=nx.MultiDiGraph())
#print(G.edges.data())
print("Number of node: "+ str(G.number_of_nodes()))
print("Number of edges: "+ str(G.number_of_edges()))

"""
add graphviz layout options (see https://stackoverflow.com/a/39662097)
G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}

"""
#G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}

#print("without id")
#print(G.nodes(data=True))
nx.write_gml(G,"C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/hos.gml")
color=[]
for index, row in ids1.iterrows():
    G.add_node(row['Node1'],ID=row['ID1'])
   
for index, row in ids2.iterrows():
    G.add_node(row['Node2'],ID=row['ID2'])
  

#print("ID")
#print(G.nodes(data=True))

#plt.figure(num=None, figsize=(40, 30), dpi=80)

#fig = plt.figure(1)


pos =nx.spring_layout(G)
#nx.draw_networkx(G,pos,with_labels=False,node_size=0.6, width=0.2)

#plt.savefig("contact_H/time_noattribute.png")
"""
degree
"""
d=[]
d=nx.degree(G)
dico=dict(d)

write_dot(G,'multi.dot')


