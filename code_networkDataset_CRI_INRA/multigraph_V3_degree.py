# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:35:54 2019

@author: Aurel
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


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

G = nx.MultiGraph()

G=nx.from_pandas_edgelist(nt,'Node1','Node2','time',create_using=nx.MultiGraph())
#print(G.edges.data())
print("Number of node: "+ str(G.number_of_nodes()))
print("Number of edges: "+ str(G.number_of_edges()))


plt.figure(num=None, figsize=(40, 30), dpi=80)

fig = plt.figure(1)


pos =nx.spring_layout(G)
#nx.draw_networkx(G,pos,with_labels=False,node_size=0.6, width=0.2)

#plt.savefig("contact_H/time_noattribute.png")
"""
degree
"""
d=[]
d=nx.degree(G)
dico=dict(d)

print(dico.values()) 

#print(G.degree(weight=None))
nx.draw_networkx(G,pos,with_labels=False,node_size=[v for v in dico.values()], width=0.4)

#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/agg_deg.png")






"""
to have a list with the time, for each time, creat a new subgraph
"""

tes=nt['time']
#print(tes)
li=[];

prev=-1
for i in tes:
   # print(i)
   if i != prev:
       li.append(i)
       prev=i

 
for x in li:
    #print(x)
   # SG=G.edge_subgraph((u, v, keys) for (u, v, keys, time) in G.edges(data='time', keys=True)if time==x)
   # print (SG.edges.data())
   # nx.draw(SG,pos,with_labels=True,node_size=0.6, width=0.2)
   # plt.savefig("ima2/im"+str(x)+".png")
    #plt.clf()#this line can be uncommented if we want to separate the pictures here they added onto each other at each loop 
    pass  
###those lines need to be commented if one wants the code to run on the whole dataset
   # if x==6980: 
    #   break
    

plt.figure(num=None, figsize=(40, 30), dpi=80)

fig = plt.figure(2) 
n_step=len(li)
print(n_step)
#dic_mean=dict(d)

#print(dic_mean.values()) 
for x, y in dico.items():
    y=y/n_step
    dico[x]=y
print (dico.values())
#print(G.degree(weight=None))
nx.draw_networkx(G,pos,with_labels=False,node_size=[v for v in dico.values()], width=0.4)

#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/time_deg.png")