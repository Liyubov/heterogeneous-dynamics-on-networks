# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:21:20 2019

@author: Aurel
"""


import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx
#from networkx.drawing.nx_agraph import to_agraph 
#import pygraphviz

import collections


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
#nx.write_gml(G,"C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/hos.gml")
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

#print(dico.values()) 
#print(G.nodes(data='ID'))
colors=[]
for n,i in G.nodes(data='ID'):
    #print(i)
   # print(G.nodes(data='ID'))
    if i =='MED':
        colors.append('g')
       # print("G")
    elif i =='PAT':
        colors.append('r')
    else :
        colors.append('b')
        

#colors = [mapping[G.node[n]['ID']] for n in nodes]]
#print(G.degree(weight=None))
#nx.draw_networkx(G,pos,with_labels=False,node_color=colors,node_size=[v for v in dico.values()], width=0.4)

#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/direcagg_deg_col.png")

#plt.title("agregate") 
#nx.draw_networkx(G,pos,with_labels=False,node_size=[v for v in dico.values()], width=0.4,arrowstyle='-')
#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/direcagg_deg.png")



"""
HISTOGRAM
"""
#print(G.degree())
plt.figure(num=None, figsize=(40, 30), dpi=80)
fig=plt.figure(1)
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
#print(degree_sequence)


degreeCount = collections.Counter(degree_sequence)
#print(degreeCount)

deg, cnt = zip(*degreeCount.items())

#print(degreeCount.items())

#plt.bar(deg, cnt, width=15, color='b')
plt.loglog(deg, cnt)

plt.title("Degree Histogram",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.xlabel("Degree",fontsize=20)
plt.tick_params(labelsize=20)
#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//bar/dh.png")
plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//log/dh.png")

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
    

#plt.figure(num=None, figsize=(40, 30), dpi=80)

#fig = plt.figure(2) 
n_step=len(li)
#print(n_step)
#dic_mean=dict(d)

#print(dic_mean.values()) 
for x, y in dico.items():
    y=y/n_step
    dico[x]=y
#print (dico.values())
#print(G.degree(weight=None))
#nx.draw_networkx(G,pos,with_labels=False,node_size=[v for v in dico.values()], width=0.4)

#plt.title("time") 
#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/directime_deg_col.png")



"""
HISTOGRAM TEMPORAL
"""
plt.figure(num=None, figsize=(40, 30), dpi=80)

fig=plt.figure(2)
#print(G.degree())
no_deg=dico.items()
#print(dico.items())
degree_sequence = sorted([d for n, d in no_deg], reverse=True)
#print(degree_sequence)


degreeCount = collections.Counter(degree_sequence)
#print(degreeCount)

deg, cnt = zip(*degreeCount.items())

#print(degreeCount.items())

#plt.bar(deg, cnt, width=0.002, color='b')
plt.loglog(deg, cnt)

#plt.title("Degree Histogram Temporal Network",fontsize=20)
plt.title("Degree Temporal Network",fontsize=20)

plt.ylabel("Count",fontsize=20)
plt.xlabel("Degree",fontsize=20)
plt.tick_params(labelsize=20)
#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//bar/th.png")
plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//log/th.png")
"""
Indegree
"""

#plt.figure(num=None, figsize=(40, 30), dpi=80)

#fig = plt.figure(3) 

ind=G.in_degree()
dicIndegree=dict(ind)

#print (dicIndegree.values())
#plt.title("agregate indegree") 
#nx.draw_networkx(G,pos,with_labels=False,node_size=[v for v in dicIndegree.values()], width=0.4)
#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/direcagg_indeg_col.png")



#plt.figure(num=None, figsize=(40, 30), dpi=80)

#fig = plt.figure(4) 
n_step=len(li)
#print(n_step)
#dic_mean=dict(d)

#print(dic_mean.values()) 
for x, y in dicIndegree.items():
    y=y/n_step
    dicIndegree[x]=y
#print (dico.values())
#print(G.degree(weight=None))
#nx.draw_networkx(G,pos,with_labels=False,node_size=[v for v in dicIndegree.values()], width=0.4)

#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/directemp_indeg.png")



"""
HISTOGRAM INDEGREE
"""
#print(G.degree())
plt.figure(num=None, figsize=(40, 30), dpi=80)

fig = plt.figure(3) 
degree_sequence = sorted([d for n, d in G.in_degree()], reverse=True)
#print(degree_sequence)


degreeCount = collections.Counter(degree_sequence)
#print(degreeCount)

deg, cnt = zip(*degreeCount.items())

#print(degreeCount.items())

#plt.bar(deg, cnt, width=10, color='b')
plt.plot(deg, cnt,'bo')

#plt.loglog(deg, cnt)
#plt.title("In_Degree Histogram",fontsize=20)
plt.title("In_Degree ",fontsize=20)

plt.ylabel("Count",fontsize=20)
plt.xlabel("In Degree",fontsize=20)
plt.tick_params(labelsize=20)

#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//bar/ih.png")
plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//log/ih.png")

"""
temporal indegreeHisto
"""
plt.figure(num=None, figsize=(40, 30), dpi=80)

fig = plt.figure(4) 

no_indeg=dicIndegree.items()
degree_sequence = sorted([d for n, d in no_indeg], reverse=True)
#print(degree_sequence)


degreeCount = collections.Counter(degree_sequence)
#print(degreeCount)

deg, cnt = zip(*degreeCount.items())

#print(degreeCount.items())

#plt.bar(deg, cnt, width=0.001, color='b')
plt.loglog(deg, cnt)

#plt.title("In_Degree Histogram Temp",fontsize=20)
plt.title("In_Degree Temp",fontsize=20)

plt.ylabel("Count",fontsize=20)
plt.xlabel("In Degree",fontsize=20)
plt.tick_params(labelsize=20)
#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//bar/ith.png")
plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//log/ith.png")

"""
OUTdegree
"""

plt.figure(num=None, figsize=(40,30), dpi=80)

fig = plt.figure(5)
 
outd=G.out_degree(G)
dicOutdegree=dict(outd)
#print(dicOutdegree)
#plt.title("agregate outdegree") 
#nx.draw_networkx(G,pos,with_labels=False,node_size=[v for v in dicOutdegree.values()], width=0.4)
#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/direcagg_outdeg_col.png")



"""
HISTOGRAM
"""
#print(G.degree())

degree_sequence = sorted([d for n, d in G.out_degree()], reverse=True)
#print(degree_sequence)


degreeCount = collections.Counter(degree_sequence)
#print(degreeCount)

deg, cnt = zip(*degreeCount.items())

#print(degreeCount.items())

#plt.bar(deg, cnt, width=5, color='b')
plt.loglog(deg, cnt)

#plt.title("Out_Degree Histogram",fontsize=20)
plt.title("Out_Degree",fontsize=20)

plt.ylabel("Count",fontsize=20)
plt.xlabel("Degree",fontsize=20)
plt.tick_params(labelsize=20)

#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//bar/oh.png")
plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//log/oh.png")


"""
OUTDEGREE TEMP
"""

#plt.figure(num=None, figsize=(40, 30), dpi=80)

#fig = plt.figure(6) 
n_step=len(li)
#print(n_step)
#dic_mean=dict(d)

#print(dic_mean.values()) 
for x, y in dicOutdegree.items():
    y=y/n_step
    dicOutdegree[x]=y
#print (dico.values())
#print(G.degree(weight=None))
#nx.draw_networkx(G,pos,with_labels=False,node_size=[v for v in dicOutdegree.values()], width=0.4)

#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/directemp_outgdeg.png")




"""
HISTOGRAM
"""
plt.figure(num=None, figsize=(40, 30), dpi=80)

#fig, ax = plt.subplots()
fig = plt.figure(6) 

#print(G.degree())
out_temp=dicOutdegree.items()
degree_sequence = sorted([d for n, d in out_temp], reverse=True)
#print(degree_sequence)


degreeCount = collections.Counter(degree_sequence)
#print(degreeCount)

deg, cnt = zip(*degreeCount.items())

#print(degreeCount.items())

#plt.bar(deg, cnt, width=0.001, color='b')
plt.loglog(deg, cnt)

#plt.title("Out Degree Histogram Temp",fontsize=20)
plt.title("Out Degree Temp",fontsize=20)

plt.ylabel("Count",fontsize=20)
plt.xlabel("Degree",fontsize=20)
plt.tick_params(labelsize=20)
#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//bar/oth.png")
plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H//log/oth.png")