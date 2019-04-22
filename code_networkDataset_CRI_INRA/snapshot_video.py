# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:39:04 2019

@author: Aurel
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:24:52 2019

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

#plt.subplots_adjust(left  = 0.1,right = 1.5, bottom = 0.1,wspace=0.5)
#plt.subplot(121)
pos =nx.spring_layout(G)
#nx.draw_networkx(G,pos,with_labels=False,node_size=0.6, width=0.2)

#plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/time_noattribute.png")


 

SG=G.edge_subgraph((u, v, keys) for (u, v, keys, time) in G.edges(data='time', keys=True)if time ==920)
#print("SUBGRAPH:")
#print(SG.edges.data())

plt.figure(1)
nx.draw(SG,pos,with_labels=True,node_size=1, width=0.2)
#plt.title("time=920")


plt.savefig("images/ima1.png")


SG=G.edge_subgraph((u, v, keys) for (u, v, keys, time) in G.edges(data='time', keys=True)if time ==940)
#print("SUBGRAPH:")
#print(SG.edges.data())
plt.figure(2)
nx.draw(SG,pos,with_labels=True,node_size=1, width=0.2)
#plt.title("time=940")


plt.savefig("images/ima2.png")
     
 
import cv2
import os

image_folder = 'images'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()    