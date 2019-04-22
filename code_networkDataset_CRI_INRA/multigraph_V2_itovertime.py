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


pos =nx.spring_layout(G)
#nx.draw_networkx(G,pos,with_labels=False,node_size=0.6, width=0.2)

#plt.savefig("contact_H/time_noattribute.png")


tes=nt['time']
#print(tes)
li=[];

prev=-1
for i in tes:
   # print(i)
   if i != prev:
       li.append(i)
       prev=i
#print(li)    
for x in li:
    #print(x)
    SG=G.edge_subgraph((u, v, keys) for (u, v, keys, time) in G.edges(data='time', keys=True)if time==x)
   # print (SG.edges.data())
    nx.draw(SG,pos,with_labels=True,node_size=0.6, width=0.2)
    plt.savefig("ima2/im"+str(x)+".png")
    
    #plt.clf()#this line can be uncommented if we want to separate the pictures here they added onto each other at each loop 
    pass  
###those lines need to be commented if one wants the code to run on the whole dataset
    if x==6980: 
       break

"""
code for the video
"""

import cv2
import os

image_folder = 'ima2'
video_name = 'V2_addition.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()    
