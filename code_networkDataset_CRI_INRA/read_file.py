# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:24:52 2019

@author: Aurel
"""
import pandas as pd
import numpy as np
import networkx as nx

####### OPEN GOWA FILES###########################
"""
opening gowa files
"""

#print("GOWALA_CHECKIN")

#df = pd.read_csv("C:/Users/Aurel/Documents/PythonFilestoOpen/Gowa/loc-gowalla_totalCheckins.txt.gz", compression='gzip', header=None, delim_whitespace =True)
#print(df)

#print("GOWALA_edges")
#df = pd.read_csv("C:/Users/Aurel/Documents/PythonFilestoOpen/Gowa/loc-gowalla_edges.txt.gz", compression='gzip', header=None, sep=',', quotechar='"')
#print(df)

##################################
"""
GOWA files put in pandas format with header
"""

#airport_col = ['ID', 'Name', 'City', 'Country','IATA', 'ICAO', 'Lat', 'Long', 'Alt', 
#               'Timezone', 'DST', 'Tz database time zone', 'type', 'source']


###########OPEN HOPITAL FILE###################
"""
hospital files put in pandas format with header
"""
col_H = ['time', 'Node1', 'Node2', 'ID1','ID2']

nodes = pd.read_csv("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/detailed_list_of_contacts_Hospital.dat_/s55Y6BgCeFB", delim_whitespace =True,names = col_H)

print(nodes)

nt=nodes[['Node1','Node2','time']]

print(nt)
G =nx.Graph() 
G=nx.from_pandas_edgelist(nt,'Node1','Node2','time')
print(G.edges.data())
#print("G")

SG=G.edge_subgraph((u,v) for u,v,d in G.edges.data('time') if (d < 580))

print ("subgraphe:" )
#print(SG.edges.data())

print(SG.edges.data())
     
     
     
