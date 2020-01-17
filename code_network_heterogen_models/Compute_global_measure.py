#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:25:17 2019

@author: pclf

"""

import networkx as nx
import random
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import os

directory = '/home/pclf/heteregeneous_epidemic_on_network'

import sys
sys.path.append(directory)
from function import *

# Create the directories that will contain the results
path_measure = directory+'/measure_global'
path_plot = directory+'/plot'

if not os.path.exists(path_measure):
    os.makedirs(path_measure)
if not os.path.exists(path_plot):
    os.makedirs(path_plot)

# Define the parameter of the Watts-Strogatz graph
n=30
k=4
p=0.2
seed=1

# Define the parameter of the SIR model
t0=0
tmax=50
beta=0.2 # Propability of infection 
gamma=0.2 # Probability of recovery 
first_infected = 0

# List of the different heterogeneous nodes you want to look at (one at a time, not all together)
list_heterogeneous_node = [1]

# Create the graph
G1 = nx.watts_strogatz_graph(n,k,p,seed=seed)

# Define the parameter of the analysis
nb_simulation=10000

# Define the global measure we look at 
measure=[end_number_S]

# Define the range of the beta_ij of the heterogeneous node
range_beta = np.arange(0.1,2.1,0.1)

# Create the matrix that will contain the results
matrix=np.zeros([len(list_heterogeneous_node),len(range_beta),2])

for i1 in range(len(list_heterogeneous_node)):
    # Print i1 to know the advancement of the code
    print(i1)
    
    # Get the heterogeneous node and it edges
    heterogeneous_node = list_heterogeneous_node[i1]
    list_edges = list(G1.edges(heterogeneous_node))
    
    # For each beta_ij, compute the measures and store it mean and std
    for i2 in range(len(range_beta)):
        beta_i=range_beta[i2]
        edges_weight = np.ones(len(list_edges))*beta_i
        G0=create_watts_strogatz_graph(n,k,p,seed,beta,gamma,[first_infected],
                                   list_heterogeneous_edges=list_edges,heterogeneous_edges_weight=edges_weight)
        
        matrix_simulation = measure_analysis(G0,t0,tmax,measure,nb_simulation)
        matrix[i1,i2,0] = np.mean(matrix_simulation,0)
        matrix[i1,i2,1] = np.std(matrix_simulation,0)

# Save the matrix
np.save(path_measure+'/WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(nb_simulation)+'.npy',matrix)
#os.system("shutdown now -h")

# Load the matrix if the previous code has been run
matrix = np.load(path_measure+'/WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(nb_simulation)+'.npy')

# Arbitrary size used for the plots
size=10

# Plot the evolution of the measure as a function of the beta_ij of the heterogeneous node
for heterogeneous_node_i in range(len(list_heterogeneous_node)):
    heterogeneous_node = list_heterogeneous_node[heterogeneous_node_i]
        
    mean = matrix[heterogeneous_node_i,:,0]
    std = matrix[heterogeneous_node_i,:,1]
     
    # Plot the reachability of the nodes on the graph for the different values of beta for the heteregenous node
    plt.figure(figsize=(size,size))
    plt.scatter(range_beta,n-mean,s=10*size)
    plt.xlabel("Beta value of the heterogeneous node",size=2*size)
    plt.ylabel('Mean number of node that have been infected',size=2*size)
    plt.tick_params(axis='both', which='major', labelsize=1.5*size)
    plt.savefig(path_plot+'/Beta-range_vs_end-S_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(first_infected)+'_'+str(heterogeneous_node)+'_'+str(nb_simulation)+'.png')
