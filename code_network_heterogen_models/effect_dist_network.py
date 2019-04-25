# -*- coding: utf-8 -*-
"""
Created on Tue Apr 02 18:10:22 2019

@author: lyubo

analytical expression for effective distance  
work with Flavio, Petter, Gael
"""


import numpy as np
#import matplotlib as plt 
#import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
#from matplotlib.patches import FancyArrowPatch, Circle
#import csv
import random
import scipy
#import seaborn as sns
from math import exp, expm1 
import math



'''
print('assigning values')
'''

print('assigning the matrices')

Nsize = 100
i_ind = 10 
j_ind = 5 
beta = 0.5 
beta_ij = 0.3 
lambda_val = 0.00001

'''
making measures
'''


def submatr(matrix1, j):  #matrix without jth column and jth row 
    #To delete the jth column
    matrix_del = np.delete(matrix1,(j-1), axis=1)
    return np.delete(matrix_del,(j-1), axis=0)
     

def matr_inv_elem(beta, beta_ij, P_matr, lambda_val, N, i_ind, k_ind, j_ind ):  #ith, kth element of inverse matrix   
    matr_inv = np.linalg.inv(np.identity(N-1) - np.exp(-lambda_val) * submatr(P_matr, j_ind)) #homogeneous effective distace
    matr_elem = matr_inv[i_ind,k_ind]
    return matr_elem

def eff_dist(beta, beta_ij, P_matr, N,lambda_val, i_ind, j_ind): #function to calculate effective distance analytically
    '''
    eff.distance between nodes i_ind and j_ind
    N is size of matrix
    
    P_matr is Markov chain matrix, 
    which is defined from degree matrix:
    P_ij = 1/deg(i)
    
    limit for lambda -> 0 is calculated for small lambda value
    '''

    
    P_matr_j = submatr(P_matr, j_ind)
    P_matr_col = P_matr[j_ind,:]
    
    print('calculating effective distance for each submatrix values')

#    x = range(T_wind) #number of steps
    k_ind = 10 
    eff_dist_comp = matr_inv_elem(beta, beta_ij, P_matr, lambda_val, N, i_ind, k_ind, j_ind ) * exp(-lambda_val)*P_matr_col[k_ind]
    sum_comp =  sum(matr_inv_elem(beta, beta_ij, P_matr, lambda_val, N, i_ind, k_ind, j_ind ) * exp(-lambda_val)*P_matr_col[k_ind] for k_ind in range(1,N-1))
    eff_dist = -math.log(sum_comp) 
    eff_dist_het = eff_dist * beta_ij
    
    return eff_dist_het


def MFPT_on_networks(P_matr, i_node, j_node):
    MFPT = 0 
    
    return MFPT


def trans_matrix(matrix, Nsize): #given adjacency matrix matrix
    trans_matr = np.zeros((Nsize, Nsize))
    
    for i in range(0, Nsize):
        sumi = np.sum(matrix, axis = 1) # sum for each row (matrix[i,:])
        print('degree')
        print(int(sumi[i]))
        trans_matr[i,:] = matrix[i,:]*1./int(sumi[1]) #normalize by degree
    return trans_matr

#***********************************************************************************************************



print('generating matrix')
Graph=nx.star_graph(Nsize)
Graph = nx.erdos_renyi_graph(Nsize,0.6)

adj_matrix = nx.adj_matrix(Graph) #P_matr = np.random.random((Nsize,Nsize))
adj_dense = adj_matrix.todense()
P_matr = trans_matrix(adj_dense, Nsize)
print('P_matr')
print(P_matr)


print('calculating effective distance')
eff_dis = eff_dist(beta, beta_ij, P_matr, Nsize,lambda_val, i_ind, j_ind)
print('finally effective distance for given G, i='+str(i_ind)+', j='+str(j_ind))
print(eff_dis)



