"""

"""

import networkx as nx
import random
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import os
import colorsys
from sklearn.metrics import mutual_info_score

directory = '/home/pclf/heteregeneous_epidemic_on_network'

import sys
sys.path.append(directory)
from function import *
from effect_dist import *

# Create the directories that will contain the results
path_measure = directory+'/measure_node'
path_plot = directory+'/plot'

if not os.path.exists(path_measure):
    os.makedirs(path_measure)
if not os.path.exists(path_plot):
    os.makedirs(path_plot)

# Define the parameter of the Watts-Strogatz graph
range_n=[12]
range_k=[4]
range_p=[0.2]
range_seed=[1] #[1,2,3,4,5]

# Define the parameter of the SIR model
beta=0.2 # Propability of infection 
gamma=0.2 # Probability of recovery 
first_infected = 0

# Define the heterogeneous node (only for the heterogeneous case)
heterogeneous_node=1
beta_i=beta*5

# Define the parameter of the analysis
t0=0
tmax=100
nb_simulation=10000

# Define the node measure we look at 
list_measure=[proba_be_infected,arrival_time]

# Run the simulation for homogeneous case
for n in range_n:
    for k in range_k:
        for p in range_p:
            for seed in range_seed:
                G = nx.watts_strogatz_graph(n,k,p,seed=seed)
                bc = nx.betweenness_centrality(G)
                #first_infected = np.argmax(list(bc.values()))
                first_infected = 0
                
                G0=create_watts_strogatz_graph(n,k,p,seed,beta,gamma,[first_infected])
                matrix= measure_analysis(G0,t0,tmax,list_measure,nb_simulation)
                for i in range(len(list_measure)):
                    measure_i = np.array([matrix[j][i] for j in range(nb_simulation)])
                    np.save(path_measure+'/Homogeneous_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_measure'+str(i)+'.npy',measure_i) 
                    del measure_i
                del matrix

# Run the simulation for heterogeneous case
for n in range_n:
    for k in range_k:
        for p in range_p:
            for seed in range_seed:
                G = nx.watts_strogatz_graph(n,k,p,seed=seed)
                list_edges = list(G.edges(heterogeneous_node))
                edges_weight = np.ones(len(list_edges))*beta_i

                G0=create_watts_strogatz_graph(n,k,p,seed,beta,gamma,[first_infected],
                                               list_heterogeneous_edges=list_edges,heterogeneous_edges_weight=edges_weight)
                matrix= measure_analysis(G0,t0,tmax,list_measure,nb_simulation)
                for i in range(len(list_measure)):
                    measure_i = np.array([matrix[j][i] for j in range(nb_simulation)])
                    np.save(path_measure+'/Heterogeneous1_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_measure'+str(i)+'.npy',measure_i) 
                    del measure_i
                del matrix

# Arbitrary size used for the plots
size=10

# Plot for homogeneous case (a lot of plots)
for n in range_n:
    for k in range_k:
        for p in range_p:
            for seed in range_seed:
                
                # Define the node color
                node_color=['blue']*n
                node_color[first_infected]='red'               
                
                # Create the graph and get the first computed measures
                G = nx.watts_strogatz_graph(n,k,p,seed=seed)
                bc = nx.betweenness_centrality(G)
                                
                G0=create_watts_strogatz_graph(n,k,p,seed,beta,gamma,[first_infected])
                plot_ring=True
                pos=central_layout(G0,first_infected)
                measure_i = np.load(path_measure+'/Homogeneous_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_measure'+str(0)+'.npy') 
                val = np.mean(measure_i,0)
                
                # Plot the graph with 'probability to be infected' as color
                plot_graph(G,pos,val=val,node_size=1,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Homogeneous_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_PBI.png')
                
                # Compute the distance of the nodes to the first infected nodes
                distance = np.array([int(np.round(np.sqrt(x1**2+x2**2),0)) for (x1,x2) in pos.values()])
                
                # Create array of color that will be used for plotting the distance as color
                max_dist = max(distance)
                HSV_tuples = [(x*1.0/max_dist+1, 0.5, 0.5) for x in range(max_dist+1)]
                list_color = tuple(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))                
                
                # Compute the effective betweenness centrality 
                measure_values = effective_BC(G0)
                measure_max = measure_values.max()
                node_size = 2*(measure_values+0.1*measure_max)/measure_max

                # Plot the graph with 'effective betweenness centrality' as node size
                plot_graph(G,pos,node_color=node_color,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Homogeneous_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_EBC.png')

                # Compute the effective distance
                measure_values = effective_distance(G0,first_infected,beta)
                measure_max = measure_values.max()
                node_size = 2*(measure_values+0.1*measure_max)/measure_max

                # Plot the graph with 'effective distance' as node size
                plot_graph(G,pos,node_color=node_color,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Homogeneous_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_ED.png')

                # Plot the 'effective distance' vs 'probability to be infected'
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i,s=10*size)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Effective distance to the first infected node",size=2*size)
                plt.ylabel('Probability to infect the node',size=2*size)
                plt.xscale('log')
                r=np.corrcoef(np.log(measure_values[measure_values>0]),val[measure_values>0])[0,1]
                plt.title('r='+str(np.round(r,2)),size=3*size)
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                #plt.title('MI = '+str(round(mutual_info_score(measure_values,val),2)),size=2*size)
                fig.savefig(path_plot+'/Homogeneous_Effective-distance_vs_proba-infection_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                
                # Create the P matrix and compute the analitycal effective distance
                adj_mat = nx.adjacency_matrix(G).todense()           
                P_mat = np.zeros([n,n])
                for i in range(n):
                    sum_col = np.sum(adj_mat[i,:])
                    for j in range(n):
                        if i==heterogeneous_node or j==heterogeneous_node:
                            P_mat[i,j] = beta_i*adj_mat[i,j]/sum_col
                        else:
                            P_mat[i,j] = beta*adj_mat[i,j]/sum_col
                
                lambda_val = 0.00001
                measure_values = np.array([eff_dist(beta, beta_i, P_mat, n,lambda_val, first_infected, i) for i in range(n)])                    
            
                # Plot the 'analitycal effective distance' vs 'probability to be infected'
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Analitycal effective distance to the first infected node",size=2*size)
                plt.ylabel('Probability to infect this node',size=2*size)
                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                #plt.title('MI = '+str(round(mutual_info_score(measure_values,val),2)),size=2*size)
                fig.savefig(path_plot+'/Homogeneous_Analitycal-effective-distance_vs_proba-be-infected_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                
                # Compute the sum of neighbors' conditionnal betwenness centrality
                cbc = node_conditionnal_BC(G0,first_infected)
                measure_values = np.zeros(n)
                for node in range(n):
                    neighbors = list(nx.neighbors(G,node))
                    measure_values[node] = sum(cbc[neighbors])

                # Plot the 'Sum of neighbors' conditionnal betweenness centrality' vs 'probability to be infected'
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i,s=10*size)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Sum of neighbors' conditionnal betweenness centrality",size=2*size)
                plt.ylabel('Probability to infect the node',size=2*size)
                plt.xscale('log')
                r=np.corrcoef(np.log(measure_values[measure_values>0]),val[measure_values>0])[0,1]
                plt.title('r='+str(np.round(r,2)),size=3*size) 
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                fig.savefig(path_plot+'/Homogeneous_Sum-Neighbors-CBC_vs_proba-infection_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')

                # Compute the conditionnal betwenness centrality                
                measure_values = node_conditionnal_BC(G0,first_infected)
                measure_max = measure_values.max()
                node_size = 2*(measure_values+0.1*measure_max)/measure_max
                
                # Plot the graph with 'conditionnal betwenness centrality ' as node size
                plot_graph(G,pos,node_color=node_color,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Homogeneous_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_CBC.png')

                # Plot the graph with 'conditionnal betwenness centrality ' as node size and 'probability to be infected' as color
                plot_graph(G,pos,val=val,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Homogeneous_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_PBI-CBC.png')

                # Plot the 'conditionnal betweenness centrality' vs 'probability to be infected'
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(cbc[distance_i],val[distance_i],color=list_color[i],label=i)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Conditionnal betweenness centrality",size=2*size)
                plt.ylabel('Probability to infect the node',size=2*size)
                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                fig.savefig(path_plot+'/Homogeneous_CBC_vs_proba-infection_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')


                # Get the second computed measures                                
                measure_i = np.array(np.load(path_measure+'/Homogeneous_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_measure'+str(1)+'.npy'))
                val=np.zeros(n)
                for i in range(n):
                    if i != first_infected:
                        val_i = measure_i[:,i]
                        val[i]=np.mean(val_i[val_i>0])
                        
                # Compute the effective distance  
                measure_values = effective_distance(G0,first_infected,beta)                    
                
                # Plot the 'effective distance' vs 'the arrival time'
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Effective distance to the first infected node",size=2*size)
                plt.ylabel('Arrival time of the infection to this node',size=2*size)
                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                #plt.title('MI = '+str(round(mutual_info_score(measure_values,val),2)),size=2*size)
                fig.savefig(path_plot+'/Homogeneous_Effective-distance_vs_arrival_time_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                                      
                # Create the P matrix and compute the analitycal effective distance
                adj_mat = nx.adjacency_matrix(G).todense()           
                P_mat = np.zeros([n,n])
                for i in range(n):
                    sum_col = np.sum(adj_mat[i,:])
                    for j in range(n):
                        if i==heterogeneous_node or j==heterogeneous_node:
                            P_mat[i,j] = beta_i*adj_mat[i,j]/sum_col
                        else:
                            P_mat[i,j] = beta*adj_mat[i,j]/sum_col
                
                lambda_val = 0.00001
                measure_values = np.array([eff_dist(beta, beta_i, P_mat, n,lambda_val, first_infected, i) for i in range(n)])                    
                
                # Plot the 'analytical effective distance' vs the 'arrival time'
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i)
                
                plt.plot(np.arange(0,15),np.arange(0,15),color='k')
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Analitycal effective distance to the first infected node",size=2*size)
                plt.ylabel('Arrival time of the infection to this node',size=2*size)
#                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                #plt.title('MI = '+str(round(mutual_info_score(measure_values,val),2)),size=2*size)
                fig.savefig(path_plot+'/Homogeneous_Analitycal-effective-distance_vs_arrival_time_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                
                # Compute the sum of neighbors' conditionnal betweenness centrality
                cbc = node_conditionnal_BC(G0,first_infected)
                measure_values = np.zeros(n)
                for node in range(n):
                    neighbors = list(nx.neighbors(G,node))
                    measure_values[node] = sum(cbc[neighbors])  

                # Plot the graph with the 'arrival time' as color
                plot_graph(G,pos,val=val,node_size=1,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Homogeneous_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_AR.png')

                # Plot the 'sum of neighbors' conditionnal betweenness centrality' vs the 'arrival time'
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})                
                plt.xlabel("Sum of neighbors' conditionnal betweenness centrality",size=2*size)
                plt.ylabel('Arrival time of the infection to this node',size=2*size)
                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                fig.savefig(path_plot+'/Homogeneous_Sum-Neighbors-CBC_vs_arrival-time_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                
                # Compute the conditionnal betweenness centrality
                measure_values = node_conditionnal_BC(G0,first_infected)
                measure_max = measure_values.max()
                node_size = 2*(measure_values+0.1*measure_max)/measure_max

                # Plot the graph with 'conditionnal betwenness centrality ' as node size
                plot_graph(G,pos,val=val,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Homogeneous_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_AR_CBC.png')
                
                # Plot the graph with 'conditionnal betwenness centrality ' as node size and 'probability to be infected' as color
                plot_graph(G,pos,node_color=node_color,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Homogeneous_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_CBC.png')

                # Plot the 'conditionnal betweenness centrality' vs the 'arrival time'
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i)

                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Conditionnal betweenness centrality",size=2*size)
                plt.ylabel('Arrival time',size=2*size)
                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                fig.savefig(path_plot+'/Homogeneous_CBC_vs_arrival-time_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')

                # Plot the graph in two layout (circular and central)              
                pos=central_layout(G0,first_infected)
                plot_graph(G,pos,node_color=node_color,plot_ring=True,with_labels=False,savefig=True,filename=path_plot+'/Homogeneous_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_layout_central.png')

                pos = nx.circular_layout(G0)
                plot_graph(G,pos,node_color=node_color,plot_ring=False,with_labels=False,savefig=True,filename=path_plot+'/Homogeneous_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_layout_circular.png')

                # Get the first computed measures
                measure_i = np.array(np.load(path_measure+'/Homogeneous_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_measure'+str(0)+'.npy'))
                val2 = np.mean(measure_i,0)
                
                # Plot the 'arrival time' vs the 'probability to be infected'
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(val[distance_i],val2[distance_i],color='k')
                        
                plt.xlabel("Arrival time",size=2*size)
                plt.ylabel('Probability to be infected',size=2*size)
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                fig.savefig(path_plot+'/Homogeneous_Arrival-time_vs_proba-be-infected_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')


# Same plots but for the heterogeneous case (without comment)
for n in range_n:
    for k in range_k:
        for p in range_p:
            for seed in range_seed:
                node_color=['blue']*n
                node_color[first_infected]='red'   
                node_color[heterogeneous_node]='black' 
                
                G = nx.watts_strogatz_graph(n,k,p,seed=seed)
                list_edges = list(G.edges(heterogeneous_node))
                edges_weight = np.ones(len(list_edges))*beta_i

                                
                G0=create_watts_strogatz_graph(n,k,p,seed,beta,gamma,[first_infected],
                                               list_heterogeneous_edges=list_edges,heterogeneous_edges_weight=edges_weight)
                plot_ring=True
                pos=central_layout(G0,first_infected)
                measure_i = np.load(path_measure+'/Heterogeneous1_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_measure'+str(0)+'.npy') 
                val = np.mean(measure_i,0)
                
                plot_graph(G,pos,val=val,node_size=1,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Heterogeneous1_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_PBI.png')
                
                distance = np.array([int(np.round(np.sqrt(x1**2+x2**2),0)) for (x1,x2) in pos.values()])

                max_dist = max(distance)
                HSV_tuples = [(x*1.0/max_dist+1, 0.5, 0.5) for x in range(max_dist+1)]
                list_color = tuple(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

                measure_values = effective_BC(G0)
                measure_max = measure_values.max()
                node_size = 2*(measure_values+0.1*measure_max)/measure_max

                plot_graph(G,pos,node_color=node_color,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Heterogeneous1_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_EBC.png')
                
                measure_values = effective_distance(G0,first_infected,beta)
                measure_max = measure_values.max()
                node_size = 2*(measure_values+0.1*measure_max)/measure_max

                plot_graph(G,pos,node_color=node_color,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Heterogeneous1_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_ED.png')

                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i,s=10*size)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Effective distance to the first infected node",size=2*size)
                plt.ylabel('Probability to infect the node',size=2*size)
                plt.xscale('log')
                r=np.corrcoef(np.log(measure_values[measure_values>0]),val[measure_values>0])[0,1]
                plt.title('r='+str(np.round(r,2)),size=3*size)                
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                #plt.title('MI = '+str(round(mutual_info_score(measure_values,val),2)),size=2*size)
                fig.savefig(path_plot+'/Heterogeneous1_Effective-distance_vs_proba-infection_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                
                ####
                adj_mat = nx.adjacency_matrix(G).todense()           
                P_mat = np.zeros([n,n])
                for i in range(n):
                    sum_col = np.sum(adj_mat[i,:])
                    for j in range(n):
                        if i==heterogeneous_node or j==heterogeneous_node:
                            P_mat[i,j] = beta_i*adj_mat[i,j]/sum_col
                        else:
                            P_mat[i,j] = beta*adj_mat[i,j]/sum_col
                
                lambda_val = 0.00001
                
                measure_values = np.array([eff_dist(beta, beta_i, P_mat, n,lambda_val, first_infected, i) for i in range(n)])                    
            
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Analitycal effective distance to the first infected node",size=2*size)
                plt.ylabel('Probability to infect this node',size=2*size)
                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                #plt.title('MI = '+str(round(mutual_info_score(measure_values,val),2)),size=2*size)
                fig.savefig(path_plot+'/Heterogeneous1_Analitycal-effective-distance_vs_proba-be-infected_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                
                cbc = node_conditionnal_BC(G0,first_infected)
                measure_values = np.zeros(n)
                for node in range(n):
                    neighbors = list(nx.neighbors(G,node))
                    measure_values[node] = sum(cbc[neighbors])

                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i,s=10*size)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Sum of neighbors' conditionnal betweenness centrality",size=2*size)
                plt.ylabel('Probability to infect the node',size=2*size)
                plt.xscale('log')
                r=np.corrcoef(np.log(measure_values[measure_values>0]),val[measure_values>0])[0,1]
                plt.title('r='+str(np.round(r,2)),size=3*size) 
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                fig.savefig(path_plot+'/Heterogeneous1_Sum-Neighbors-CBC_vs_proba-infection_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                
                measure_values = node_conditionnal_BC(G0,first_infected)
                measure_max = measure_values.max()
                node_size = 2*(measure_values+0.1*measure_max)/measure_max

                plot_graph(G,pos,val=val,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Heterogeneous1_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_PBI-CBC.png')
                plot_graph(G,pos,node_color=node_color,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Heterogeneous1_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_CBC.png')

                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(cbc[distance_i],val[distance_i],color=list_color[i],label=i)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Conditionnal betweenness centrality",size=2*size)
                plt.ylabel('Probability to infect the node',size=2*size)
                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                fig.savefig(path_plot+'/Heterogeneous1_CBC_vs_proba-infection_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                                
                measure_i = np.array(np.load(path_measure+'/Heterogeneous1_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_measure'+str(1)+'.npy'))
                val=np.zeros(n)
                for i in range(n):
                    if i != first_infected:
                        val_i = measure_i[:,i]
                        val[i]=np.mean(val_i[val_i>0])
                      
                measure_values = effective_distance(G0,first_infected,beta)                    
            
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Effective distance to the first infected node",size=2*size)
                plt.ylabel('Arrival time of the infection to this node',size=2*size)
                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                #plt.title('MI = '+str(round(mutual_info_score(measure_values,val),2)),size=2*size)
                fig.savefig(path_plot+'/Heterogeneous1_Effective-distance_vs_arrival_time_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                
                ####
                adj_mat = nx.adjacency_matrix(G).todense()           
                P_mat = np.zeros([n,n])
                for i in range(n):
                    sum_col = np.sum(adj_mat[i,:])
                    for j in range(n):
                        if i==heterogeneous_node or j==heterogeneous_node:
                            P_mat[i,j] = beta_i*adj_mat[i,j]/sum_col
                        else:
                            P_mat[i,j] = beta*adj_mat[i,j]/sum_col
                
                lambda_val = 0.00001
                
                measure_values = np.array([eff_dist(beta, beta_i, P_mat, n,lambda_val, first_infected, i) for i in range(n)])                    
            
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i)
                
                plt.plot(np.arange(0,15),np.arange(0,15),color='k')
                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Analitycal effective distance to the first infected node",size=2*size)
                plt.ylabel('Arrival time of the infection to this node',size=2*size)
#                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                #plt.title('MI = '+str(round(mutual_info_score(measure_values,val),2)),size=2*size)
                fig.savefig(path_plot+'/Heterogeneous1_Analitycal-effective-distance_vs_arrival_time_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                                                 
                cbc = node_conditionnal_BC(G0,first_infected)
                measure_values = np.zeros(n)
                for node in range(n):
                    neighbors = list(nx.neighbors(G,node))
                    measure_values[node] = sum(cbc[neighbors])  

                plot_graph(G,pos,val=val,node_size=1,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Heterogeneous1_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_AR.png')

                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i)
                
                plt.legend(title='Distance', prop={'size': 1.5*size})                
                plt.xlabel("Sum of neighbors' conditionnal betweenness centrality",size=2*size)
                plt.ylabel('Arrival time of the infection to this node',size=2*size)
                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                fig.savefig(path_plot+'/Heterogeneous1_Sum-Neighbors-CBC_vs_arrival-time_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
                
                measure_values = node_conditionnal_BC(G0,first_infected)
                measure_max = measure_values.max()
                node_size = 2*(measure_values+0.1*measure_max)/measure_max

                plot_graph(G,pos,val=val,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Heterogeneous1_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_AR_CBC.png')
                plot_graph(G,pos,node_color=node_color,node_size=node_size,plot_ring=plot_ring,with_labels=False,savefig=True,filename=path_plot+'/Heterogeneous1_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_CBC.png')

                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(measure_values[distance_i],val[distance_i],color=list_color[i],label=i)

                plt.legend(title='Distance', prop={'size': 1.5*size})
                plt.xlabel("Conditionnal betweenness centrality",size=2*size)
                plt.ylabel('Arrival time',size=2*size)
                plt.xscale('log')
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                fig.savefig(path_plot+'/Heterogeneous1_CBC_vs_arrival-time_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')
         
                pos=central_layout(G0,first_infected)
                plot_graph(G,pos,node_color=node_color,plot_ring=True,with_labels=False,savefig=True,filename=path_plot+'/Heterogeneous1_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_layout_central.png')

                pos = nx.circular_layout(G0)
                plot_graph(G,pos,node_color=node_color,plot_ring=False,with_labels=False,savefig=True,filename=path_plot+'/Heterogeneous1_Graph_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_layout_circular.png')

                measure_i = np.array(np.load(path_measure+'/Heterogeneous1_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'_measure'+str(0)+'.npy'))
                val2 = np.mean(measure_i,0)
                
                fig=plt.figure(figsize=(size,size))
                for i in set(distance):
                    if i >0:
                        distance_i = distance==i
                        plt.scatter(val[distance_i],val2[distance_i],color='k')

                plt.xlabel("Arrival time",size=2*size)
                plt.ylabel('Probability to be infected',size=2*size)
                plt.tick_params(axis='both', which='major', labelsize=1.5*size)
                fig.savefig(path_plot+'/Heterogeneous1_Arrival-time_vs_proba-be-infected_WS_'+str(n)+'_'+str(k)+'_'+str(p)+'_'+str(seed)+'_'+str(beta)+'_'+str(gamma)+'_'+str(first_infected)+'_'+str(nb_simulation)+'.png')                 