# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import networkx as nx 
import matplotlib.pyplot as plt

"""
fh="C:/Users/Aurel/Documents/PythonFilestoOpen/bio-CE-LC.edges"
#fh="C:/Users/Aurel/Documents/PythonFilestoOpen/txtfile.txt"
plt.subplot(121)

plt.subplots_adjust(left  = 0.1,right = 1.5, bottom = 0.1,wspace=0.5)
G = nx.read_weighted_edgelist(fh)

#print(G.edges(data = True))
#print(G.nodes)
#nx.draw(pos,node_size=0.6)
#pos=nx.spring_layout(G)
pos=nx.circular_layout(G)
x,y=pos.get('1')

print("Number of node: "+ str(G.number_of_nodes()))


nx.draw_networkx(G,pos, with_labels=False, node_size=0.6)
plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/bio-CE-LC/circular_bio-CE-LC.png")
nx.write_gml(G,"C:/Users/Aurel/Documents/PythonFilestoOpen/bio-CE-LC/circularbio.gml")

"""

"""
test for the spreading

"""
N=10**4
G=nx.barabasi_albert_graph(N, 5)

print("begin")

import EoN

tmax = 70
iterations = 5  #run 5 simulations
tau = 0.9           #transmission rate
gamma = 1.0    #recovery rate
rho = 0.500 #intial infected??? 0.005 in rate ? like 0.005 = 0.5% ou 5%of the population infected
plt.subplot(122)

for counter in range(iterations): #run simulations
    t, S, I, R = EoN.fast_SIR(G, tau, gamma, rho=rho, tmax = tmax)
    if counter == 0:
        plt.plot(t, I, color = 'k', alpha=0.3, label='Simulation')
    plt.plot(t, I, color = 'blue', alpha=0.3)
    

plt.xlabel('$t$')
plt.ylabel('Number infected')


"""
test with display for the spreading

"""

"""
sim = EoN.fast_SIR(G, tau, gamma, rho=rho,return_full_data=True, tmax = tmax)
sim.set_pos(pos)
sim.display(7, node_size = 4) #display time 7

plt.savefig("C:/Users/Aurel/Documents/PythonFilestoOpen/bio-CE-LC/circular2_bio-CE-LC.png")


"""
