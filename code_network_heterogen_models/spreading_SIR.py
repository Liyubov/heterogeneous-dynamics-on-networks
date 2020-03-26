# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:54:29 2020

@author: lyubo


## Content  
1. Simple Epidemics spreaing: Susceptible-Infected model on a network (viral spreading)
2. Epidemics spreaing: Susceptible-Infected-Recovered model


## Spreding on networks 


The main resources on spreading on networks can be found in [1].
We start with the simplest class of spreading, viral spreading on a network. This is a similar process to what you might observe in the case of rumor or inovation spreading in a social network. At each step, the "healthy" 
neighbors of any of the infectious nodes gets infected with fixed probability beta.

The SIR model is one of the simplest compartmental models 
(population), and many models are derivations of this basic form. 
The model consists of three parts or compartments – S for the number susceptible, I for the number of infectious, and R for the number recovered (or immune). This model is reasonably predictive for infectious diseases which are transmitted from human to human, 
and where recovery confers lasting resistance, such as measles, mumps and rubella. 
The SIR model is described by differential equations which we simulate below. 

[1] Spreading in networks, Vespigniani, et al. https://arxiv.org/abs/1408.2701


"""


import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
import scipy.integrate


def SIR_model(n,t,beta,mu):
    '''
    input: 
        n is total number of people
        t is time 
        beta is probability of getting sick 
        mu is probability of getting recovered
    '''
    S,I,R=n
    dS_dt=-beta*S*I
    dI_dt=beta*S*I-mu*I
    dR_dt=mu*I
    return([dS_dt,dI_dt,dR_dt])

S0=0.9
I0=0.1
R0=0.0
beta=0.35
mu=0.1

t=np.linspace(0,100,10000)

solution=scipy.integrate.odeint(SIR_model,[S0,I0,R0],t,args=(beta,mu))
solution=np.array(solution)

plt.figure(figsize=[6,4])
plt.plot(t,solution[:,0],label="S(t)")
plt.plot(t,solution[:,1],label="I(t)")
plt.plot(t,solution[:,2],label="R(t)")
plt.grid()
plt.legend()
plt.xlabel("Time")
plt.ylabel("Proportions")
plt.title("SIR model")
plt.show()


