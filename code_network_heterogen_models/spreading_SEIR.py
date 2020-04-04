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


def SEIR_model(n,t,beta,mu, gamma,a):
    '''
    input: 
        n is total number of people
        t is time 
        beta is probability of getting sick because of the contact rate
        mu is probability of dieing
        gamma probability of recovering
        a is additional parameter for SEIR model for incubation period parameter
        the average incubation period is a^−1
    '''
    S,E,I,R=n
    dS_dt= -beta*S*I #- mu*S
    dE_dt= beta*S*I-(mu+a)*E
    dI_dt= a*E - (gamma +mu)*I
    dR_dt= gamma*I #- mu*R  #dissipation term
    

    return([dS_dt,dE_dt,dI_dt,dR_dt])



S0=0.7
E0=0.1
I0=0.1
R0=0.1
beta=0.35
mu=0.1
#assuming that the incubation period is a random variable with exponential distribution with parameter a {\displaystyle a} a 
a = 0.1
gamma = 0.9


#Then the basic reproduction number is 
R0 = a*1./(mu+a) *beta*1./(mu +gamma)
print('reproduction', R0)

t=np.linspace(0,100,10000)

solution=scipy.integrate.odeint(SEIR_model,[S0,E0,I0,R0],t,args=(beta,mu, gamma,a))
solution=np.array(solution)

plt.figure(figsize=[6,4])
plt.plot(t,solution[:,0],label="S(t)")
plt.plot(t,solution[:,1],label="E(t)")
plt.plot(t,solution[:,2],label="I(t)")
plt.plot(t,solution[:,3],label="R(t)")
plt.grid()
plt.legend()
plt.xlabel("Time")
plt.ylabel("Proportions")
plt.title("SEIR model")
plt.savefig('SEIR_beta_'+str(beta)+'_mu_'+str(mu)+'.png')
plt.show()


