# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:08:39 2019

@author: Aurel
"""
"""

fh="C:/Users/Aurel/Documents/PythonFilestoOpen/bio-CE-LC.edges"
file = open(fh, "r")
fout= open("C:/Users/Aurel/Documents/PythonFilestoOpen/txtfile.txt","w")
for line in file:   
  fout.write(line)
"""

"""
BUG!!!WHY ??? OK NOW It is working without doing anything... so Be carefull to make sure it does the right thing
from a mtx file to a txt file, delet the two first line in order to not have the comments
"""

"""
fh2="C:/Users/Aurel/Documents/PythonFilestoOpen/ia-infect-dublin.mtx"
file = open(fh2, "r")
fout2= open("C:/Users/Aurel/Documents/PythonFilestoOpen/txtfile3.txt","w")
index=1
for line in file:   
    if index<3:
        index=index+1 
    else:
        print(line)
        fout2.write(line)

"""
"""
NOT WORKING FOR THIS ONE? OPENING MTX IN PROGRESS

"""
"""
from scipy.io import mmread
#from scipy.io import mminfo
#B=mminfo("C:/Users/Aurel/Documents/PythonFilestoOpen/road_asia/road-asia-osm.mtx")
A = mmread("C:/Users/Aurel/Documents/PythonFilestoOpen/road-minnesota.mtx")
"""
"""
fout2= open("C:/Users/Aurel/Documents/PythonFilestoOpen/road_asia/r_awithhead.txt","w")

for line in A:   
        fout2.write(line)
"""

"""
fh="C:/Users/Aurel/Documents/PythonFilestoOpen/road-minnesota.mtx"
file = open(fh, "r")
fout= open("C:/Users/Aurel/Documents/PythonFilestoOpen/r_Mi.txt","w")
for line in file:   
  fout.write(line)

"""

##############################



fh="C:/Users/Aurel/Documents/PythonFilestoOpen/Gowa/loc-gowalla_edges.txt/s56ZD6S6vWA"
file = open(fh, "r")
#fout= open("C:/Users/Aurel/Documents/PythonFilestoOpen/contact_H/H.txt","w")
for line in file:   
  #fout.write(line)
  print(line)


