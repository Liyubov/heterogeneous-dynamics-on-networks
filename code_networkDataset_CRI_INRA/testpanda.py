# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:13:00 2019

@author: Aurel
"""
import networkx as nx

import pandas as pd
pd.options.display.max_columns = 20
import numpy as np
rng = np.random.RandomState(seed=5)
ints = rng.randint(1, 11, size=(3,2))
a = ['A', 'B', 'C']
b = ['D', 'A', 'E']
df = pd.DataFrame(ints, columns=['weight', 'cost'])
df[0] = a
df['b'] = b
df[['weight', 'cost', 0, 'b']]




G = nx.from_pandas_edgelist(df, 0, 'b', ['weight', 'cost'])
print(G['E']['C']['weight'])

print(G['E']['C']['cost'])

edges = pd.DataFrame({'source': [0, 1, 2],
                      'target': [2, 2, 3],
                      'weight': [3, 4, 5],
                      'color': ['red', 'blue', 'blue']})
print( edges)
G = nx.from_pandas_edgelist(edges, edge_attr=True)
print(G[0][2]['color'])

print("node")
print(G.edges.data())
