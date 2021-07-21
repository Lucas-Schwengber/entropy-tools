import networkx as nx
import numpy as np
import entropy_tools_slim as es
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import pandas as pd

#Generate ORBIS graph
G = nx.Graph()

nodes = pd.read_csv('orbis_nodes_corrected.csv', sep=',', encoding='latin-1')
edges = pd.read_csv('orbis_edges_0514.csv', sep=',', encoding='latin-1')

n = len(nodes)
m = len(edges)

#Add nodes
G.add_nodes_from(list(nodes["id"]))

#Add edges, as long as the associated nodes are on the graph
G.add_edges_from([(edges.iloc[i,0],edges.iloc[i,1]) for i in range(0,m)])

#Add node attributes

#Label
nx.set_node_attributes(G, dict(zip(nodes["id"], nodes["label"])), 'label')

#x and y position on map
pos = dict(zip(nodes["id"],list(zip(nodes["y"], nodes["x"]))))
nx.set_node_attributes(G, pos, 'pos')

#Store number of nodes
n = len(G.nodes)

A = nx.to_numpy_array(G)

#Range for temperature parameter
logti = -2
logtf = 1

ti = 10**logti
tf = 10**logtf
m = 101

#Compute total comunicability
TC,t = es.TCf(G,ti,tf,m)

#Other centrality measures for comparison
Deg = list(nx.degree_centrality(G).values())

Betw = list(nx.betweenness_centrality(G).values())

Close = list(nx.closeness_centrality(G).values())

Eigen = list(nx.eigenvector_centrality(G).values())

#Spearman correlation
Spearman = np.zeros((4,m))

print("Computing correlations...")
for j in tqdm(range(0,m)):
    Spearman[0,j] = stats.spearmanr(Deg,TC[:,j])[0]
    Spearman[1,j] = stats.spearmanr(Betw,TC[:,j])[0]
    Spearman[2,j] = stats.spearmanr(Eigen,TC[:,j])[0]
    Spearman[3,j] = stats.spearmanr(Close,TC[:,j])[0]

#Salva os picos de cada correlacao
picos = []
picos.append(t[np.argmax(Spearman[0,:])])
picos.append(t[np.argmax(Spearman[1,:])])
picos.append(t[np.argmax(Spearman[2,:])])
picos.append(t[np.argmax(Spearman[3,:])])

np.savetxt("picos_orbis.txt",picos)

#Plot results
fig,axs = plt.subplots(1,1,figsize = (8,8))

lw = 3

axs.semilogx(t,Spearman[0,:],label = r"Degree",lw=lw,color='Blue')
plt.semilogx(t,Spearman[1,:],label = r"Betweenness",lw=lw,color='Orange')
axs.semilogx(t,Spearman[2,:],label = r"Eigenvector",lw=lw,color='red')
axs.semilogx(t,Spearman[3,:],label = r"Closeness",lw=lw,color='green')

#Plot params
x_ticks = np.round([0.01,0.05,0.3,1.8,10],2)
axs.set_xticks(tuple(x_ticks))
axs.set_xticklabels(tuple([str(tick) for tick in x_ticks]),fontsize=15)

y_ticks = np.round(np.linspace(0,1.0,6),4)
axs.set_yticks(tuple(y_ticks))
axs.set_yticklabels(tuple([str(tick) for tick in y_ticks]),fontsize=15)

axs.legend(loc='lower left',fontsize='xx-large')
axs.set_ylabel(r"$\rho(TC(\beta),\cdot)$",fontsize=25)
axs.set_xlabel(r"$\beta$",fontsize=25)

#Show plot
plt.show()