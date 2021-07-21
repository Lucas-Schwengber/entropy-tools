import networkx as nx
import numpy as np
import entropy_tools_slim as es
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

#Funcion to compute row sum of A^k where A = adjacency matrix of G 
def row_sum(G,k):
    A = nx.to_numpy_array(G)
    M = np.linalg.matrix_power(A, k)
    return np.sum(M,axis=0)

#Get network from netscience.gml
G = nx.read_gml("netscience.gml",label='id')

#Get main connected component
G = G.subgraph(max(nx.connected_components(G), key=len))

n = len(G)

A = nx.to_numpy_array(G)

#Range for temperature parameter
logti = -2
logtf = 1

ti = 10**logti
tf = 10**logtf
m = 101

#Compute total comunicability
TC,t = es.TCf(G,ti,tf,m)

#Sequence of powers of A to be evaluated
pot = np.arange(1,35,2)

scales = len(pot)

row_s = np.zeros((scales,n))

#Evaluate the row sums for each power of A
for i in range(0,scales):
    row_s[i,:] = row_sum(G,pot[i])


#Sets the colormap
colors = plt.cm.ScalarMappable(cmap='plasma').to_rgba(pot)

#Evaluate Closeness and Eigen vector centralities
Close = list(nx.closeness_centrality(G).values())

Eigen = list(nx.eigenvector_centrality(G).values())

fig,axs = plt.subplots(1,1,figsize = (10,8))

lw = 3

#Matrix to store the spearman correlations
spearman = np.zeros((scales+2,m))

#Compute and plot the correlation with the row sums of each power of A
for j in range(0,scales):
    for i in range(0,m):
        spearman[j,i] = stats.spearmanr(row_s[j,:],TC[:,i])[0]
    
    if(j == 0):
        axs.semilogx(t,spearman[j,:],label="Degree",color='Blue',lw=lw)
    else:
        axs.semilogx(t,spearman[j,:],color=colors[j],lw=lw/2)

#Compute the spearman correlation with closeness and eigenvector
for i in range(0,m):
    spearman[scales,i] = stats.spearmanr(Close,TC[:,i])[0]
    spearman[scales+1,i] = stats.spearmanr(Eigen,TC[:,i])[0]

#Plot the correlation with Closeness and Eigenvector
axs.semilogx(t,spearman[scales+1,:],label="Eigenvector",color='Red',lw=lw)
axs.semilogx(t,spearman[scales,:],label="Closeness",color='Green',lw=lw)

#Set plot parameters
x_ticks = np.round([0.01,0.05,0.3,1.8,10],2)
axs.set_xticks(tuple(x_ticks))
axs.set_xticklabels(tuple([format(tick,'.2f') for tick in x_ticks]),fontsize=15)

y_ticks = np.round(np.linspace(0,1.0,6),2)
axs.set_yticks(tuple(y_ticks))
axs.set_yticklabels(tuple([str(tick) for tick in y_ticks]),fontsize=15)

axs.legend(loc='lower left',fontsize='xx-large')
axs.set_ylabel(r"$\rho(TC(\beta),\cdot)$",fontsize=25)
axs.set_xlabel(r"$\beta$",fontsize=25)

cmap = plt.cm.ScalarMappable(cmap='plasma')
cmap.set_clim(vmin=pot[0],vmax=pot[-1])

cbar = plt.colorbar(cmap)
cbar.set_label(label='$k$', size=25, weight='bold')
cbar.ax.tick_params(labelsize=15) 

#Show plot
plt.show()