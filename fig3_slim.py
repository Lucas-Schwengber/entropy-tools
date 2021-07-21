import networkx as nx
import numpy as np
import entropy_tools_slim as es
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

#Generate karate club graph
G = nx.karate_club_graph()

#Save number of nodes
n = len(G.nodes)

#Get adjacency matrix
A = nx.to_numpy_array(G)

#Final and initial time for t scale
ti = 10**(-1.25)
tf = 1.5

#Time scale grid size
m = 1000

#Evaluate the entropy of the diffusion starting at all nodes 
HC,t = es.HCf(G,ti,tf,m)

#Evaluate other centralities for comparison
Deg = list(nx.degree_centrality(G).values())

Betw = list(nx.betweenness_centrality(G).values())

Close = list(nx.closeness_centrality(G).values())

Eigen = list(nx.eigenvector_centrality(G).values())

#Spearmanrank
Spearman = np.zeros((4,m))

print("Computing correlations...")
for j in tqdm(range(0,m)):
    Spearman[0,j] = stats.spearmanr(Deg,HC[:,j])[0]
    Spearman[1,j] = stats.spearmanr(Close,HC[:,j])[0]
    Spearman[2,j] = stats.spearmanr(Eigen,HC[:,j])[0]

#Time parameters associated with the peaks in correlation with other centrality measures (used in fig 4)
peaks = []
peaks.append(t[np.argmax(Spearman[0,:])])
peaks.append(t[np.argmax(Spearman[1,:])])
peaks.append(t[np.argmax(Spearman[2,:])])

np.savetxt("peaks_karate_club.txt",peaks)

#Plotting the results

fig,axs = plt.subplots(1,1,figsize = (8,8))

lw = 3

axs.semilogx(t,Spearman[0,:],label = r"Degree",lw=lw,color='Blue')
axs.semilogx(t,Spearman[1,:],label = r"Closeness",lw=lw,color='green')
axs.semilogx(t,Spearman[2,:],label = r"Eigenvector",lw=lw,color='red')

x_ticks = np.round([0.05,0.15,0.30,0.60,1.50],2)
axs.set_xticks(tuple(x_ticks))
axs.set_xticklabels(tuple([format(tick,'.2f') for tick in x_ticks]),fontsize=15)

y_ticks = np.round(np.linspace(0.6,1.0,5),4)
axs.set_yticks(tuple(y_ticks))
axs.set_yticklabels(tuple([str(tick) for tick in y_ticks]),fontsize=15)

axs.legend(loc='lower left',fontsize='xx-large')
axs.set_ylabel(r"$\rho(C^H(t),\cdot)$",fontsize=25)
axs.set_xlabel('t',fontsize=25)

#Show figure
plt.show()
