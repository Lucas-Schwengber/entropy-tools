import networkx as nx
import numpy as np
import entropy_tools_slim as es
import matplotlib.pyplot as plt
from tqdm import tqdm


#Generate karate club graph
G = nx.karate_club_graph()

#Save number of nodes
n = len(G.nodes)

#Get degree of nodes
Deg = np.array(list(nx.degree_centrality(G).values()))*(n-1)

#Evaluates entropy for small time scales
HC,t = es.HCf(G,0.00001,0.000016,101,logscale=False)

#Plot everything
fig, axs = plt.subplots(1,1,figsize=(8,8))

#Sets the colormap
cmap = plt.cm.ScalarMappable(cmap='plasma')
cmap.set_clim(vmin=np.min(Deg),vmax=np.max(Deg))
colors = cmap.to_rgba(Deg)

#Plot the entropy of each node
for i in range(0,n):
    axs.plot(t,HC[i,:],color=colors[i],lw=1)

#Plot params
x_ticks = t[0::25]
axs.set_xticks(tuple(x_ticks))
axs.set_xticklabels(tuple(["{:.2e}".format(tick) for tick in x_ticks]),fontsize=12)

y_ticks = np.linspace(0,0.0009,10)
axs.set_yticks(tuple(y_ticks))
axs.set_yticklabels(tuple([0]+["{:.0e}".format(tick) for tick in y_ticks[1:]]),fontsize=12)

plt.yticks(fontsize=15)
plt.xticks(fontsize=15)

axs.set_ylabel("$C^H_i(t)$",fontsize=25)
axs.set_xlabel("t",fontsize=25)
cbar = plt.colorbar(cmap,orientation='vertical')
cbar.set_label(label='Degree', size='xx-large', weight='bold')
axs.tick_params(labelsize=15) 

#Show plot
plt.show()
