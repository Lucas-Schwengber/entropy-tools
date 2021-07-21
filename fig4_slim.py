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

#Get the time parameters associated with the peaks in correlation with other centralities
peaks = np.genfromtxt("peaks_karate_club.txt")

HC = [np.zeros((n,1)),np.zeros((n,1)),np.zeros((n,1))]

#Peak with degree
HC[0],_ = es.HCf(G,peaks[0],peaks[0],1)

#Peak with eigenvector
HC[1],_ = es.HCf(G,peaks[1],peaks[1],1)

#Peak with closeness
HC[2],_ = es.HCf(G,peaks[2],peaks[2],1)

#Computing the other centralities for comparison
Deg = list(nx.degree_centrality(G).values())

Close = list(nx.closeness_centrality(G).values())

Eigen = list(nx.eigenvector_centrality(G).values())

#Setting column widths for the plot
gridspec = {'width_ratios': [1, 1, 1, 0.1]}
fig, axs = plt.subplots(2,4,figsize=(20,12),gridspec_kw=gridspec)

#Spacing between plots
fig.tight_layout(pad=3.0)

#Graph layout
pos = nx.kamada_kawai_layout(G)

#Plot axis limits
xlims = (-0.66,0.66)
ylims = (-0.8,1.2)

#Colormap for nodes
cmap = plt.cm.magma
colors = plt.cm.ScalarMappable(cmap=cmap)

#Scheme to change node-size according to the relative values of HC at the given time
base_node_size = 300
var_node_size = 0
relHC_d = (HC[0]-np.min(HC[0]))/(np.max(HC[0])-np.min(HC[0]))
rel_d = (Deg-np.min(Deg))/(np.max(Deg)-np.min(Deg))

relHC_e = (HC[1]-np.min(HC[1]))/(np.max(HC[1])-np.min(HC[1]))
rel_e = (Eigen-np.min(Eigen))/(np.max(Eigen)-np.min(Eigen))

relHC_c = (HC[2]-np.min(HC[2]))/(np.max(HC[2])-np.min(HC[2]))
rel_c = (Close-np.min(Close))/(np.max(Close)-np.min(Close))

node_size = 120

fontsize = 40

nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[0,0],node_color = HC[0],cmap = cmap,node_size=base_node_size + var_node_size*relHC_d)
axs[0,0].set_title("(a)",y = 1,fontsize=fontsize)
axs[0,0].set_xlabel("$C^H_i(t_d)$",fontsize=fontsize)
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[0,0].spines["top"].set_visible(False)
axs[0,0].spines["bottom"].set_visible(False)
axs[0,0].spines["left"].set_visible(False)
axs[0,0].spines["right"].set_visible(False)

nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[1,0],node_color = Deg,cmap = cmap,node_size=base_node_size + var_node_size*rel_d)
axs[1,0].set_xlabel("Deg(i)",fontsize=fontsize)
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])
axs[1,0].spines["top"].set_visible(False)
axs[1,0].spines["bottom"].set_visible(False)
axs[1,0].spines["left"].set_visible(False)
axs[1,0].spines["right"].set_visible(False)

nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[0,2],node_color =  HC[1],cmap = cmap,node_size=base_node_size + var_node_size*relHC_e)
axs[0,1].set_title("(b)",y = 1,fontsize=fontsize)
axs[0,1].set_xlabel("$C^H_i(t_e)$",fontsize=fontsize)
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,1].spines["top"].set_visible(False)
axs[0,1].spines["bottom"].set_visible(False)
axs[0,1].spines["left"].set_visible(False)
axs[0,1].spines["right"].set_visible(False)

nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[1,2],node_color = Eigen,cmap = cmap,node_size=base_node_size + var_node_size*rel_e)
axs[1,1].set_xlabel("Eigenvector(i)",fontsize=fontsize)
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])
axs[1,1].spines["top"].set_visible(False)
axs[1,1].spines["bottom"].set_visible(False)
axs[1,1].spines["left"].set_visible(False)
axs[1,1].spines["right"].set_visible(False)


nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[0,1],node_color =  HC[2],cmap = cmap,node_size=base_node_size + var_node_size*relHC_c)
axs[0,2].set_title("(c)",fontsize=fontsize)
axs[0,2].set_xlabel("$C^H_i(t_c)$",fontsize=fontsize)
axs[0,2].set_xticks([])
axs[0,2].set_yticks([])
axs[0,2].spines["top"].set_visible(False)
axs[0,2].spines["bottom"].set_visible(False)
axs[0,2].spines["left"].set_visible(False)
axs[0,2].spines["right"].set_visible(False)

nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[1,1],node_color = Close,cmap = cmap,node_size=base_node_size + var_node_size*rel_c)
axs[1,2].set_xlabel("Closeness(i)",fontsize=fontsize)
axs[1,2].set_xticks([])
axs[1,2].set_yticks([])
axs[1,2].spines["top"].set_visible(False)
axs[1,2].spines["bottom"].set_visible(False)
axs[1,2].spines["left"].set_visible(False)
axs[1,2].spines["right"].set_visible(False)


#Setting the colorbar
gs = axs[0, 3].get_gridspec()

for ax in axs[0:,3]:
    ax.remove()

axbig = fig.add_subplot(gs[0:,3])

cax = axbig

cax.tick_params(labelsize=40)

cbar = plt.colorbar(colors,cax=cax)
cbar.set_label(label='$C^H_i(t)$', size=40, weight='bold')

#Save figure (too wide to show)
plt.savefig("karate_heatmap.jpeg",bbox_inches="tight", dpi=300)