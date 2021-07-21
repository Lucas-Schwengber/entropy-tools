import networkx as nx
import numpy as np
import entropy_tools_slim as es
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import re
from mpl_toolkits.axes_grid1 import ImageGrid

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

df = gpd.read_file('custom.geo.json')

#Store number of nodes
n = len(G.nodes)

A = nx.to_numpy_array(G)

#Gera intervalo de tempos
peaks = np.genfromtxt("peaks_orbis.txt")

HC = [np.zeros((n,1)),np.zeros((n,1)),np.zeros((n,1))]

#Peak with degree
HC[0],_ = es.HCf(G,peaks[0],peaks[0],1)

#Peak with eigenvector
HC[1],_ = es.HCf(G,peaks[1],peaks[1],1)

#Peak with closeness
HC[2],_ = es.HCf(G,peaks[2],peaks[2],1)

#Other centralities
Deg = list(nx.degree_centrality(G).values())

Eigen = list(nx.eigenvector_centrality(G).values())

Close = list(nx.closeness_centrality(G).values())

#Setting column widths for the plot
gridspec = {'width_ratios': [1, 1, 1, 0.1]}
fig, axs = plt.subplots(2,4,figsize=(20,12),gridspec_kw=gridspec)

#Linewidth
lw = 0.4

#Colormap
cmap = plt.cm.magma
colors = plt.cm.ScalarMappable(cmap=cmap)

#Scheme to change node-size according to the relative values of HC at the given time
node_size = 5
base_node_size = 7
var_node_size = 20
relHC_d = (HC[0]-np.min(HC[0]))/(np.max(HC[0])-np.min(HC[0]))
rel_d = (Deg-np.min(Deg))/(np.max(Deg)-np.min(Deg))

relHC_e = (HC[1]-np.min(HC[1]))/(np.max(HC[1])-np.min(HC[1]))
rel_e = (Eigen-np.min(Eigen))/(np.max(Eigen)-np.min(Eigen))

relHC_c = (HC[2]-np.min(HC[2]))/(np.max(HC[2])-np.min(HC[2]))
rel_c = (Close-np.min(Close))/(np.max(Close)-np.min(Close))

#node_size = base_node_size + var_node_size*relHC
fontsize = 30

#axes range
xlims = (-12,44)
ylims = (20,57)

#Color of background of the plot
axs[0,0].set_facecolor("#76F7FF")
axs[0,1].set_facecolor("#76F7FF")
axs[1,0].set_facecolor("#76F7FF")
axs[1,1].set_facecolor("#76F7FF")
axs[0,2].set_facecolor("#76F7FF")
axs[1,2].set_facecolor("#76F7FF")

#Plot heatmap
df.plot(ax = axs[0,0],color="#8AEB80")
nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[0,0],node_color = HC[0],cmap = cmap,node_size=base_node_size + var_node_size*relHC_d,width=lw)
axs[0,0].set_title("(a)",fontsize=fontsize)
axs[0,0].set_xlabel("$C^H_i(t_d)$",fontsize=fontsize)
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[0,0].set_xlim(xlims)
axs[0,0].set_ylim(ylims)


df.plot(ax = axs[1,0],color="#8AEB80")
nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[1,0],node_color = Deg,cmap = cmap,node_size=base_node_size + var_node_size*rel_d,width=lw)
axs[1,0].set_xlabel("Deg(i)",fontsize=fontsize)
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])
axs[1,0].set_xlim(xlims)
axs[1,0].set_ylim(ylims)

df.plot(ax = axs[0,1],color="#8AEB80")
nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[0,1],node_color = HC[1],cmap = cmap,node_size=base_node_size + var_node_size*relHC_e,width=lw)

axs[0,1].set_title("(b)",y = 1,fontsize=fontsize)
axs[0,1].set_xlabel("$C^H_i(t_e)$",fontsize=fontsize)
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,1].set_xlim(xlims)
axs[0,1].set_ylim(ylims)

df.plot(ax = axs[1,1],color="#8AEB80")
nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[1,1],node_color = Eigen,cmap = cmap,node_size=base_node_size + var_node_size*rel_e,width=lw)

axs[1,1].set_xlabel("Eigenvector(i)",fontsize=fontsize)
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])
axs[1,1].set_xlim(xlims)
axs[1,1].set_ylim(ylims)

df.plot(ax = axs[0,2],color="#8AEB80")
nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[0,2],node_color = HC[2],cmap = cmap,node_size=base_node_size + var_node_size*relHC_c,width=lw)

axs[0,2].set_title("(c)",fontsize=fontsize)
axs[0,2].set_xlabel("$C^H_i(t_c)$",fontsize=fontsize)
axs[0,2].set_xticks([])
axs[0,2].set_yticks([])
axs[0,2].set_xlim(xlims)
axs[0,2].set_ylim(ylims)

df.plot(ax = axs[1,2],color="#8AEB80")
nx.draw_networkx(G, pos = pos, with_labels=False,ax = axs[1,2],node_color = Close,cmap = cmap,node_size=base_node_size + var_node_size*rel_c,width=lw)

axs[1,2].set_xlabel("Closeness(i)",fontsize=fontsize)
axs[1,2].set_xticks([])
axs[1,2].set_yticks([])
axs[1,2].set_xlim(xlims)
axs[1,2].set_ylim(ylims)

#Settings to plot the colorbar
gs = axs[0, 3].get_gridspec()
# remove the underlying axes
for ax in axs[0:,3]:
    ax.remove()

axbig = fig.add_subplot(gs[0:,3])

cax = axbig

cax.tick_params(labelsize=30)

cbar = plt.colorbar(colors,cax=cax)
cbar.set_label(label='$C^H_i(t)$', size=35, weight='bold')

#Show plot
plt.show()
