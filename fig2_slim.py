import networkx as nx
import numpy as np
import entropy_tools_slim as es
import matplotlib.pyplot as plt
from tqdm import tqdm

#Generate a tree-like graph with different degrees for each level
def tree_star(T,G,lista):
    if(len(lista) > 0):
        for i in range(1,lista[0]+1):
            C = len(G.nodes)
            G.add_node(C)
            G.add_edge(C,T)
            tree_star(C,G,lista[1:])

G = nx.Graph()

G.add_node(0)

deg = [3,5,7]

tree_star(0,G,deg)

n = len(G.nodes)

#Compute eigenvector centrality and closeness
eigen = nx.eigenvector_centrality(G)
close = nx.closeness_centrality(G)

#Evaluate the entropy of a diffusion on the graph starting at all nodes
HC,t = es.HCf(G,10**(-2.1),10**(0),300) 

#Create plot
fig, axs = plt.subplots(1,1,figsize=(8,8))

#Colors
cores = ['blue','orange','green','#F08080']

#Plot entropy of diffusion starting at different nodes
axs.plot(t,HC[0],color=cores[0],label="$C^H_1(t)$")
axs.plot(t,HC[1],color=cores[1],label="$C^H_2(t)$")
axs.plot(t,HC[2],color=cores[2],label="$C^H_3(t)$")
axs.semilogx(t,HC[3],color=cores[3],label="$C^H_4(t)$")
axs.legend(loc='center',fontsize='xx-large',bbox_to_anchor=(0.2,0.8))

#Title
axs.set_title("",y = 1)

#ylim
axs.set_ylim((0,1.01))

#xlim
axs.set_xlim((t[0],t[-1]))

#Labels
axs.set_xlabel("t",fontsize=25)
axs.set_ylabel("$C^H(t)$",fontsize=25)

#x ticks
axs.set_xticks((0.01,0.4,0.8))
axs.set_xticklabels(('0.01','0.40','0.80'),fontsize=15)
axs.set_yticks((0.2,0.4,0.6,0.8,1.0))
axs.set_yticklabels(('0.2','0.4','0.6','0.8','1.0'),fontsize=15)

#vertical lines
axs.axvspan(0.01,0.01001, alpha=0.75,color='gray')
axs.axvspan(0.4,0.401, alpha=0.75,color='gray')
axs.axvspan(0.8,0.801, alpha=0.75,color='gray')

#Show image
plt.show()
