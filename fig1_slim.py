import networkx as nx
import numpy as np
import entropy_tools_slim as es
import matplotlib.pyplot as plt

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

#Sequence of degrees
deg = [3,5,7]

tree_star(0,G,deg)

n = len(G.nodes)

#Node colors
cor = ['#0000ff','#0066ff','#3399ff','#33ccff']

#Setting node colors, size and labels
cores = ['']*n
sizes = [300]*n
labels = {}
for i in range(0,n):
    labels[i] = ''
    if(G.degree[i] == deg[0]):
        cores[i] = cor[0]
        sizes[i] = 450
    if(G.degree[i] == deg[1]+1):
        cores[i] = cor[1]
        sizes[i] = 400
    if(G.degree[i] == deg[2]+1):
        cores[i] = cor[2]
        sizes[i] = 350
    if(G.degree[i] == 1):
        cores[i] = cor[3]
        sizes[i] = 300

cores[0] = 'blue'
cores[1] = 'orange'
cores[2] = 'green'
cores[3] = '#F08080'
labels[0] = 1
labels[1] = 2
labels[2] = 3
labels[3] = 4

#Graph layout
pos = nx.kamada_kawai_layout(G)

#Create plot
fig = plt.figure(figsize=(7,7))
plt.axis("off")
#Draw graph
nx.draw_networkx(G, pos = pos, with_labels=True,node_color = cores, node_size = sizes, labels = labels)

#Show image
plt.show()