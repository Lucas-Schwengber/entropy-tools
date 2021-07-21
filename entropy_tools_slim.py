import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.linalg import expm

#####################################
## UTILITY FUNCTIONS AND CONSTANTS ##
#####################################

#tolerance for error on probability mass sum
tolH = 0.001

#Computes x*log(x)
def logl(x):
    if x <= 0:
        return 0
    else:
        return x*np.log2(x)

#Entropy of a given vector representing a probability distribution
def Hv(P):
    n = len(P)
    if all(P>=-tolH) and abs(sum(P)-1)<=tolH :
        h = 0
        for i in range(0,n):
            h = h - logl(P[i])
        
        return h
    else:
        print("Invalid input")

#Laplacian matrix
def lapM(A):
    n = len(A)
    D = np.diag(np.array((np.dot(np.ones((1,n)),A)))[0])
    L = D - A
    return L

###############
#EVALUATING HC#
###############

##Complete HC-centrality, all nodes and all times (default in log scale)
#Currently supporting only for symmetric matrices and connected graphs
#For non-connected graphs one may simply calculate the centrality of each individual component without normalization
#Input: 
# G = a networkx graph
# ti = initial time
# tf = final time
# m = number of points in the time grid
def HCf(G,ti,tf,m,logscale=True,norm=True):
    #Extract Laplacian Matrix
    A = nx.to_numpy_array(G)
    L = lapM(A)

    n = len(L)

    HC = np.zeros((n,m))

    #Check for logscale
    if(logscale):
        if(ti <= 0):
            raise ValueError("Initial time must be greater than 0 for log-scale")
        if(ti>tf):
            raise ValueError("Final time must be greater or equal to the Initial time")

        a = np.log10(ti)
        b = np.log10(tf)
        t = np.power(10,np.linspace(a,b,m))
    else:
        if(ti < 0):
            raise ValueError("Initial time must be non-negative")
        if(ti>tf):
            raise ValueError("Final time must be greater or equal to the Initial time")
        
        t = np.linspace(ti,tf,m)
    
    #Computes the spectral decomposition of L
    print("Getting spectral decomposition...")
    D, Q = np.linalg.eigh(L)
    Qinv = np.matrix.transpose(Q)

    print("Calculating the entropies at each time...")
    for j in tqdm(range(0,m)):
        P = Q @ expm(-t[j]*np.diag(D)) @ Qinv

        for i in range(0,n):
            HC[i,j] = Hv(P[:,i])

    if(norm):
        return HC/np.log2(n),t
    else:
        return HC,t

#############
#OTHER STUFF#
#############

#Total comunicability. Input format is the same as HC
def TCf(G,ti,tf,m,logscale=True):
    A = nx.to_numpy_array(G)

    n = len(A)

    TC = np.zeros((n,m))

    #Check for logscale
    if(logscale):
        if(ti <= 0):
            raise ValueError("Initial time must be greater than 0 for log-scale")
        if(ti>tf):
            raise ValueError("Final time must be greater or equal to the Initial time")

        a = np.log10(ti)
        b = np.log10(tf)
        t = np.power(10,np.linspace(a,b,m))
    else:
        if(ti < 0):
            raise ValueError("Initial time must be non-negative")
        if(ti>tf):
            raise ValueError("Final time must be greater or equal to the Initial time")
        
        t = np.linspace(ti,tf,m)

    print("Calculating total comunicability at each time...")
    for j in tqdm(range(0,m)):
        M = expm(t[j]*A)

        for i in range(0,n):
            TC[i,j] = np.sum(M[i,:])

    return TC,t