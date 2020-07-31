# MFGC connection network

import numpy as np
from numpy import random
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

random.seed( 1 )

DELTAT = 0.001
NSTEPS = 1000

class param:
    def  __init__(type):
        self.pS0=


paramlist = [param(1), param(2)]

class GranuleCell:
    def __init__(self,mflist):
        self.MF = mflist
        self.response = np.zeros(NSTEPS,dtype=np.float)
        self.synapses = []
        for i in np.arange(4):
            self.synapses.append(Synapse(mflist[i]))

class Synapse:
    def __init__(self,T):
        self.X = np.zeros(shape=[2,NSTEPS],dtype=np.float)
        self.P = np.zeros(shape=[2,NSTEPS],dtype=np.float)

class MFGC:
    def __init__(self, nMF, nGC, MFTYPES):
        self.nMF = nMF
        self.nGC = nGC
        self.MFTYPES = MFTYPES
        self.connectionArray = np.zeros(shape=[nGC,4],dtype=np.int)

        ## assign MF labels, please
        self.MFTYPES = np.array([1,2])

        labelMFDict = {}
        for loopMF in np.arange(nMF):
            labelMFDict.update({loopMF: np.random.choice(self.MFTYPES)})
            #I made a dictionary to randomly assign types to each MF 

        print (labelMFDict)
        print (labelMFDict.get(2))

        self.wire()

    def wire(self):
        for gc in np.arange(self.nGC):
            self.connectionArray[gc,:] = np.random.choice(np.arange(nMF),size=4,replace=False)

        #self.connectionArray=np.random.choice(np.arange(nMF),size=(nGC,4),replace=True)

    def display(self):
        G = nx.Graph()
        MFnodeNames=['MF{}'.format(i) for i in range(self.nMF)] ## very mysterious comprehension but works
        GCnodeNames=['GC{}'.format(i) for i in range(self.nGC)]
        center_MFL = (self.nMF)/2 - 1
        center_GCL = (self.nGC)/2 - 1
        for i in np.arange(self.nMF):
            G.add_node(MFnodeNames[i],pos=(0,(i-center_MFL)), color="red")

        for i in np.arange(self.nGC):
            G.add_node(GCnodeNames[i],pos=(.25,(i-center_GCL)))

        for gc in np.arange(self.nGC):
            for i in np.arange(4):
                G.add_edge(GCnodeNames[gc],MFnodeNames[self.connectionArray[gc,i]])


        nx.draw(G,nx.get_node_attributes(G,'pos'),node_size=10, alpha=0.4)
        plt.show()
        
    #I used this to test the plotting functions but I have tried as many solutions as I could find and i cannot get your plt.show() to show the graph of the connections and nodes
    plt.plot([1,2,3,4],[1,4,6,8], alpha=0.5)
    plt.plot([2,4,6,8],[3,6,9,12], color="green", alpha = 0.5)
    plt.ylabel('some numbers', color="red")
    plt.show()


nGC = 50
nMF = 10
MFTYPES=2

net = MFGC(nMF,nGC,MFTYPES)

