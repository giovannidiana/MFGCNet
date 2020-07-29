# MFGC connection network

import numpy as np
from numpy import random
import networkx as nx
import matplotlib.pyplot as plt

random.seed( 1 )


class MFGC:
    def __init__(self, nMF, nGC, MFTYPES):
        self.nMF = nMF
        self.nGC = nGC
        self.MFTYPES = MFTYPES
        self.connectionArray = np.zeros(shape=[nGC,4],dtype=np.int)

        ## assign MF labels, please

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
            G.add_node(MFnodeNames[i],pos=(0,(i-center_MFL)))

        for i in np.arange(self.nGC):
            G.add_node(GCnodeNames[i],pos=(.5,(i-center_GCL)))

        for gc in np.arange(self.nGC):
            for i in np.arange(4):
                G.add_edge(GCnodeNames[gc],MFnodeNames[self.connectionArray[gc,i]])


        nx.draw(G,nx.get_node_attributes(G,'pos'),node_size=10)
        plt.show()


nGC = 50
nMF = 10
MFTYPES=2

net = MFGC(nMF,nGC,MFTYPES)

