import numpy as np
from numpy import random
import networkx as nx
from matplotlib import pyplot as plt

random.seed(1)

DELTAT = 0.001
NSTEPS = 1000
Tau_fast = 20.0 / 1000.0
Tau_slow = 2.0
tauF = 15.0/1000


class Param:
    def __init__(self, type):
        self.type = type
        if self.type == 1:
            # driver tyoe
            # slow pool
            self.NSMAX = 3.5
            self.pS0 = 0.8
            self.DELTAFS = self.pS0
            self.pRefS = 0.6
            # fast pool
            self.NFMAX = 14
            self.pF0 = 0.6
            self.pRefF = 0
            self.DELTAFF = self.pF0
            self.Vmf = 200

        elif self.type == 2:
            #supporter type
            # slow pool
            self.NSMAX = 4
            self.pS0 = 0.4
            self.pRefS = 0.6
            self.DELTAFS = self.pS0
            # fast pool
            self.NFMAX = 6
            self.pF0 = 0.2
            self.pRefF = 0
            self.DELTAFF = self.pF0
            self.Vmf = 20

    def SteadyState(self):
            self.XiFast =((self.DELTAFF*self.Vmf*tauF+1)/((self.DELTAFF*self.Vmf**2*tauF*Tau_fast)+(self.DELTAFF*self.Vmf*tauF)+(self.Vmf*self.pF0*tauF)+1))
            self.XiSlow =((self.DELTAFS*self.Vmf*tauF+1)/((self.DELTAFS*self.Vmf**2*tauF*Tau_slow)+(self.DELTAFS*self.Vmf*tauF)+(self.Vmf*self.pS0*tauF)+1))
            self.PiFast =((self.pF0+self.DELTAFF*tauF*self.Vmf)/(1+self.DELTAFF*tauF*self.Vmf))
            self.PiSlow =((self.pS0+self.DELTAFS*tauF*self.Vmf)/(1+self.DELTAFS*tauF*self.Vmf))


Parameter1 = Param(1)
Parameter2 = Param(2)
print(Parameter1.NSMAX)


class GranuleCell:
    def __init__(self, mflist):
        self.MF = mflist
        self.response = np.zeros(NSTEPS, dtype=np.float)
        self.synapses = []
        for i in np.arange(4):
            self.synapses.append(Synapse(mflist[i]))

    def set_gain_and_threshold(self):
        pass


class Synapse:
    def __init__(self, T):
        self.X = np.zeros(shape=[2, NSTEPS], dtype=np.float)
        self.P = np.zeros(shape=[2, NSTEPS], dtype=np.float)
        self.type = T


class MFGC:
    def __init__(self, nMF, nGC, MFTYPES):
        self.nMF = nMF
        self.nGC = nGC
        self.MFTYPES = MFTYPES
        self.connectionArray = np.zeros(shape=[nGC, 4], dtype=np.int)
        self.MFTYPES = np.array([1, 2])

        labelMFDict = {}
        for loopMF in np.arange(nMF):
            labelMFDict.update({loopMF: np.random.choice(self.MFTYPES)})

        print(labelMFDict)
       
        self.wire()

    def wire(self):
        for gc in np.arange(self.nGC):
            self.connectionArray[gc, :] = np.random.choice(np.arange(nMF), size=4, replace=False)
        # print(self.connectionArray)

        # self.connectionArray=np.random.choice(np.arange(nMF),size=(nGC,4),replace=True)

    def display(self):
        G = nx.Graph()
        MFnodeNames = ['MF{}'.format(i) for i in range(self.nMF)]  ## very mysterious comprehension but works (
        GCnodeNames = ['GC{}'.format(i) for i in range(self.nGC)]
        center_MFL = (self.nMF) / 2 - 1
        center_GCL = (self.nGC) / 2 - 1
        for i in np.arange(self.nMF):
            G.add_node(MFnodeNames[i], pos=((i - center_MFL), 0), color="red")

        for i in np.arange(self.nGC):
            G.add_node(GCnodeNames[i], pos=((i - center_GCL), 1))

        for gc in np.arange(self.nGC):
            for i in np.arange(4):
                G.add_edge(GCnodeNames[gc], MFnodeNames[self.connectionArray[gc, i]])

        plt.ylim(-1, 1)
        nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=10, alpha=0.4)
        plt.show()

  


nGC = 50
nMF = 10
MFTYPES = 2

net = MFGC(nMF, nGC, MFTYPES)
net.display()


