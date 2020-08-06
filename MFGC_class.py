import numpy as np
from numpy import random
import networkx as nx
from matplotlib import pyplot as plt

random.seed(1)

DELTAT = 0.001
NSTEPS = 1000
Tau_fast = 20.0 / 1000.0 # recovery time scale of the fast pool
Tau_slow = 2.0           # recovery time scale of the slow pool
TauF = 15.0/1000         # facilitation time scale
TauG = 10.0/1000         # granule cell response time scale
Q0 = 1                   # Quantal response


class Param:
    def __init__(self, type):
        self.type = type
        if self.type == 0:
            # driver tyoe
            # slow pool
            self.NSMAX = 3.5
            self.pS0 = 0.8
            self.DELTAFS = self.pS0
            self.pRefS = 0.6
            # fast pool
            self.NFMAX = 14
            self.pF0 = 0.6
            self.DELTAFF = self.pF0
            self.pRefF = 0

            self.nuMF_mean = 200

        elif self.type == 1:
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

            self.nuMF_mean = 20

    def steady_state(self,nuMF):

        P_ss_f =((self.pF0+self.DELTAFF*TauF*nuMF)/(1+self.DELTAFF*TauF*nuMF))
        P_ss_s =((self.pS0+self.DELTAFS*TauF*nuMF)/(1+self.DELTAFS*TauF*nuMF))
        X_ss_f = 1/(1+P_ss_f*(1-self.pRefF)*nuMF*Tau_fast)
        X_ss_s = 1/(1+P_ss_s*(1-self.pRefS)*nuMF*Tau_slow)

        return(np.column_stack((X_ss_f,X_ss_s,P_ss_f,P_ss_s)))

Parameter1 = Param(0)
Parameter2 = Param(1)
parameters=[Parameter1,Parameter2]


class GCLayer:
    def __init__(self, nGC):
        self.response = np.zeros([nGC,NSTEPS])
        self.threshold = np.zeros(nGC)
        self.gain = np.zeros(nGC)


class SynapseLayer:
    def __init__(self, size, synapse_type_vec): #size should be 4xGC

        self.size = size
        # X and P are 3D arrays. The first index is the fast/slow pool index
        self.X = np.zeros(shape=[2,size, NSTEPS])
        self.P = np.zeros(shape=[2,size, NSTEPS])
        self.nuMF = np.zeros(shape=size)
        self.SS = np.zeros(shape=[size,4])
        self.types=synapse_type_vec

    def compute_steady_state(self):

        ## set steady state response for type 0 MF
        t0 = (self.types==0)
        t1 = (self.types==1)
        self.SS[t0,:] = parameters[0].steady_state(self.nuMF[t0])
        self.SS[t1,:] = parameters[1].steady_state(self.nuMF[t1])

    def generate_MF_rates(self):
        t0 = (self.types==0)
        t1 = (self.types==1)

        self.nuMF[t0] = np.random.gamma(10,parameters[0].nuMF_mean/10,size=np.sum(t0))
        self.nuMF[t1] = np.random.gamma(10,parameters[0].nuMF_mean/10,size=np.sum(t1))

    def combine_SS_input(self):
        nmax_fast = np.array([parameters[0].NFMAX,parameters[1].NFMAX])[self.types]
        nmax_slow = np.array([parameters[0].NSMAX,parameters[1].NSMAX])[self.types]
        npxq=(self.SS[:,0]*self.SS[:,2]*nmax_fast+self.SS[:,1]*self.SS[:,3]*nmax_slow)*Q0
        return(np.sum(np.reshape(npxq,[int(self.size/4),4]),1))

class MFGC:
    def __init__(self, nMF, nGC, MFTYPES):
        self.nMF = nMF
        self.nGC = nGC
        self.connectionArray = np.zeros(shape=[nGC, 4], dtype=int)
        self.synapse_type_vec = np.zeros(nGC*4,dtype=int)
        self.mf_type_vec = np.random.choice(np.arange(MFTYPES),size=nMF)

        self.wire()
        self.SL = SynapseLayer(4*nGC,self.synapse_type_vec)
        self.GCL = GCLayer(nGC)
        self.set_gain_and_threshold(1000)


    def wire(self):
        for gc in np.arange(self.nGC):
            self.connectionArray[gc] = np.random.choice(np.arange(nMF), size=4, replace=False)
            self.synapse_type_vec[gc*4+np.arange(4)] = self.mf_type_vec[self.connectionArray[gc]]

    ## Setting threshold and gain for all GC
    def set_gain_and_threshold(self,nsamples):
        sample_input_to_GC = np.zeros(shape=[self.nGC,nsamples])
        for i in np.arange(nsamples):
            self.SL.generate_MF_rates()
            self.SL.compute_steady_state()
            sample_input_to_GC[:,i] = self.SL.combine_SS_input()

        for gc in np.arange(self.nGC):
            self.GCL.threshold[gc] = np.quantile(sample_input_to_GC[gc],0.8)


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

SL = SynapseLayer(4,np.array([0,1,0,0]))
SL.generate_MF_rates()
SL.compute_steady_state()
SL.combine_SS_input()

