import numpy as np
from numpy import random
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd

random.seed(1)

DELTAT = 0.001
NSTEPS = 1000
Tau_fast = 20.0 / 1000.0 # recovery time scale of the fast pool
Tau_slow = 2.0           # recovery time scale of the slow pool
TauF = 15.0/1000         # facilitation time scale
TauG = 10.0/1000         # granule cell response time scale
Q0 = 1                   # Quantal response
FinalTime = 400

## Runge-Kutta integrator
def rk4_syn(dxdt,dpdt,t0, x0,p0, t1, n):
    vt = np.zeros(n+1)
    vx = np.zeros(shape=[n+1,len(x0)])
    vp = np.zeros(shape=[n+1,len(p0)])
    h = (t1 - t0) / float(n)
    vt[0] = t = t0
    vx[0] = x = x0
    vp[0] = p = p0
    for i in np.arange(1, n + 1):
        k1x = h *dxdt(t, x, p)
        k1p = h *dpdt(t, p)
        
        k2x = h * dxdt(t + 0.5 * h, x + 0.5 * k1x, p + 0.5 * k1p)
        k2p = h * dpdt(t + 0.5 * h, p + 0.5 * k1p)

        k3x = h * dxdt(t + 0.5 * h, x + 0.5 * k2x, p + 0.5 * k2p)
        k3p = h * dpdt(t + 0.5 * h, p + 0.5 * k2p)
        
        k4x = h * dxdt(t + h, x + k3x, p + k3p)
        k4p = h * dpdt(t + h, p + k3p)

        vt[i] = t = t0 + i * h
        vx[i] = x = x + (k1x + k2x + k2x + k3x + k3x + k4x) / 6
        vp[i] = p = p + (k1p + k2p + k2p + k3p + k3p + k4p) / 6
    
    return(vx,vp)

def rk4_G(dgdt,t0, g0, t1, n):
    vt = np.zeros(n+1)
    vg = np.zeros(shape=[n+1,len(g0)])
    h = (t1 - t0) / float(n)
    vt[0] = t = t0
    vg[0] = g = g0
    for i in np.arange(1, n + 1):
        k1g = h * dgdt(t, g)
        k2g = h * dgdt(t + 0.5 * h, g + 0.5 * k1g)
        k3g = h * dgdt(t + 0.5 * h, g + 0.5 * k2g)
        k4g = h * dgdt(t + h, g + k3g)

        vt[i] = t = t0 + i * h
        vg[i] = g = g + (k1g + k2g + k2g + k3g + k3g + k4g) / 6
    
    return(vg)

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
        self.response = np.zeros([NSTEPS,nGC])
        self.threshold = np.zeros(nGC)
        self.gain = np.full(nGC,1)

class SynapseLayer:
    def __init__(self, size, synapse_type_vec): #size should be 4xGC

        self.size = size
        # X and P are 3D arrays. The first index is the fast/slow pool index
        self.X = np.zeros(shape=[2, NSTEPS+1, size])
        self.P = np.zeros(shape=[2, NSTEPS+1, size])
        self.nuMF = np.zeros(shape=size)
        self.SS = np.zeros(shape=[size,4])
        self.types=synapse_type_vec

    def compute_steady_state(self):

        ## set steady state response for type 0 MF
        t0 = (self.types==0)
        t1 = (self.types==1)
        self.SS[t0,:] = parameters[0].steady_state(self.nuMF[t0])
        self.SS[t1,:] = parameters[1].steady_state(self.nuMF[t1])

    def combine_SS_input(self):
        nmax_fast = np.array([parameters[0].NFMAX,parameters[1].NFMAX])[self.types]
        nmax_slow = np.array([parameters[0].NSMAX,parameters[1].NSMAX])[self.types]
        npxq=(self.SS[:,0]*self.SS[:,2]*nmax_fast+self.SS[:,1]*self.SS[:,3]*nmax_slow)*Q0
        return(np.sum(np.reshape(npxq,[int(self.size/4),4]),1))

    # this function is missing in the current github version...
    def combine_input(self):
        nmax_fast = np.array([parameters[0].NFMAX,parameters[1].NFMAX])[self.types]
        nmax_slow = np.array([parameters[0].NSMAX,parameters[1].NSMAX])[self.types]
        npxq=(self.X[0]*self.P[0]*nmax_fast+self.X[1]*self.P[1]*nmax_slow)*Q0
        self.total_input=npxq.reshape(-1,self.size/4,4).sum(axis=2)

    def dxdt_fast(self,t,x,p):
        return((1-x)/Tau_fast - p*x*(1-self.param_data_frame['pRefF'])*self.SL.nuMF)
    def dxdt_slow(self,t,x,p):
        return((1-x)/Tau_slow - p*x*(1-self.param_data_frame['pRefS'])*self.SL.nuMF)
    def dpdt_fast(self,t,p):
        return((self.param_data_frame['pS0']-p)/TauF + self.param_data_frame['pS0']*(1-p)*self.SL.nuMF)
    def dpdt_slow(self,t,p):
        return((self.param_data_frame['pF0']-p)/TauF + self.param_data_frame['pF0']*(1-p)*self.SL.nuMF)
    def dgdt(self,t,g):
        n=int(np.floor(t/FinalTime*NSTEPS))
        return(-g/TauG + self.GL.gain*np.maximum(0,self.total_input[n]-self.GL.threshold))
        





class MFGC:
    def __init__(self, nMF, nGC, MFTYPES):
        self.nMF = nMF
        self.nGC = nGC
        self.connectionArray = np.zeros(shape=[nGC, 4], dtype=int)
        self.synapse_type_vec = np.zeros(nGC*4,dtype=int)
        self.mf_type_vec = np.random.choice(np.arange(MFTYPES),size=nMF)
        self.nuMF = np.zeros(nMF)

        self.wire()
        self.SL = SynapseLayer(4*nGC,self.synapse_type_vec)
        self.GCL = GCLayer(nGC)
        self.set_gain_and_threshold(1000)

    def generate_pattern(self):
        t0 = (self.mf_type_vec==0)
        t1 = (self.mf_type_vec==1)

        self.nuMF[t0] = np.random.gamma(10,parameters[0].nuMF_mean/10,size=np.sum(t0))
        self.nuMF[t1] = np.random.gamma(10,parameters[1].nuMF_mean/10,size=np.sum(t1))

        for gc in np.arange(self.nGC):
            self.SL.nuMF[gc*4+np.arange(4)]=self.nuMF[self.connectionArray[gc]]

    def wire(self):
        for gc in np.arange(self.nGC):
            self.connectionArray[gc] = np.random.choice(np.arange(nMF), size=4, replace=False)
            self.synapse_type_vec[gc*4+np.arange(4)] = self.mf_type_vec[self.connectionArray[gc]]

    ## Setting threshold and gain for all GC
    def set_gain_and_threshold(self,nsamples):
        sample_input_to_GC = np.zeros(shape=[self.nGC,nsamples])
        for i in np.arange(nsamples):
            self.generate_pattern()
            self.SL.compute_steady_state()
            sample_input_to_GC[:,i] = self.SL.combine_SS_input()

        for gc in np.arange(self.nGC):
            self.GCL.threshold[gc] = np.quantile(sample_input_to_GC[gc],0.8)

    def set_synaptic_data_frame(self):
        parnames = ["type","pS0","pF0","pRefS","pRefF","nuMF"]
        self.param_data_frame = pd.DataFrame(np.zeros(shape=[self.SL.size,len(parnames)]),columns=parnames)

        self.param_data_frame['type'] = self.synapse_type_vec
        self.param_data_frame['pS0'] = np.array([parameters[0].pS0,parameters[1].pS0])[self.param_data_frame['type']]
        self.param_data_frame['pF0'] = np.array([parameters[0].pF0,parameters[1].pF0])[self.param_data_frame['type']]
        self.param_data_frame['pRefS'] = np.array([parameters[0].pRefS,parameters[1].pRefS])[self.param_data_frame['type']]
        self.param_data_frame['pRefF'] = np.array([parameters[0].pRefF,parameters[1].pRefF])[self.param_data_frame['type']]
        self.param_data_frame['nuMF'] = self.SL.nuMF

    def integrate(self): # pattern is the array of MF firing rates
        XF_init = np.full(shape=self.SL.size,fill_value=1)
        XS_init = np.full(shape=self.SL.size,fill_value=1)
        PF_init = self.param_data_frame['pF0']
        PS_init = self.param_data_frame['pS0']

        self.SL.X[0], self.SL.P[0] = rk4_syn(self.dxdt_fast,dpdt_fast,0, XF_init,PF_init,FinalTime,NSTEPS) 
        self.SL.X[1], self.SL.P[1] = rk4_syn(self.dxdt_slow,dpdt_slow,0, XS_init,PS_init,FinalTime,NSTEPS) 
        self.SL.combine_input()

        ## Instead of using a zero array we should compute the steady state of G
        self.GC.response = rk4_G(self.dgdt,0, np.zeros(self.nGC), FinalTime, NSTEPS)

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
SL.generate_pattern()
SL.compute_steady_state()
SL.combine_SS_input()

