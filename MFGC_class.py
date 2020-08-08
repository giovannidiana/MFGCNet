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


class Param: #setting parameters depending on the type 
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
        #nuMF is empty rn because it will take the values for the firing rate later on 
        P_ss_f =((self.pF0+self.DELTAFF*TauF*nuMF)/(1+self.DELTAFF*TauF*nuMF))
        P_ss_s =((self.pS0+self.DELTAFS*TauF*nuMF)/(1+self.DELTAFS*TauF*nuMF))
        X_ss_f = 1/(1+P_ss_f*(1-self.pRefF)*nuMF*Tau_fast)
        X_ss_s = 1/(1+P_ss_s*(1-self.pRefS)*nuMF*Tau_slow)

        return(np.column_stack((X_ss_f,X_ss_s,P_ss_f,P_ss_s)))
        #converts the steady state equations to columns which can be called upon by their vectors 

Parameter1 = Param(0)
Parameter2 = Param(1)
parameters=[Parameter1,Parameter2]


class GCLayer:
    def __init__(self, nGC):
        self.response = np.zeros([nGC,NSTEPS])
        self.threshold = np.zeros(nGC) #theta for each Granule cell to fire 
        self.gain = np.full(nGC,1)
        #It is useful to have layers as you can do all the calculations in their relevant layers rather than looping through each one 
        #this sets a response array with the structure number of GC cells: numbmber of time interval steps

class SynapseLayer:
    def __init__(self, size, synapse_type_vec): #size should be 4xGC because each granule cell has 4 synapses from MF

        self.size = size
        # X and P are 3D arrays. The first index is the fast/slow pool index
        self.X = np.zeros(shape=[2,size, NSTEPS]) #[two rows for fast slow pool, size is collums 4*GC for each synapse, NSTEPS is depth for each interval]
        self.P = np.zeros(shape=[2,size, NSTEPS])
        self.nuMF = np.zeros(shape=size) #setting the array to be the size of each granule cells synapses to take the firing rates 
        self.SS = np.zeros(shape=[size,4]) #for each row of synapses there are 4 columns which are each of the steady state values 
        self.types=synapse_type_vec #setting each synapse into a vector type to do all the equations at once 

    def compute_steady_state(self):

        ## set steady state response for type 0 MF
        t0 = (self.types==0)
        t1 = (self.types==1)
        self.SS[t0,:] = parameters[0].steady_state(self.nuMF[t0])#insert into the colum the steady state for each synapse of a type 0 MF
        self.SS[t1,:] = parameters[1].steady_state(self.nuMF[t1])

    def combine_SS_input(self):
        nmax_fast = np.array([parameters[0].NFMAX,parameters[1].NFMAX])[self.types] #creating an array for the fast pool with the max for each type
        nmax_slow = np.array([parameters[0].NSMAX,parameters[1].NSMAX])[self.types]
        npxq=(self.SS[:,0]*self.SS[:,2]*nmax_fast+self.SS[:,1]*self.SS[:,3]*nmax_slow)*Q0 
        #combines the ss for collumn 0 which is X_ss_f then * c2 P_ss_f * fast NSMAX + then c1: X_ss_s and c3: P_ss_s* slow NSMAX * current per one vesicle 
        #is the h (vesicle release) = [p1*n1(max)+p2*n2(max)]Q0*input freq 
        return(np.sum(np.reshape(npxq,[int(self.size/4),4]),1))

    def integrate(self,pattern): # pattern is the array of MF firing rates



class MFGC:
    def __init__(self, nMF, nGC, MFTYPES):
        self.nMF = nMF
        self.nGC = nGC
        self.connectionArray = np.zeros(shape=[nGC, 4], dtype=int)
        self.synapse_type_vec = np.zeros(nGC*4,dtype=int)
        self.mf_type_vec = np.random.choice(np.arange(MFTYPES),size=nMF)#randomly assigning a type to each MF
        self.nuMF = np.zeros(nMF) #assigning the input freq an array size of the MF  

        self.wire()
        self.SL = SynapseLayer(4*nGC,self.synapse_type_vec) #creates the synapses with the corresponding vector for each synapse type
        self.GCL = GCLayer(nGC)#setting it up to have these layers in parallel 
        self.set_gain_and_threshold(1000)

    def generate_pattern(self):
        t0 = (self.mf_type_vec==0)
        t1 = (self.mf_type_vec==1)

        self.nuMF[t0] = np.random.gamma(10,parameters[0].nuMF_mean/10,size=np.sum(t0))
        self.nuMF[t1] = np.random.gamma(10,parameters[1].nuMF_mean/10,size=np.sum(t1))
        #gamma distrobution is always positive, allows each input freq to be positive 
        #shape = 10, shape 1 is very skewed to the left as shape increases skedness decreased
        #scale = parameters[0].nuMF_mean/10 , has the effect of stretching or compressing the range of the Gamma distribution. in this case takes the mean of the input freqs/10 as the distrobution
        #size - the values given - Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. 

        for gc in np.arange(self.nGC):
            self.SL.nuMF[gc*4+np.arange(4)]=self.nuMF[self.connectionArray[gc]]
            #for each granule cell: for each nuMF in the synapse layer give the 4 values of nuMF depending on which one it has been assigned by the random connection array
            #the +4 skipps over the values that have already been assigned values 
            ##setting the array to be the size of each granule cells synapses to take the firing rates 
            #the [self.connectionArray[gc]] is a format that adds the whole row

    def wire(self):
        for gc in np.arange(self.nGC):
            self.connectionArray[gc] = np.random.choice(np.arange(nMF), size=4, replace=False) #this assigns the random 4 MFs to each granule cell
            self.synapse_type_vec[gc*4+np.arange(4)] = self.mf_type_vec[self.connectionArray[gc]] #same start of skipping over values that have already been assigned 
            #for each granule cell assign the mf types from each randomly assigned MF in a row (which row not entirely sure about?)
            # in the synapse type vector add the types for each MF type 

    ## Setting threshold and gain for all GC
    def set_gain_and_threshold(self,nsamples):
        sample_input_to_GC = np.zeros(shape=[self.nGC,nsamples])
        for i in np.arange(nsamples):
            self.generate_pattern()
            self.SL.compute_steady_state()
            sample_input_to_GC[:,i] = self.SL.combine_SS_input()

        for gc in np.arange(self.nGC):
            self.GCL.threshold[gc] = np.quantile(sample_input_to_GC[gc],0.8)

    def integrate(self):
        pass


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

