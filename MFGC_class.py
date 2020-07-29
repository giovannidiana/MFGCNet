# MFGC connection network

import numpy as np
from numpy import random
random.seed( 1 )


class MFGC:
    def __init__(self, nMF, nGC):
        self.nMF = nMF
        self.nGC = nGC

nGC = 5
nMF = 4*nGC
arangeGC = np.arange(0, nGC)
arangeMF = np.arange(0, nMF)


for loopGC in arangeGC:
    print(loopGC)
    connectionarray = np.zeros(0)
    n = 0
    while n < 4:
        GCMF = random.choice(arangeMF)
        connectionarray = np.append(connectionarray, GCMF)
        arangeMF = arangeMF[arangeMF != GCMF]
        n += 1
    print (connectionarray)
