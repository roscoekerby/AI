from pylab import *
import matplotlib.pyplot as plt
import os
import pyAgrum as gum

bn=gum.BayesNet('MontyHallOpens')

pd=bn.add(gum.LabelizedVariable('pd','prizedoor ?',3))
fp=bn.add(gum.LabelizedVariable('fp','firstpick ?',3))
mo=bn.add(gum.LabelizedVariable('mo','montyopens ?',3))

bn.addArc(pd,mo)
bn.addArc(fp,mo)

bn.cpt("pd").fillWith([0.33,0.33,0.33])
bn.cpt("fp").fillWith([0.33,0.33,0.33])

bn.cpt("mo")[{'pd':0,'fp':0}] = [0,0.5,0.5]
bn.cpt("mo")[{'pd':0,'fp':1}] = [0,0,1]
bn.cpt("mo")[{'pd':0,'fp':2}] = [0,1,0]
bn.cpt("mo")[{'pd':1,'fp':0}] = [0,0,1]
bn.cpt("mo")[{'pd':1,'fp':1}] = [0.5,0,0.5]
bn.cpt("mo")[{'pd':1,'fp':2}] = [1,0,0]
bn.cpt("mo")[{'pd':2,'fp':0}] = [0,1,0]
bn.cpt("mo")[{'pd':2,'fp':1}] = [1,0,0]
bn.cpt("mo")[{'pd':2,'fp':2}] = [0.5,0.5,0]


gum.saveBN(bn,"out/MHP.bif")

with open("out/MHP.bif","r") as out:
    print(out.read())

#print(bn)
