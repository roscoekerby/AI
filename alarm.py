from pylab import *
import matplotlib.pyplot as plt
import os
import pyAgrum as gum

bn=gum.BayesNet('alarm')

a=bn.add(gum.LabelizedVariable('a', 'alarm ?', 2))

b=bn.add(gum.LabelizedVariable('b', 'burglary ?', 2))
e=bn.add(gum.LabelizedVariable('e', 'earthquake ?', 2))

jc=bn.add(gum.LabelizedVariable('j', 'johncalls ?', 2))
mc=bn.add(gum.LabelizedVariable('m', 'marycalls ?', 2))

bn.addArc(b,a)
bn.addArc(e,a)
bn.addArc(a,jc)
bn.addArc(a,mc)

bn.cpt("b").fillWith([0.999,0.001])

bn.cpt("e").fillWith([0.998,0.002])

bn.cpt("j")[{'a':1}] = [0.1,0.9]
bn.cpt("j")[{'a':0}] = [0.95,0.05]

bn.cpt("m")[{'a':1}] = [0.3,0.7]
bn.cpt("m")[{'a':0}] = [0.99,0.01]

bn.cpt("a")[{'b':1,'e':1}] = [0.05,0.95]
bn.cpt("a")[{'b':1,'e':0}] = [0.06, 0.94]
bn.cpt("a")[{'b':0,'e':1}] = [0.71,0.29]
bn.cpt("a")[{'b':0,'e':0}] = [0.999,0.001]

gum.saveBN(bn,"out/alarm.bif")

with open("out/alarm.bif","r") as out:
    print(out.read())

