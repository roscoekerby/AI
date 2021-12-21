from pylab import *
import matplotlib.pyplot as plt
import os
import pyAgrum as gum
#import pyAgrum.lib.notebook as gnb

bn=gum.fastBN("Cloudy?->Sprinkler?->Wet Grass?<-Rain?<-Cloudy?")

bn.cpt("Cloudy?").fillWith([0.5,0.5])

bn.cpt("Sprinkler?")[:]=[[0.5,0.5],
                         [0.9,0.1]]

bn.cpt("Rain?")[{'Cloudy?':0}]=[0.8,0.2]
bn.cpt("Rain?")[{'Cloudy?':1}]=[0.2,0.8]

bn.cpt("Wet Grass?")[{'Rain?': 0, 'Sprinkler?': 0}] = [1, 0]
bn.cpt("Wet Grass?")[{'Rain?': 0, 'Sprinkler?': 1}] = [0.1, 0.9]
bn.cpt("Wet Grass?")[{'Rain?': 1, 'Sprinkler?': 0}] = [0.1, 0.9]
bn.cpt("Wet Grass?")[{'Rain?': 1, 'Sprinkler?': 1}] = [0.01, 0.99]

gum.config['notebook','potential_visible_digits']=2
print(bn)
#gnb.sideBySide(bn.cpt("Cloudy?"),captions=['$P(Cloudy)$'])
#gnb.sideBySide(
#  gnb.getSideBySide(bn.cpt("Sprinkler?"),captions=['$P(Sprinkler|Cloudy)$']),
#  gnb.getBN(bn,size="3!"),
#  gnb.getSideBySide(bn.cpt("Rain?"),captions=['$P(Rain|Cloudy)$']))
#gnb.sideBySide(bn.cpt("Wet Grass?"),captions=['$P(WetGrass|Sprinkler,Rain)$'])
