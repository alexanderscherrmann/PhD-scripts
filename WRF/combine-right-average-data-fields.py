import pickle
import numpy as np
f = open('moderate-average-fields.txt','rb')
md = pickle.load(f)
f.close()
f = open('new-weak-intense-fields.txt','rb')
nwi = pickle.load(f)
f.close()
f = open('most-intense-moderate-weak-average-fields.txt','rb')
gd = pickle.load(f)
f.close()
                
newdict = dict()                
for sea in md.keys():
    newdict[sea] = dict()
    for wi in gd[sea].keys():
        newdict[sea][wi] = dict()
        for ll in gd[sea][wi].keys():
            newdict[sea][wi][ll] = dict()
            for we in md[sea]['moderate-cyclones.csv'][ll].keys():
                newdict[sea][wi][ll][we] = dict()
                
                
for sea in md.keys():
    for wi in ['weak-cyclones.csv','intense-cyclones.csv']:
        for ll in gd[sea][wi].keys():
            for we in newdict[sea][wi][ll].keys():
                for var in nwi[sea][wi][ll][we].keys():
                    newdict[sea][wi][ll][we][var] = nwi[sea][wi][ll][we][var]
                for var in gd[sea][wi][ll][we].keys():
                    newdict[sea][wi][ll][we][var] = gd[sea][wi][ll][we][var]


newdict[sea][wi][ll][we][var]
for sea in md.keys():
    for wi in md[sea].keys():
        for ll in md[sea][wi].keys():
            for we in newdict[sea][wi][ll].keys():
                for var in md[sea][wi][ll][we].keys():
                    newdict[sea][wi][ll][we][var] = md[sea][wi][ll][we][var]


for sea in md.keys():
    for wi in gd[sea].keys():
        for ll in gd[sea][wi].keys():
            for we in newdict[sea][wi][ll].keys():
                for wcb in ['wcbascfreq','wcbout500freq','wcbout400freq']:
                    newdict[sea][wi][ll][we][wcb] *=ll
                for oth in ['SLPcycfreq','omega500','MSL']:
                    newdict[sea][wi][ll][we][oth] /=ll


f = open('new-most-intense-moderate-weak-average-fields.txt','wb')
pickle.dump(newdict,f)
f.close()
