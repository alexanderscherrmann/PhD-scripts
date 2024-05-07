import numpy as np
import pickle

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patch
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm

pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
PVdata = pickle.load(f)
f.close()

dipv = PVdata['dipv']
datadi = PVdata['rawdata']
dit = PVdata['dit']

both = np.array([])
adv = np.array([])
cyclonic = np.array([])
environmental = np.array([])

savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()

SLP = var[0]
lon = var[1]
lat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
dates = var[5]
avaID = np.array([])
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))

adv = np.array([])
cyc = np.array([])
env = np.array([])
c = 'cyc'
e = 'env'

ac = dict()
pressuredi = dict()

PVstart = np.array([])
PVend = np.array([])
ca = 0
ct = 0
ol = np.array([])
ol2 = np.array([])
pvloc = dict()
cycd = dict()
envd = dict()

for h in np.arange(0,49):
    pvloc[h] = np.array([])
    cycd[h] = np.array([])
    envd[h] = np.array([])

slps = np.array([])
envs = np.array([])
cycs = np.array([])
timeidentified = np.array([])
coniv = np.arange(-40,101,5)
condic = dict()
condie = dict()

slpdic = dict()
slpdie = dict()
for co in coniv[:-1]:
    condic[co] = np.array([])
    slpdic[co] = np.array([])
    condie[co] = np.array([])
    slpdie[co] = np.array([])

for ll,k in enumerate(dipv.keys()):
    q = np.where(avaID==int(k))[0][0]
    d = k
    ac[d] = dict()
    pressuredi[d] = dict()
    for h in np.arange(-48,49):
        ac[d][h] = np.array([])
        pressuredi[d][h] = np.array([])

#    if (hourstoSLPmin[q][0]>-6):
#        continue

    PV = PVdata['rawdata'][d]['PV']
    i = np.where(PV[:,0]>=0.75)[0]

    pvend = PV[i,0]
    pvstart = PV[i,-1]
    PVstart = np.append(PVstart,pvstart)
    PVend = np.append(PVend,pvend)
    
    cypv = dipv[d][c][i,0]
    enpv = dipv[d][e][i,0]
    cy = np.mean(cypv)

    adv = np.append(adv,(pvstart)/pvend)
    cyc = np.append(cyc,cypv/pvend)
    env = np.append(env,enpv/pvend)
    envs = np.append(envs,np.mean(enpv/pvend))
    cycs = np.append(cycs,np.mean(cypv/pvend))
    slps = np.append(slps,SLP[q][abs(hourstoSLPmin[q][0]).astype(int)])
    timeidentified = np.append(timeidentified,hourstoSLPmin[q][0])

    for rr, co in enumerate(coniv[:-1]):
        if helper.where_greater_smaller(np.mean(enpv/pvend)*100,co,coniv[rr+1]).size:
#            condie[co] = np.append(condie[co],np.mean(enpv/pvend)*100)
            slpdie[co] = np.append(slpdie[co],SLP[q][abs(hourstoSLPmin[q][0]).astype(int)])

        if helper.where_greater_smaller(np.mean(cypv/pvend)*100,co,coniv[rr+1]).size:
#            condic[co] = np.append(condic[co],np.mean(cypv/pvend)*100)
            slpdic[co] = np.append(slpdic[co],SLP[q][abs(hourstoSLPmin[q][0]).astype(int)])

enmed = np.array([])
cymed = np.array([])
for co in coniv[:-1]:
    enmed = np.append(enmed,np.mean(np.sort(slpdie[co])))
    cymed = np.append(cymed,np.median(np.sort(slpdic[co])))
        

fig,ax = plt.subplots()
ax.scatter(envs*100,slps,marker='.',color='k')
ax.scatter(coniv[:-1],enmed,color='r')
ax.set_ylim(965,1010)
ax.set_xlim(-40,100)
ax.set_ylabel('minimal SLP [hPa]')
ax.set_xlabel(r'env. contribution to PVend [%]')
fig.savefig('/home/ascherrmann/009-ERA-5/MED/environmental-SLP.png',dpi=300,bbox_inches='tight')
plt.close('all')


fig,ax = plt.subplots()
ax.scatter(cycs*100,slps,marker='.',color='k')
ax.scatter(coniv[:-1],cymed,color='r')
ax.set_ylim(965,1010)
ax.set_xlim(-40,100)
ax.set_ylabel('minimal SLP [hPa]')
ax.set_xlabel(r'cyc. contribution to PVend [%]')
fig.savefig('/home/ascherrmann/009-ERA-5/MED/cyclonic-SLP.png',dpi=300,bbox_inches='tight')
plt.close('all')


fig,ax = plt.subplots()
ax.scatter(timeidentified,envs*100,marker='.',color='k')
ax.set_xlim(-48,0)
ax.set_xticks(ticks = np.arange(-48,1,6))
ax.set_ylim(-40,100)
ax.set_xlabel('time of first track poing from mature stage [h]')
ax.set_ylabel(r'env. contribution to PVend [%]')
fig.savefig('/home/ascherrmann/009-ERA-5/MED/environmental-identify.png',dpi=300,bbox_inches='tight')
plt.close('all')


fig,ax = plt.subplots()
ax.scatter(timeidentified,cycs*100,marker='.',color='k')
ax.set_xlim(-48,0)
ax.set_xticks(ticks = np.arange(-48,1,6))
ax.set_ylim(-40,100)
ax.set_xlabel('time of first track poing from mature stage [h]')
ax.set_ylabel(r'cyc. contribution to PVend [%]')
fig.savefig('/home/ascherrmann/009-ERA-5/MED/cyclonic-identify.png',dpi=300,bbox_inches='tight')
plt.close('all')
