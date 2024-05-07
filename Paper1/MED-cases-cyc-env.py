import numpy as np
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import pickle

parser = argparse.ArgumentParser(description="plot accumulated average PV gain that is associated with the cyclone and the environment")
parser.add_argument('rdis',default='',type=int,help='distance from center in km that should be considered as cyclonic')
args = parser.parse_args()

p = '/atmosdyn2/ascherrmann/009-ERA-5/' + 'MED/cases/'
traced = np.array([])
for d in os.listdir(p):
    if(d.startswith('trajectories-mature')):
            traced = np.append(traced,d)
traced = np.sort(traced)

fsl=4

#f = open(p + 'lats-Manos.txt','rb')
#clat = pickle.load(f)
#f.close()

#f = open(p + 'lons-Manos.txt','rb')
#clon = pickle.load(f)
#f.close()

#f = open(p + 'dates-Manos.txt','rb')
#dates = pickle.load(f)
#f.close()


clon = []
clat = []
d1 = np.loadtxt(p + 'medicane-data-20051214_04.txt')
d2 = np.loadtxt(p + 'medicane-zorbas-data-20180928_05.txt')
clon.append(np.flip(d1[:,0]))
clon.append(np.flip(d2[:,0]))
clat.append(np.flip(d1[:,1]))
clat.append(np.flip(d2[:,1]))

dates = []
da1 = np.array([])
da2 = np.array([])
date1 = '20051214_04'
date2 = '20180928_05'

yyyy = int(date1[0:4])
MM = int(date1[4:6])
DD = int(date1[6:8])
hh = int(date1[9:])
for k in range(0,len(d1[:,0])):
        if (hh<0):
            hh=23
            DD-=1
            if(DD<1):
                MM-=1
                DD=helper.month_days(yyyy)[int(MM)-1]
        da1 = np.append(da1,str(yyyy)+'%02d%02d_%02d'%(MM,DD,hh))
        hh-=1
yyyy = int(date2[0:4])
MM = int(date2[4:6])
DD = int(date2[6:8])
hh = int(date2[9:])
for k in range(0,len(d2[:,0])):
        if (hh<0):
            hh=23
            DD-=1
            if(DD<1):
                MM-=1
                DD=helper.month_days(yyyy)[int(MM)-1]
        da2 = np.append(da2,str(yyyy)+'%02d%02d_%02d'%(MM,DD,hh))
        hh-=1
print(len(da2))
print(len(d2[:,0]))

dates.append(da1)
dates.append(da2)

#labs = helper.traced_vars_ERA5()
labs = ['time','lon','lat','P','PV']
ptitle = np.array(['700 hPa < P$_{0}$', 'P$_{0} <$ 700 hPa', '400 < P$_{-48} <$ 600 hPa',  'P$_{-48} <$ 400 hPa'])
linestyle = ['-',':']

#LON=np.linspace(-120,30,301)
#LAT=np.linspace(-20,90,221)
LON=np.linspace(-180,180,721)
LAT=np.linspace(-90,90,361)

rdis = int(args.rdis)
#deltaLONLAT = helper.convert_radial_distance_to_lon_lat_dis(rdis)

### INFO
### Trajetories start 200km around the center between 925 and 600 hPa if PV>0.6 PVU
###

wql = 0
meandi = dict()
env = 'env'
cyc = 'cyc'
split=[cyc,env]

datadi = dict() ####raw data of traced vars
dipv = dict() ####splited pv is stored here
dit = dict() ### 1 and 0s, 1 if traj is close to center at given time, 0 if not
meandi[env] = dict()
meandi[cyc] = dict()

H = 48
xlim = np.array([-1*H,0])
ylim = np.array([-0.3,1.0])
ylim2 = np.array([-0.3,1.6])
pressure_stack = np.zeros(H+1)

linewidth=1.5
alpha=1.

f = open('/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/PV-data-dPSP-100-ZB-800-2-400-correct-distance-noro.txt','rb')
data = pickle.load(f)
f.close()

dipv = data['dipv']
oro = data['oro']
datadi = data['rawdata']
noro = data['noro']

for k in dipv.keys():
    if k!='108215' and k!='119896':
        continue
    fig, axes = plt.subplots(1,1,figsize=(8,6))#1,2,sharex=True, sharey=True)
    cycID=k
    ### PLOTTING
    t = datadi[cycID]['time'][0]
    if True:
                ax = axes
                ax.plot([],[],marker=None,ls='-',color='black',alpha=0.4)
                ax.plot([],[],marker=None,ls='-',color='red',alpha=0.4)
                ax.plot([],[],marker=None,ls='-',color='black',alpha=1.0)
                ax.plot([],[],marker=None,ls=':',color='black',alpha=1.0)
                ax.plot([],[],marker=None,ls=':',color='grey',alpha=1.0)

                idp = np.where((datadi[cycID]['PV'][:,0]>=0.75) & (datadi[cycID]['P'][:,0]>=700))
                axes.plot(t,np.mean(datadi[cycID]['PV'][idp],axis=0),color='black',alpha=0.4)
                axes.plot(t,np.mean(datadi[cycID]['PV'][idp],axis=0)-np.mean(datadi[cycID]['PV'][idp,-1]),color='red',alpha=0.4)
                ax = axes

                for pl,key in enumerate(['cyc']):#,'env']):
                    meantmp = np.mean(dipv[cycID][key][idp],axis=0)
                    ax.plot(t,meantmp,color='k',ls=linestyle[pl])
                ax.axvline(-15,color='grey',ls='-')
                ax.plot(t,np.mean(noro[cycID]['env'][idp],axis=0),ls=':',color='k')
                ax.plot(t,np.mean(oro[cycID]['env'][idp],axis=0),ls=':',color='grey')
    axes.set_xlim(xlim)
    axes.set_xticks(ticks=np.arange(-48,1,6))
    axes.set_ylim(ylim2)
    ax.legend([r'PV$(t)$',r'APV$(t)$','APVC$(t)$', 'APVENO$(t)$','APVEO$(t)$'],frameon=False,loc='upper left')
    axes.tick_params(labelright=False,right=True)
    axes.set_xlabel('time until mature stage [h]') 
    axes.set_ylabel('PV [PVU]')
    axes.text(-0.135,0.875,'(b)',transform=ax.transAxes,fontsize=16)
    if k=='119896':
        n=9
    else:
        n=10
    name = 'fig%02d.png'%n
    p = '/home/ascherrmann/publications/cyclonic-environmental-pv/'
    fig.savefig(p + name,dpi=300,bbox_inches="tight")
    plt.close()

