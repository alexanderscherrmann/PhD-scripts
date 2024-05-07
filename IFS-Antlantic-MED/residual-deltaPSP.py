import numpy as np
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
import pickle

pload = '/home/ascherrmann/010-IFS/ctraj/MED/use/'

CT = 'MED'

f = open(pload + 'PV-data-' + CT + 'dPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()
labs = helper.traced_vars_IFS()
dipv = data['dipv']

ylim = [0,1.0]
xlim = [0,400]

xal = np.array([])
yal = np.array([])

xend = np.array([])
yend = np.array([])
zbb = np.array([])
datadi = data['rawdata']

for q,date in enumerate(datadi.keys()):
    
    PVf = data['rawdata'][date]['PV']
    ZB = data['rawdata'][date]['ZB']
    datadi = data['rawdata']
    idp = np.where(PVf[:,0]>=0.75)[0]
    PV = np.mean(PVf[idp,:],axis=0)

    datadi[date]['DELTAPV'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['DELTAPV'][:,1:] = datadi[date]['PV'][:,:-1]-datadi[date]['PV'][:,1:]

    datadi[date]['PVRTOT'] = np.zeros(datadi[date]['PV'].shape)
    for pv in labs[8:]:
        datadi[date]['PVRTOT']+=datadi[date][pv]

    datadi[date]['newPVR'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['newPVR'][:,1:] = datadi[date]['PVRTOT'][:,1:]

    datadi[date]['locres'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['locres'][:,1:] = datadi[date]['newPVR'][:,1:] - datadi[date]['DELTAPV'][:,1:]
    datadi[date]['PSP'] = datadi[date]['PS']-datadi[date]['P']

    
    xal = np.append(xal,datadi[date]['PSP'][idp].flatten())
    yal = np.append(yal,datadi[date]['locres'][idp].flatten())
    zbb = np.append(zbb,ZB[idp].flatten())

    for k in range(len(idp)):
        xend = np.append(xend,np.min(datadi[date]['PSP'][idp[k]]))


xlab=r'$\Delta$ PS-P [hPa]'
fig, axes = plt.subplots()

nbin = 64

axes.set_xlabel(xlab)
axes.set_ylabel('residual [PVU]')

h = axes.hist2d(xal,abs(yal),range=[xlim,ylim],cmin=10,norm=matplotlib.colors.LogNorm(),bins=nbin,cmap='nipy_spectral') 
ticklabels = np.array([10,100,1000,1e4,1e5])
labels = np.array(['10','100',r'10$^3$',r'10$^4$'])
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h[3], ticks=ticklabels,cax=cbax)
func=resize_colorbar_vert(cbax, axes, pad=0.00, size=0.02)
fig.canvas.mpl_connect('draw_event', func)
ax =axes
ax.text(-0.1,0.98,'d)',transform=ax.transAxes,fontweight='bold',fontsize=12)
cbax.set_yticklabels(labels=labels,fontsize=10)
axes.tick_params(labelright=False,right=True)
name='Residual-PSP.png'
fig.savefig('/home/ascherrmann/010-IFS/' + name,dpi=300,bbox_inches="tight")
plt.close() 

ylim = [0,1.0]
xlim = [0,100]
nbin = 32
xlab=r'min. $\Delta$ PS-P [hPa]'
fig, axes = plt.subplots()

axes.set_xlabel(xlab,fontsize=8)
axes.set_ylabel('residual [PVU]')

#axes.hist2d(xend,yend,range=[xlim,ylim],cmin=5,norm=matplotlib.colors.LogNorm(),bins=nbin,cmap='nipy_spectral')

ticklabels = np.array([1,10,100,1000,1e4])
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h[3], ticks=ticklabels,cax=cbax)
func=resize_colorbar_vert(cbax, axes, pad=0.01, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

axes.tick_params(labelright=False,right=True)
name='Residual-minimal-PSP.png'
#fig.savefig('/home/ascherrmann/010-IFS/' + name,dpi=300,bbox_inches="tight")
plt.close()
        

fig, axes = plt.subplots()

nbin = 64
axes.set_xlabel('residual [PVU]')
axes.set_ylabel('elevation [m]')
h = axes.hist2d(abs(yal),zbb,range=[ylim,[0,2400]],cmin=10,norm=matplotlib.colors.LogNorm(),bins=nbin,cmap='nipy_spectral')
ticklabels = np.array([10,100,1000,1e4,1e5])
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h[3], ticks=ticklabels,cax=cbax)
func=resize_colorbar_vert(cbax, axes, pad=0.01, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

axes.tick_params(labelright=False,right=True)
name='Residuals-elevation.png'
fig.savefig('/home/ascherrmann/010-IFS/' + name,dpi=300,bbox_inches="tight")
plt.close()


