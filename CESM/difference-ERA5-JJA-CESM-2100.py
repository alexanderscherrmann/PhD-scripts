from netCDF4 import Dataset as ds
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy
import numpy as np
import wrf

wrfd = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pappath = '/atmosdyn2/ascherrmann/015-CESM-WRF/'

# delts SLP, delta U300, delta T900, delta SST

sim1='JJA-clim/'
sim2='CESM-2100-DJF-clim/'

di = dict()
for sim in [sim1,sim2]:
    tmp = ds(wrfd + sim + 'wrfout_d01_2000-12-01_00:00:00','r')
    lon = wrf.getvar(tmp,'lon')[0]
    lat = wrf.getvar(tmp,'lat')[:,0]
    slp = wrf.getvar(tmp,'slp')
    U = wrf.getvar(tmp,'U')
    U = (U[:,:,:-1] + U[:,:,1:])/2
    P = wrf.getvar(tmp,'pres')/100
    u300 = wrf.interplevel(U,P,300,meta=False)
    sst = wrf.getvar(tmp,'SST')
    TH = wrf.getvar(tmp,'th')
    th900 = wrf.interplevel(TH,P,900,meta=False)
    di[sim]=dict()
    di[sim]['TH'] = th900[:]
    di[sim]['SLP'] = slp[:]
    di[sim]['SST'] = sst[:]
    di[sim]['U'] = u300[:]
    tmp.close()

fig = plt.figure(figsize=(12,12))
gs = gridspec.GridSpec(nrows=2, ncols=2)

var = ['TH','SST','SLP','U']
levels = [np.arange(-5,5.1,1),np.arange(-4,4.1,0.8),np.arange(-15,16,3),np.arange(-10,11,2)]
cmaps = [matplotlib.cm.seismic,matplotlib.cm.coolwarm,matplotlib.cm.BrBG,matplotlib.cm.PiYG]
cb = dict()
for va,lvl,cmap,q in zip(var,levels,cmaps,range(4)):
    r,c = int(q/2),q%2
    ax = fig.add_subplot(gs[r,c],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    cb[q] = ax.contourf(lon,lat,di[sim1][va]-di[sim2][va],levels=lvl,cmap=cmap,extend='both')
    ax.set_aspect('auto')
    
fig.subplots_adjust(top=0.5,hspace=0.25)
axes = fig.get_axes()

cbk = list(cb.keys())
xtext,ytext=0.03,1.03
labels = ['(a) TH@900 hPa', '(b) SST','(c) SLP','(d) U@300 hPa']
for ax,lvl,c,lab in zip(axes,levels,cbk,labels):
    pos = ax.get_position()
    cbax = fig.add_axes([pos.x0+pos.width,pos.y0,0.005,pos.height])
    cbar=plt.colorbar(cb[c], ticks=lvl,cax=cbax)


    text = ax.text(xtext,ytext,lab,transform=ax.transAxes,zorder=100)
    ax.set_yticks([20,40,60,80])
    ax.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
    ax.set_xticks([-120,-90,-60,-30,0,30,60])
    ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])


fig.savefig(pappath + 'ERA5-JJA-CESM-2100-IC-diff.png',dpi=300,bbox_inches='tight')
plt.close('all')
