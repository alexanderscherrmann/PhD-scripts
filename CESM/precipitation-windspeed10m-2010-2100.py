from netCDF4 import Dataset as ds
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy
import numpy as np
import wrf
import os
import pickle
wrfd = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pappath = '/atmosdyn2/ascherrmann/015-CESM-WRF/'

# delts SLP, delta U300, delta T900, delta SST
c2100='CESM-ERA5-DJF'
c2010='CESM-2010-DJF'

precipdi = dict()

for sim in os.listdir(wrfd):
#    if not c2100 in sim and not c2010 in sim or (sim[-1]!='V' and sim[-1]!='m') or 'tropo' in sim:
    if not 'ERA5' in sim:
        continue
#    if not '0-km-max-0.7-QGPV' in sim:
        d = ds(wrfd +sim + '/wrfout_d01_2000-12-10_00:00:00','r')
    else:
        d = ds(wrfd +sim + '/wrfout_d01_2000-12-09_00:00:00','r')
    convprecip = np.squeeze(wrf.getvar(d,'RAINC'))
    microprecip = np.squeeze(wrf.getvar(d,'RAINNC'))
    totalprecip = convprecip+microprecip

    precipdi[sim]=dict()
    precipdi[sim]['conv']=convprecip
    precipdi[sim]['micro']=microprecip
    precipdi[sim]['total']=totalprecip

    
    count=0
    for out in os.listdir(wrfd+sim):
        if not 'wrfout_d01' in out:
            continue
        
        d = ds(wrfd +sim +'/' +out)

        u10=np.squeeze(wrf.getvar(d,'U10'))
        v10=np.squeeze(wrf.getvar(d,'V10'))

        uv=np.sqrt(u10**2+v10**2)
        if count==0:
            uvs = uv
            count+=1
        else:
            uvs=np.vstack((uvs,uv))

    precipdi[sim]['uvs']=uvs

f = open(pappath + 'ERA5-precipitation-UV10M-data-DJF.txt','wb')
pickle.dump(precipdi,f)
f.close()
#di = dict()
#for sim in [sim1,sim2]:
#    tmp = ds(wrfd + sim + 'wrfout_d01_2000-12-01_00:00:00','r')
#    lon = wrf.getvar(tmp,'lon')[0]
#    lat = wrf.getvar(tmp,'lat')[:,0]
#    slp = wrf.getvar(tmp,'slp')
#    U = wrf.getvar(tmp,'U')
#    U = (U[:,:,:-1] + U[:,:,1:])/2
#    P = wrf.getvar(tmp,'pres')/100
#    u300 = wrf.interplevel(U,P,300,meta=False)
#    sst = wrf.getvar(tmp,'SST')
#    TH = wrf.getvar(tmp,'th')
#    th900 = wrf.interplevel(TH,P,900,meta=False)
#    th300 = wrf.interplevel(TH,P,300,meta=False)
#
#    di[sim]=dict()
#    di[sim]['TH'] = th900[:]
#    di[sim]['SLP'] = slp[:]
#    di[sim]['SST'] = sst[:]
#    di[sim]['U'] = u300[:]
#    di[sim]['th300']=th300[:]
#    tmp.close()
#
#fig = plt.figure(figsize=(12,12))
#gs = gridspec.GridSpec(nrows=2, ncols=2)
#
#var = ['U','SLP','TH','SST']
#levels = [np.arange(-5,6,1),np.arange(-8,9,2),np.arange(-6,6.1,1),np.arange(-5,5.1,1),np.arange(-8,9,2),np.arange(-5,6,1)]
#cmaps = [matplotlib.cm.PiYG,matplotlib.cm.BrBG,matplotlib.cm.seismic,matplotlib.cm.coolwarm,matplotlib.cm.BrBG,matplotlib.cm.PiYG]
#cb = dict()
#for va,lvl,cmap,q in zip(var,levels,cmaps,range(4)):
#    r,c = int(q/2),q%2
#    ax = fig.add_subplot(gs[r,c],projection=ccrs.PlateCarree())
#    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
#    cb[q] = ax.contourf(lon,lat,di[sim1][va]-di[sim2][va],levels=lvl,cmap=cmap,extend='both')
#    ax.set_aspect('auto')
#
#fig.subplots_adjust(top=0.5,hspace=0.25,wspace=0.25)
#axes = fig.get_axes()
#
#cbk = list(cb.keys())
#xtext,ytext=0.03,1.03
#labels = ['(a) U@300 hPa', '(b) SLP','(c) TH@900 hPa','(d) SST']
#for ax,lvl,c,lab in zip(axes,levels,cbk,labels):
#    pos = ax.get_position()
#    cbax = fig.add_axes([pos.x0+pos.width,pos.y0,0.005,pos.height])
#    cbar=plt.colorbar(cb[c], ticks=lvl,cax=cbax)
#
#
#    text = ax.text(xtext,ytext,lab,transform=ax.transAxes,zorder=100)
#    ax.set_yticks([20,40,60,80])
#    ax.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
#    ax.set_xticks([-120,-90,-60,-30,0,30,60])
#    ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])
#
#
#fig.savefig(pappath + 'CESM-2010-2100-IC-diff.png',dpi=300,bbox_inches='tight')
#plt.close('all')
#
#fig=plt.figure(figsize=(6,4))
#gs=gridspec.GridSpec(nrows=1,ncols=1)
#
#ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
#ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
#lvl=np.arange(-5,6,1)
#cmap=matplotlib.cm.seismic
#cb=ax.contourf(lon,lat,di[sim1]['th300']-di[sim1]['TH']-(di[sim2]['th300']-di[sim2]['TH']),levels=lvl,cmap=cmap,extend='both')
#pos = ax.get_position()
#cbax = fig.add_axes([pos.x0+pos.width,pos.y0,0.01,pos.height])
#cbar=plt.colorbar(cb,ticks=lvl,cax=cbax)
#ax.set_yticks([20,40,60,80])
#ax.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
#ax.set_xticks([-120,-90,-60,-30,0,30,60])
#ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])
#
#fig.savefig(pappath + 'CESM-2010-2100-stratification-diff.png',dpi=300,bbox_inches='tight')
#plt.close('all')
