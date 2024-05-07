import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2
import wrfsims
import helper
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
def get_ncl_colortable(name,revert=False):
    '''
    # Read in a colortable from the NCL website:  http://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml
    # name: string providing the name of the colorbar (e.g. 'seaice_1')
    #
    #
    # authors: N. Piaget, M. Duetsch, S. Crezee, Jul 2014
    '''
    import urllib
    url = 'http://www.ncl.ucar.edu/Document/Graphics/ColorTables/Files/'+name+'.rgb'
    rgbfile = urllib.request.urlopen(url).read()
    #rgbfile = urllib.urlopen(url).read()
    filelines = rgbfile.splitlines()
    rgblines = []
    #return filelines
    for l in filelines:
        if l.split() and False not in [is_number(a) for a in l.split()]:
            rgblines.append(l)
    # print(rgblines)
    # print(rgblines[0].decode().split())
    if '.' in rgblines[0].decode().split()[0]:
        # The numbers are already floats
        ndivide = 1
    else:
        ndivide = 255
    cm=[[float(line.split()[0])/ndivide,float(line.split()[1])/ndivide,float(line.split()[2])/ndivide] for line in rgblines]
    n_colors = np.size(cm,0)
    if revert:
        cm = cm[::-1]
    seq=[]
    for i in range(0,n_colors):
        seq = seq + [cm[i],(float(i))/(float(n_colors-1)),cm[i]]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('ncl_'+name, cdict, N=n_colors)

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')

colors = ['dodgerblue','darkgreen','saddlebrown']
seas = np.array(['DJF','MAM','SON'])
amps = np.array([0.7,1.4,2.1])

pvcmap,pvlevels,pvnorm,ticklabels=PV_cmap2()
labels=['(a)','(b)','(c)']
levels=np.arange(10,101,10)
clevels=levels
cmap=matplotlib.cm.Reds
for se in seas[:1]:
  for t in ['03_00','03_12','04_00','04_12','05_00','05_12','06_00','06_12','07_00','07_12','08_00','08_12','09_00'][2:3]:
   #gfig=plt.figure(figsize=(10,3))  
   #ggs = gridspec.GridSpec(nrows=1, ncols=3)
   #qq = 0 
   for amp in amps[1:2]:
    fig=plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

    axx = gfig.add_subplot(ggs[0,qq],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    axx.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    axx.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    counter = 0
    density = np.zeros_like(ref.variables['T'][0,0])
    for simid,sim in enumerate(SIMS):
        medid = np.array(MEDIDS[simid])
        atid = np.array(ATIDS[simid])
    
        sea = sim[:3]
        if 'not' in sim:
            continue
#        print(sim)
        if se!=sea:
            continue
        if 'check' in sim:
            continue
        if '800' in sim:
            continue
        if sim[-4:]=='clim':
            continue
        if float(sim[-8:-5])!=amp:
            continue

        data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
        pv = wrf.getvar(data,'pvo')
        p = wrf.getvar(data,'pressure')
        pv300 = wrf.interplevel(pv,p,300,meta=False)
        density[pv300>=2]+=1
        counter +=1

    hc = ax.contourf(LON[0],LAT[:,0],density*100/counter,levels=levels,cmap=cmap)
    
    ax.set_xlim(-20,40)
    ax.set_ylim(20,65)

    hb = axx.contourf(LON[0],LAT[:,0],density*100/counter,levels=levels,cmap=cmap)
    kk=4
    axx.set_xlim(-20,40)
    axx.set_ylim(20,65)
    axx.set_aspect('auto')
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(hc, ticks=levels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
    fig.canvas.mpl_connect('draw_event', func)
    cbar.ax.set_yticklabels(labels=np.append(levels[:-1],r'%'))

    fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/composites-normal-runs-med-streamer/density-normal-runs-med-streamer-%s-%s-%.1f.png'%(t,se,amp),dpi=300,bbox_inches='tight')
    plt.close(fig)
#    tex=axx.text(-17.5,61.5,labels[qq],zorder=15,fontsize=8)
#    tex.set_bbox(dict(facecolor='white',edgecolor='white'))
#    qq+=1

#   plt.subplots_adjust(hspace=0,wspace=0)
#   pos=axx.get_position()
#   cbax = gfig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
#   cbar=plt.colorbar(hb, ticks=levels,cax=cbax)
#   cbar.ax.set_yticklabels(labels=np.append(levels[:-1],r'%'))
#   gfig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/composites-normal-runs-med-streamer/density-normal-runs-med-streamer-%s-%s.png'%(t,se),dpi=300,bbox_inches='tight')
#   plt.close(gfig)

#simc = ['200-km-west','400-km-west','200-km-east','400-km-east','200-km-north','400-km-north','200-km-south','400-km-south','clim-max']
#ticklabels=['2W','4W','2E','4E','2N','4N','2S','4S','C']
##levels=np.array([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
#lcol = ['cyan','navy','orange','red','grey','k','plum','purple','saddlebrown']
#lcol = ['cyan','navy','yellow','orange','forestgreen','lime','red','saddlebrown','grey']
##cmap=get_ncl_colortable('percent_11lev')
##cmap=matplotlib.cm.nipy_spectral
##norm=BoundaryNorm(levels,cmap.N)
#
#for se in seas[:1]:
#  for t in ['03_00','03_12','04_00','04_12','05_00','05_12','06_00','06_12','07_00','07_12','08_00','08_12','09_00']:
#   gfig=plt.figure(figsize=(10,3))
#   ggs = gridspec.GridSpec(nrows=1, ncols=3)
#   qq = 0
#   for amp in amps:
#    fig=plt.figure(figsize=(8,6))
#    gs = gridspec.GridSpec(nrows=1, ncols=1)
#    ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
#    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
#
#    axx = gfig.add_subplot(ggs[0,qq],projection=ccrs.PlateCarree())
#    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
#    axx.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
#    for col,labbb in zip(lcol,ticklabels):
#        axx.plot([],[],color=col)
#
#    axx.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
#    density = np.zeros_like(ref.variables['T'][0,0])
#    for simid,sim in enumerate(SIMS):
#        medid = np.array(MEDIDS[simid])
#        atid = np.array(ATIDS[simid])
#
#        sea = sim[:3]
#        if 'not' in sim:
#            continue
#        if se!=sea:
#            continue
#        if 'check' in sim:
#            continue
#        if sim[-4:]=='clim':
#            continue
#        if float(sim[-8:-5])!=amp:
#            continue
#
#        data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
#        pv = wrf.getvar(data,'pvo')
#        p = wrf.getvar(data,'pressure')
#        pv300 = wrf.interplevel(pv,p,300,meta=False)
#        for q,si in enumerate(simc):
#            if si in sim:
#                density[pv300>=2]+=levels[q]
#                col=lcol[q]
#        axx.contour(LON[0],LAT[:,0],pv300,levels=[2],colors=col,linewidths=1)
#
#    
#    ax.set_xlim(-20,40)
#    ax.set_ylim(20,65)
#
#    kk=4
#    axx.set_xlim(-20,40)
#    axx.set_ylim(20,65)
#    axx.set_aspect('auto')
#
#    fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/composites-normal-runs-med-streamer/idv-density-normal-runs-med-streamer-%s-%s-%.1f.png'%(t,se,amp),dpi=300,bbox_inches='tight')
#    plt.close(fig)
#    tex=axx.text(-17.5,61.5,labels[qq],zorder=15,fontsize=8)
#    qq+=1
#
#   plt.subplots_adjust(hspace=0,wspace=0)
#   pos=axx.get_position()
#   axx.legend(ticklabels,loc='lower right',fontsize=6)
#   gfig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/composites-normal-runs-med-streamer/idv-density-normal-runs-med-streamer-%s-%s.png'%(t,se),dpi=300,bbox_inches='tight')
#   plt.close(gfig)
#
#### density + contours
#cmap=matplotlib.cm.Reds
#for se in seas[:1]:
#  for t in ['03_00','03_12','04_00','04_12','05_00','05_12','06_00','06_12','07_00','07_12','08_00','08_12','09_00']:
#   gfig=plt.figure(figsize=(10,3))
#   ggs = gridspec.GridSpec(nrows=1, ncols=3)
#   qq = 0
#   for amp in amps:
#    fig=plt.figure(figsize=(8,6))
#    gs = gridspec.GridSpec(nrows=1, ncols=1)
#    ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
#
#    axx = gfig.add_subplot(ggs[0,qq],projection=ccrs.PlateCarree())
#
#    for col,labbb in zip(lcol,ticklabels):
#        ax.plot([],[],color=col)
#        axx.plot([],[],color=col)
#    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
#    axx.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
#    counter = 0
#    density = np.zeros_like(ref.variables['T'][0,0])
#    for simid,sim in enumerate(SIMS):
#        medid = np.array(MEDIDS[simid])
#        atid = np.array(ATIDS[simid])
#
#        sea = sim[:3]
#        if 'not' in sim:
#            continue
##        print(sim)
#        if se!=sea:
#            continue
#        if 'check' in sim:
#            continue
#        if sim[-4:]=='clim':
#            continue
#        if float(sim[-8:-5])!=amp:
#            continue
#
#        data = ds(dwrf + sim + '/wrfout_d01_2000-12-%s:00:00'%t)
#        pv = wrf.getvar(data,'pvo')
#        p = wrf.getvar(data,'pressure')
#        pv300 = wrf.interplevel(pv,p,300,meta=False)
#        for q,si in enumerate(simc):
#            if si in sim:
##                density[pv300>=2]+=levels[q]
#                col=lcol[q]
#        ax.contour(LON[0],LAT[:,0],pv300,levels=[2],colors=col,linewidths=0.5,alpha=0.5)
#        axx.contour(LON[0],LAT[:,0],pv300,levels=[2],colors=col,linewidths=0.5,alpha=0.5)
#
#        density[pv300>=2]+=1
#        counter +=1
#
#    hc = ax.contourf(LON[0],LAT[:,0],density*100/counter,levels=clevels,cmap=cmap)
#    print(np.max(density)/levels[-1])
#    hb = axx.contourf(LON[0],LAT[:,0],density*100/counter,levels=clevels,cmap=cmap)
#
#    ax.set_xlim(-20,40)
#    ax.set_ylim(20,65)
#
#    kk=4
#    axx.set_xlim(-20,40)
#    axx.set_ylim(20,65)
#    axx.set_aspect('auto')
#    cbax = fig.add_axes([0, 0, 0.1, 0.1])
#    cbar=plt.colorbar(hc, ticks=clevels,cax=cbax)
#    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
#    fig.canvas.mpl_connect('draw_event', func)
#
#    ax.legend(ticklabels,loc='lower right').set_zorder(100)
#
#    cbar.ax.set_yticklabels(labels=np.append(clevels[:-1],r'%'))
#    fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/composites-normal-runs-med-streamer/con-density-normal-runs-med-streamer-%s-%s-%.1f.png'%(t,se,amp),dpi=300,bbox_inches='tight')
#    plt.close(fig)
#    tex=axx.text(-17.5,61.5,labels[qq],zorder=15,fontsize=8)
#    tex.set_bbox(dict(facecolor='white',edgecolor='white'))
#    qq+=1
#
#   plt.subplots_adjust(hspace=0,wspace=0)
#   pos=axx.get_position()
#   axx.legend(ticklabels,loc='lower right',fontsize=6).set_zorder(100)
#   cbax = gfig.add_axes([pos.x0+pos.width, pos.y0, 0.02, pos.height])
#   cbar=plt.colorbar(hb, ticks=clevels,cax=cbax)
#   cbar.ax.set_yticklabels(labels=np.append(clevels[:-1],r'%'))
#   gfig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/composites-normal-runs-med-streamer/con-density-normal-runs-med-streamer-%s-%s.png'%(t,se),dpi=300,bbox_inches='tight')
#   plt.close(gfig)
