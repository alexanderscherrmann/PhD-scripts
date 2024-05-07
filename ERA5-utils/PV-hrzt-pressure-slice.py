import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
#raphaels modules
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2
#cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import argparse
import matplotlib.patches as patches
import pickle
import helper
def PV_cmap2():
    """
    # Return the PV colormap
    # Usage:
                    from colormaps import PV_cmap
                    cmap, norm, levels = PV_cmap()['PV']
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import numpy as np
    #R G B
    _pv_data = [ # 14 colors
        [104, 137, 252], #blue
        [145, 172, 243],
        [176, 198, 235],
        [220, 195, 195],
        [220, 150, 150],
        [235, 80, 50],      #red
        [250, 120, 0],
        [250, 180, 0],
        [250, 210, 5],
        [255, 245, 5],
        [190, 220, 15],
        [120, 200, 15],  #green
        [100, 150, 20],
        [20, 130, 20]
    ]
    _pv_levels =  [-1.5,-0.5,0.,1.,1.5,2.,3.,4.,5.,6.,7.,8.,9.,10.,12,] # 15 levels
    cmaps = {
        'PV': {'array': _pv_data,
               'levels': _pv_levels},
    }
    extend_lower=np.array([0,   100, 255])/255.
    extend_upper=np.array([208, 232, 178])/255.

    cmap2=ListedColormap(np.array(_pv_data)/255.)
    cmap2.set_under(extend_lower)
    cmap2.set_over(extend_upper)

    norm2=BoundaryNorm(_pv_levels, cmap2.N)

    ticklabels=[]
    for i in _pv_levels:
       if i % 1 == 0:
           ticklabels.append(int(i))
       else:
           ticklabels.append(i)

    return cmap2,_pv_levels,norm2,ticklabels


parser = argparse.ArgumentParser(description="horizontal slice of PV at pressure:")
parser.add_argument('CYID', default=1, type=int, help='cyclone ID')
parser.add_argument('pressure', default=1, type=float, help='pressure level slice')

args = parser.parse_args()
CYID=int(args.CYID)
pressure=float(args.pressure)

pload = '/home/ascherrmann/009-ERA-5/MED/traj/use/'

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
minSLP = np.array([])
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))


f = open(pload + 'PV-data-dPSP-100-ZB-800-2-300.txt','rb')
PVdata = pickle.load(f)
f.close()

for q in PVdata['rawdata'].keys():
    if int(q)==CYID:
        i = np.where(avaID==CYID)[0][0]
        clon12 = lon[i][abs(hourstoSLPmin[i][0]).astype(int)] 
        clat12 = lat[i][abs(hourstoSLPmin[i][0]).astype(int)]
        date = dates[i][abs(hourstoSLPmin[i][0]).astype(int)]
        print(SLP[i][abs(hourstoSLPmin[i][0]).astype(int)],dates[i][0])

date = '19971206_06'
clon12 = lon[i][0]
clat12 = lat[i][0]

figpath = '/home/ascherrmann/009-ERA-5/MED/'

ana_path='/home/ascherrmann/009-ERA-5/MED/data/'

lonmin=-10
lonmax=65
latmin=25
latmax=65

s_file = ana_path+'/'+'S'+date
p_file = ana_path+'/'+'P'+date
b_file = '/home/era5/cdf/'+date[:4] + '/'+ date[4:6] + '/B' + date
latu = np.where((np.linspace(-90,90,361)>=latmin) & (np.linspace(-90,90,361)<=latmax))[0]
lonu = np.where((np.linspace(-180,180,721)>=lonmin) & (np.linspace(-180,180,721)<=lonmax))[0]
pltlat =np.linspace(-90,90,361)[latu]
pltlon = np.linspace(-180,180,721)[lonu]

minlatc = np.min(latu)
maxlatc = np.max(latu)
minpltlatc = pltlat[0]
maxpltlatc = pltlat[-1]

minlonc = np.min(lonu)
maxlonc = np.max(lonu)
minpltlonc = pltlon[0]
maxpltlonc = pltlon[-1]

PV = np.zeros((len(latu), len(lonu)))
THE = np.zeros((len(latu), len(lonu)))
U = np.zeros((len(latu), len(lonu)))
V = np.zeros((len(latu), len(lonu)))
slp = np.zeros((len(latu), len(lonu))) 

s = xr.open_dataset(s_file)
p = xr.open_dataset(p_file, drop_variables=['Q','T','OMEGA','RWC','LWC','IWC','SWC','CC'])
b = xr.open_dataset(b_file)

hya = s.hyai.values
hyb = s.hybi.values



for k in latu:
    for n in lonu:
        PS = s.PS.values[0,k,n]
        P = helper.modellevel_ERA5(PS,hya,hyb)
        I = np.where(abs(P-pressure)==np.min(abs(P-pressure)))[0][0]
        PV[k-minlatc,n-minlonc] = s.PV.values[0,I,k,n]
        slp[k-minlatc,n-minlonc] = b.MSL.values[0,k,n]/100
        U[k-minlatc,n-minlonc] = p.U.values[0,I,k,n]
        V[k-minlatc,n-minlonc] = p.V.values[0,I,k,n]


PV[PV<-999]=np.nan
slp[slp<-999]=np.nan

#open a new figure (ccrs.PlateCarre() ist the simplest projection of cartopy. For other projections have a look at: http://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html )
fig, axes = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))  
ax=axes
ax.coastlines()
ax.scatter(clon12,clat12,marker='.',s=100,color='k',zorder=100)
lonticks=np.arange(lonmin, lonmax+1,10)
latticks=np.arange(latmin, latmax+1,5)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=10)
ax.set_yticklabels(labels=latticks,fontsize=10)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

#longrids=np.arange(np.min(pltlon),np.max(pltlon),30)
#latgrids=np.arange(np.min(pltlon),np.max(pltlon),30)
ax.plot([13,25],[34.5,34.5],color='grey')
ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

cmap,pv_levels,norm,ticklabels=PV_cmap2()

h=ax.contourf(pltlon,pltlat,PV,levels=pv_levels,norm=norm,cmap=cmap,extend='both')

plevels=np.arange(960,1041,5)

h2=ax.contour(pltlon,pltlat,slp,levels=plevels,colors='purple',animated=True,linewidths=1, alpha=1) 

#plt.clabel(h2, inline=1, fontsize=8, fmt='%1.0f')
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=pv_levels,cax=cbax)  
func=resize_colorbar_vert(cbax, ax, pad=0.01, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

#rect = patches.Rectangle((minpltlonc+1,minpltlatc+0.25),11,4,edgecolor='none',facecolor='white',zorder=1.,alpha=1)
#ax.add_patch(rect)
#qv=ax.quiver(pltlon[::4],pltlat[::4],U[::4,::4],V[::4,::4],units='width',color='black')
#qk=ax.quiverkey(qv,minpltlonc+6,minpltlatc+4.,10,r'10 m s$^{-1}$', labelpos='S',coordinates='data')
#qk.text.set_zorder(5.)
#qk.Q.set_zorder(5.)
#qk.Q.set_backgroundcolor('w')
#qk.text.set_backgroundcolor
#qk.text.set_backgroundcolor('w')

ax.text(0.03, 0.95, 'b)', transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
cbar.ax.tick_params(labelsize=10)
cbar.ax.set_xlabel('PVU',fontsize=10)
cbar.ax.set_xticklabels(ticklabels)

figname=figpath+'PV-at-'+'%03dhPa-'%pressure+date+'-winds.png'       
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close()

