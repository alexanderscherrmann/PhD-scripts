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
import argparse
import cartopy
import matplotlib.gridspec as gridspec
import pandas as pd

parser = argparse.ArgumentParser(description="plot accumulated average PV gain that is associated with the cyclone and the environment")
parser.add_argument('rdis',default='',type=int,help='distance from center in km that should be considered as cyclonic')
args = parser.parse_args()
rdis = int(args.rdis)

pload = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'
pload2 = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-%d-correct-distance.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

kickids = np.loadtxt('/home/ascherrmann/009-ERA-5/MED/kick-IDS.txt')

f = open('/home/ascherrmann/009-ERA-5/MED/check-IDS.txt','rb')
getids = pickle.load(f)
f.close()


#newID = getids['newID']


df = pd.read_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
### take only the ones with atleast 200 trajectories
df = df.loc[df['ntraj075']>=200]

#cf = open('/home/ascherrmann/009-ERA-5/MED/climatologyPV.txt','rb')
#clim = pickle.load(cf)
#cf.close()

clim = np.loadtxt('/home/ascherrmann/009-ERA-5/MED/clim-avPV.txt')

df2 = pd.DataFrame(columns=['PV','count','avPV'],index=['Year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
#print(list(clim['PV'].values()))
#df2['PV'] = list(clim['PV'].values())
#df2['count'] = list(clim['count'].values())
df2['avPV'] = np.append(np.mean(clim),clim)#df2['PV']/df2['count']

dipv = PVdata['dipv']
datadi = PVdata['rawdata']
dit = PVdata['dit']

both = np.array([])
adv = np.array([])
cyclonic = np.array([])
environmental = np.array([])


SLP = df['minSLP'].values
lon = df['lon'].values
lat = df['lat'].values
ID = df['ID'].values
hourstoSLPmin = df['htminSLP'].values
print(len(ID),len(hourstoSLPmin))
maturedates = df['date'].values
#savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
#var = []
#
#for u,x in enumerate(savings):
#    f = open(pload2[:-4] + x + 'furthersel.txt',"rb")
#    var.append(pickle.load(f))
#    f.close()
#
#SLP = var[0]
#lon = var[1]
#lat = var[2]
#ID = var[3]
#hourstoSLPmin = var[4]
#dates = var[5]
#avaID = np.array([])
#maturedates = np.array([])
#for k in range(len(ID)):
#    avaID=np.append(avaID,ID[k][0].astype(int))
#    maturedates = np.append(maturedates,dates[k][abs(hourstoSLPmin[k][0]).astype(int)])

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

cycounter = 0
envcounter = 0

cycounter2 = 0
envcounter2 = 0
cycounter3 = 0
envcounter3 = 0

counters = dict()
counters[60] = dict()
counters[85] = dict()
for q in range(6):
    counters[60][q] = np.zeros((361,721))
    counters[85][q] = np.zeros((361,721))

size = 1
for h in np.arange(0,49):
    pvloc[h] = np.array([])
    cycd[h] = np.array([])
    envd[h] = np.array([])

cmap = ListedColormap(['steelblue','dodgerblue','lightskyblue','lightcoral','red','firebrick'])
norm = BoundaryNorm([0,1,2,3,4,5,6],cmap.N)#,7,8],cmap.N)
cmap.set_over('darkred')
cmap.set_under('navy')

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=2,ncols=3)
axes = []
for k in range(2):
    for l in range(3):
        axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))
plt.close(fig)        

fig, axes = plt.subplots(3,1, subplot_kw=dict(projection=ccrs.PlateCarree()),sharex=True,sharey=False)
plt.subplots_adjust(left=0.1,bottom=None,top=None,right=0.4,hspace=0,wspace=0)
axes = axes.flatten()
identifi = np.array([])
fib = 0.5
tb = 0.75
thb =0.85
fb=0.5
tb=0.85
hcyc = 6

ncyc = 0
nenv = 0
nadv = 0 

yearlyc = dict()
yearlye = dict()
yearlya = dict()

months = ['J','F','M','A','M','J','J','A','S','O','N','D']
numbers = np.arange(1,13)

LON = np.arange(-180,180.1,0.5)
LAT = np.arange(-90,90.1,0.5)

#cycano = np.zeros(len(ID))
#envano = np.zeros(len(ID))
#advano = np.zeros(len(ID))
#advano2 = np.zeros(len(ID))

cycano = np.array([])
envano = np.array([])
advano = np.array([])
advano2 = np.array([])

cyper= np.zeros(len(ID))
enper= np.zeros(len(ID))
adper = np.zeros(len(ID))
for mon in numbers:
    yearlyc[mon] = np.array([])
    yearlye[mon] = np.array([])
    yearlya[mon] = np.array([])
    
for ll,k in enumerate(dipv.keys()):
    if np.all(ID!=int(k)):
      continue
#   if np.all(kickids!=int(k)):         
#   if np.all(newID!=int(k)):
    q = np.where(ID==int(k))[0][0]
    MON = df['mon'].values[q]
    if (hourstoSLPmin[q]<(hcyc)):
        continue
    CLON = lon[q]#[abs(hourstoSLPmin[q][0]).astype(int)]
    CLAT = lat[q]#[abs(hourstoSLPmin[q][0]).astype(int)]

    CLON = LON[np.where(abs(LON-CLON)==np.min(abs(LON-CLON)))[0][0]]
    CLAT = LAT[np.where(abs(LAT-CLAT)==np.min(abs(LAT-CLAT)))[0][0]]
    
    d = k
    ac[d] = dict()
    pressuredi[d] = dict()
    for h in np.arange(-48,49):
        ac[d][h] = np.array([])
        pressuredi[d][h] = np.array([])

    d = k
    OL = PVdata['rawdata'][d]['OL']    
    pre = PVdata['rawdata'][d]['P']
    PV = PVdata['rawdata'][d]['PV']
    i = np.where(PV[:,0]>=0.75)[0]

    pvend = PV[i,0]
    pvstart = PV[i,-1]

    cypv = dipv[d][c][i,0]
    enpv = dipv[d][e][i,0]

    cy = np.mean(cypv)
    uu = np.where(LON==CLON)[0][0]
    ll = np.where(LAT==CLAT)[0][0]
    
    PVstart = np.append(PVstart,pvstart)
    PVend = np.append(PVend,pvend)

    for h in np.arange(0,49):
        pvloc[h] = np.append(pvloc[h],PV[i,h])
        cycd[h] = np.append(cycd[h],dipv[d][c][i,h])
        envd[h] = np.append(envd[h],dipv[d][e][i,h])

    montmp = int(maturedates[q][4:6])
    yearlyc[montmp] = np.append(yearlyc[montmp],cypv/pvend)
    yearlye[montmp] = np.append(yearlye[montmp],enpv/pvend)
    yearlya[montmp] = np.append(yearlya[montmp],(pvstart)/pvend)

    adv = np.append(adv,(pvstart)/pvend)
    cyc = np.append(cyc,cypv/pvend)
    env = np.append(env,enpv/pvend)

    cyper[q]= np.mean(cypv/pvend)
    enper[q] = np.mean(enpv/pvend)
    adper[q] =np.mean(pvstart/pvend)


    cycano = np.append(cycano,cypv/(pvend-df2['avPV'][MON]))
    envano = np.append(envano,enpv/(pvend-df2['avPV'][MON]))
    advano = np.append(advano,(pvstart-df2['avPV'][MON])/(pvend-df2['avPV'][MON]))
    advano2 = np.append(advano2,pvstart/(pvend-df2['avPV'][MON]))
    
    #cycano[q] = np.mean(cypv/(pvend-df2['avPV'][MON]))
    #envano[q] = np.mean(enpv/(pvend-df2['avPV'][MON]))
    #advano[q] = np.mean((pvstart-df2['avPV'][MON])/(pvend-df2['avPV'][MON]))

    ct +=1

#    if np.mean(cypv/pvend)>fib:
    if np.mean(cypv/(pvend-df2['avPV'][MON]))>=fib:
        counters[60][0][ll,uu]+=1
        ncyc+=1
###
    if np.mean(enpv/(pvend-df2['avPV'][MON]))>=fib:
        counters[60][1][ll,uu]+=1
        nenv+=1
##    if np.mean(cypv/pvend)>tb:
##        counters[85][0][ll,uu]+=1
##    if np.mean(enpv/pvend)>tb:
##        counters[85][1][ll,uu]+=1
    if np.mean((pvstart-df2['avPV'][MON])/(pvend-df2['avPV'][MON]))>=fib:
        counters[60][2][ll,uu]+=1
        nadv+=1
##        countera60+=1
##    if np.mean(pvstart/pvend)>tb:
##        counters[85][2][ll,uu]+=1
##
#####################################
#    if (hourstoSLPmin[q]<(hcyc)):
#        continue
##
##    if len(i)<100:
##        continue
####################################
#    if np.mean(cypv/(pvend-df2['avPV'][MON]))>fib:
#        counters[60][3][ll,uu]+=1
##        counterc60+=1
###
#    if np.mean(enpv/(pvend-df2['avPV'][MON]))>fib:
#        counters[60][4][ll,uu]+=1
##        countere60+=1
##    if np.mean(cypv/pvend)>tb:
##        counters[85][3][ll,uu]+=1
##        counterc75+=1
##    if np.mean(enpv/pvend)>tb:
##        counters[85][4][ll,uu]+=1
###        countere75+=1
###
#    if np.mean((pvstart-df2['avPV'][MON])/(pvend-df2['avPV'][MON]))>fib:
#        counters[60][5][ll,uu]+=1
#        countera60+=1
#    if np.mean(pvstart/pvend)>tb:
#        counters[85][5][ll,uu]+=1
#        if d=='241105' or d=='230058': 
#        countera75+=1

#        envcounter2+=1

#    if (cy<0.05):
#        ca+=1
#        continue

#    for we in i:
#        if np.where(dit[d][c][we]==0)[0].size==0:
#            cyid = -1
#        else:
#            cyid = np.where(dit[d][c][we]==0)[0][0]
#        tn = np.flip(np.arange(-48,1))-np.flip(np.arange(-48,1))[cyid]
#        for a,b in enumerate(tn):
#            ac[d][b] = np.append(ac[d][b],PV[we,a])
#            pressuredi[d][b] = np.append(pressuredi[d][b],pre[we,a])
#
 #   cycpvc = np.zeros(dipv[d][c].shape)
 #   cycpvc[i,:-1] = dipv[d][c][i,:-1] - dipv[d][c][i,1:]

#f = open('/home/ascherrmann/009-ERA-5/MED/counter.txt','wb')
#count = dict()
#for k in [60]:
#    count[k]=counters[k]
#pickle.dump(count,f)
#f.close()
#
#f = open('/home/ascherrmann/009-ERA-5/MED/counter.txt','rb')
#counters = dict()
#counters[60] = dict()
#count=pickle.load(f)
#k = 60
#for q in range(6):
#    counters[k][q] = count[k][q]
#f.close()
#

#df['cycper'] = cyper
#df['envper'] = enper
#df['advper'] = adper
#df['cycperano'] =cycano
#df['envperano'] = envano
#df['advperano'] = advano
#
#df.to_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv',index=False)
count2 = dict()
k = 60
#for k in [60,85]:
count2[k] = dict()
count3 = dict()
count3[k] = dict()
count4 = dict()
count4[k] = dict()
steps = [[0,0],[0,1],[1,0],[1,1]]
for q in range(0,6):
        print(np.sum(counters[k][q]))

        count2[k][q]=np.zeros((361,721))
        count3[k][q]=np.zeros((181,361))
        count4[k][q]=np.zeros((181,361))
        for l in range(0,361):
            for a in range(0,721):
                tmp=[]
                for z,s in steps:
                    tmp.append(np.sum(counters[k][q][l-1+z:l+z+2,a+s-1:a+s+2]))
                count2[k][q][l,a] = np.mean(tmp)
                #count2[k][q][l,a] = np.sum(counters[k][q][l-1:l+2,a-1:a+2])
        for l in range(0,181):
            for a in range(361):
                count3[k][q][l,a] = np.sum(counters[k][q][2*l:2*(l+1),2*a:2*(a+1)])

        for l in range(0,181):
            for a in range(361):
                count4[k][q][l,a] = np.mean(count3[k][q][l-1:l+2,a-1:a+2])



#boxpv = []
#pvav = np.array([])
#pv10 = np.array([])
#pv90 = np.array([])
#pvmed = np.array([])
#pv25= np.array([])
#pv75= np.array([])
#for h in np.flip(np.arange(0,49)):
#    pvloc[h] = np.sort(pvloc[h])
#    boxpv.append(pvloc[h])
#    pvav = np.append(pvav,np.mean(pvloc[h]))
#
#    pv10= np.append(pv10,np.percentile(pvloc[h],10))
#    pv90 = np.append(pv90,np.percentile(pvloc[h],90))
#    pv25= np.append(pv25,np.percentile(pvloc[h],25))
#    pv75= np.append(pv75,np.percentile(pvloc[h],75))
#    pvmed= np.append(pvmed,np.percentile(pvloc[h],50))
##
print(np.mean(adv),np.mean(cyc),np.mean(env),ct)
#
print('anomalies')
print(np.mean(advano),np.mean(cycano),np.mean(envano),np.mean(advano2))
print('numbers')
print(ncyc,nenv,nadv)
#print(np.mean(identifi))
#print(np.mean(PVstart),np.mean(PVend),np.mean(PVstart)/np.mean(PVend))
#print(ca,ct)
#print(cycounter,envcounter,ct)
#print(cycounter2,envcounter2)
#print(cycounter3,envcounter3)

#print(counterc60,countere60,countera60)
#print(counterc75,countere75,countera75)
#print(counterc85,countere85,countera85)

minpltlonc = -10
maxpltlonc = 45
minpltlatc = 25
maxpltlatc = 50
steps = 5

lonticks=np.arange(minpltlonc, maxpltlonc+1,steps*3)
latticks=np.arange(minpltlatc, maxpltlatc+1,steps*2)

#lab = ['60% cyc', '60% env','60% adv', '75% cyc','75%env', '75% adv','85% cyc','85% env','85% adv','90% cyc','90% env','90% adv']
labs = ['cyc','env','adv']
labels = ['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)','k)','l)']
colo = ['blue','red']
DLON = np.arange(-180,180.1,1)
DLAT = np.arange(-90,90.1,1)
levels = np.arange(2,100,12)
colors=['k','purple']
for q, ax in enumerate(axes):
##    if q==0:
##        ax.plot([],[],ls=' ',marker='.',color='k',markersize=2)
#    #if q==1:
##        ax.plot([],[],ls=' ',marker='.',color='k',markersize=28/3.5)
##    print(np.max(counters[60][q]),np.max(counters[85][q]))
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc])
    k = 60
    cf = ax.contour(LON,LAT,count2[k][q],linewidths=0.5,colors='k',levels=levels)


##    for u,k in enumerate([60,85]):
##        latids,lonids = np.where(counters[k][q]!=0)
##        ax.scatter(LON[lonids],LAT[latids],color=colo[u],s=counters[k][q][latids,lonids]/2)
##        if q<4:
##            if q==3 and k==85:
##                latids,lonids = np.where(counters[k][q]!=0)
##                ax.scatter(LON[lonids],LAT[latids],color=colo[u],s=counters[k][q][latids,lonids]/2)
##            else:
##                cf = ax.contour(LON,LAT,count2[k][q],linewidths=0.5,colors=colors[u],levels=levels)
##        else:
##            latids,lonids = np.where(counters[k][q]!=0)
##            ax.scatter(LON[lonids],LAT[latids],color=colo[u],s=counters[k][q][latids,lonids]/2)
#
##    ax.coastlines()
    ax.set_aspect('auto')
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree())
    ax.set_yticks(latticks, crs=ccrs.PlateCarree())
##    if q%3==0:
    ax.set_yticklabels(labels=latticks,fontsize=6)
    ax.yaxis.set_major_formatter(LatitudeFormatter())
##    if q>=3:
    ax.set_xticklabels(labels=lonticks,fontsize=6)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
##    if q<3:
##        lab1 = '%d'%(fib*100) + r'% '
##    elif q<6:
##        lab1 = '%d'%(tb*100) + r'% '
##    elif q<9:
##        lab1 = '%d'%(thb*100) + r'% '
##    else:
##        lab1 = '%d'%(fb*100) + r'% '
##
##    if q%3==0:
##        lab2 = 'cyc'
##    elif q%3==1:
##        lab2 = 'env'
##    else:
##        lab2 = 'adv'
##    labs = lab1 + lab2
# #   ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc])
#    if q%3!=0:
#        ax.set_yticklabels([])
#
#   # ax.text(0.45, 0.95, labs[q%3], transform=ax.transAxes,fontsize=8,va='top')
    ax.text(0.01, 0.98, labels[q], transform=ax.transAxes,fontsize=10, fontweight='bold',va='top')
#
#
##axes[0].legend(['1 cyclone','30 cyclones'],loc='lower right',frameon=True,fontsize='x-small',ncol=2)
##axes[1].legend([' 30 cyclones'],loc='lower right',frameon=False,fontsize='small')
#
#for ax in [axes[1],axes[2],axes[4],axes[5]]:
#    ax.set_yticklabels([])
#
#plt.subplots_adjust(bottom=0.1,top=0.6,wspace=0,hspace=0.15)
#
##cax = fig.add_axes([0, 0, 0.1, 0.1])
##cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,orientation='vertical',extend='both')
##cbar.ax.set_yticklabels(['90','85','75','60','50','60','75','85','90'])
##func=resize_colorbar_vert(cax, axes[1::2], pad=0.0, size=0.02)
##fig.canvas.mpl_connect('draw_event', func)
##cax = fig.add_axes([0.925,0.11,0.0175,0.77])
#
##cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax ,extend='both',orientation='vertical')
##cbar.ax.set_yticklabels(['90','85','75','60','75','85','90'])
plt.subplots_adjust(left=0.1,bottom=None,top=None,right=None,hspace=0,wspace=0)

#fig.savefig('/home/ascherrmann/009-ERA-5/MED/' + 'cyclones-colored-contribution-new-adv-all-%01dh-2-%d-color.png'%(hcyc,rdis),dpi=300,bbox_inches="tight")
#fig.savefig('/home/ascherrmann/009-ERA-5/MED/' + 'contribution-0h-%d-vertical.png'%(rdis),dpi=300,bbox_inches="tight")
plt.close('all')

fig, axes = plt.subplots(1,3, subplot_kw=dict(projection=ccrs.PlateCarree()),sharex=False,sharey=True)
plt.subplots_adjust(left=0.1,bottom=None,top=0.4,right=None,wspace=0)
axes = axes.flatten()
k=60
for q, ax in enumerate(axes):
    q+=3
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc])
    k = 60
    cf = ax.contour(LON,LAT,count2[k][q],linewidths=0.5,colors='k',levels=levels)

    ax.set_aspect('auto')
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree())
    ax.set_yticks(latticks, crs=ccrs.PlateCarree())
    ax.set_yticklabels(labels=latticks,fontsize=6)
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xticklabels(labels=lonticks,fontsize=6)
    ax.xaxis.set_major_formatter(LongitudeFormatter())

    ax.text(0.01, 0.98, labels[q], transform=ax.transAxes,fontsize=10, fontweight='bold',va='top')

#fig.savefig('/home/ascherrmann/009-ERA-5/MED/' + 'contribution-0h-%d.png'%(rdis),dpi=300,bbox_inches="tight")
plt.close('all')



tsc = []
#tscx = np.array([])
#tmpd = dict()
#pred = dict()
#for h in np.arange(-48,49):
#    tmpd[h] = np.array([])
#    pred[h] = np.array([])
#for h in np.arange(-48,49):
#    for q,d in enumerate(dipv.keys()):
#        tmpd[h] = np.append(tmpd[h],ac[d][h])
#        pred[h] = np.append(pred[h],pressuredi[d][h])
#
#ntraj = np.array([])
#prec = np.array([])
#for h in tmpd.keys():
#    if len(tmpd[h]>10):
#        tscx = np.append(tscx,h)
#        tsc.append(tmpd[h])
#        prec = np.append(prec,np.mean(pred[h]))
#        ntraj = np.append(ntraj,len(tmpd[h]))
#
#
cycbox = []
envbox = []
avlic = np.array([])
tenlic = np.array([])
ninetylic = np.array([])
avlie= np.array([])
tenlie = np.array([])
ninetylie= np.array([])
cy25= np.array([])
cy75= np.array([])
cy50= np.array([])
en25= np.array([])
en75= np.array([])
en50= np.array([])

#for h in np.flip(np.arange(0,49)):
#        cycbox.append(np.sort(cycd[h]))
#        envbox.append(np.sort(envd[h]))
#        avlic = np.append(avlic,np.mean(cycd[h]))
#        tenlic = np.append(tenlic,np.percentile(np.sort(cycd[h]),10))
#        ninetylic = np.append(ninetylic,np.percentile(np.sort(cycd[h]),90))
#        avlie = np.append(avlie,np.mean(envd[h]))
#        tenlie = np.append(tenlie,np.percentile(np.sort(envd[h]),10))
#        ninetylie = np.append(ninetylie,np.percentile(np.sort(envd[h]),90))
#        cy25= np.append(cy25,np.percentile(np.sort(cycd[h]),25))
#        cy75= np.append(cy75,np.percentile(np.sort(cycd[h]),75))
#        cy50= np.append(cy50,np.percentile(np.sort(cycd[h]),50))
#        en25= np.append(en25,np.percentile(np.sort(envd[h]),25))
#        en75= np.append(en75,np.percentile(np.sort(envd[h]),75))
#        en50= np.append(en50,np.percentile(np.sort(envd[h]),50))




#
#xx = np.array([])
#data = []
#PVends = np.append(np.arange(0.75,2.01,0.125),1000)
#for q,k in enumerate(PVends[:-1]):
#    ids = helper.where_greater_smaller(PVend,k,PVends[q+1])
#    xx = np.append(xx,k)
#    data.append(np.sort(PVstart[ids]))

#flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
#meanline = dict(linestyle='-',linewidth=1,color='red')
#
#meanline2 = dict(linestyle=':',linewidth=1,color='navy')
#capprops = dict(linestyle=':',linewidth=1,color='dodgerblue')
#medianprops = dict(linestyle=':',linewidth=1,color='purple')
#boxprops = dict(linestyle=':',linewidth=1.,color='slategrey')
#whiskerprops= dict(linestyle=':',linewidth=1,color='dodgerblue')

flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')
#
meanline2 = dict(linestyle='-',linewidth=1,color='dodgerblue')
capprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops = dict(linestyle='-',linewidth=1,color='grey')
medianprops1 = dict(linestyle='-',linewidth=1,color='black')
boxprops = dict(linestyle='-',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle='-',linewidth=1,color='grey')

#fig,ax = plt.subplots()
#ax.set_ylabel(r'PV$_{start}$ [PVU]')
#ax.set_xlabel(r'PV$_{end}$ [PVU]')
#bp = ax.boxplot(data,whis=(10,90),labels=xx,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False)
#ax.set_xticks(ticks=range(1,len(xx)+1))
#ax.set_xticklabels(labels=xx)
#ax.set_ylim(-0.25,1.25)
#fig.savefig('/home/ascherrmann/009-ERA-5/MED/boxwis-6hprio-PV-start-end.png',dpi=300,bbox_inches="tight")
#plt.close('all')

#
#
#fig,ax = plt.subplots()
#ax.set_ylabel(r'PV [PVU]')
#ax.set_xlabel(r'time to mature stage [h]')
#ax.set_ylim(-.25,2.25)
#ax.set_xlim(-48,0)
#t = np.arange(-48,1)
#print(pvav[0]-pvav[-1])
#print(avlic[0]-avlic[-1],avlie[0]-avlie[-1])
#ax.plot(t,pvav,color='k',linewidth=2)
#ax.fill_between(t,pv10,pv90,color='grey',alpha=0.5)
#ax.set_xticks(ticks=np.arange(-48,1,6))
#ax.tick_params(labelright=False,right=True)
#ax.set_xticklabels(labels=t[0::6])
#ax.text(-0.1, 0.98,'a)',transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
#fig.savefig('/home/ascherrmann/009-ERA-5/MED/PV-evol-lines-t-%d-dis-%d.png'%(hcyc,rdis),dpi=300,bbox_inches="tight")
#plt.close('all')
#
#fig,ax = plt.subplots()
#ax.set_ylabel(r'PV [PVU]')
#ax.set_xlabel(r'time to mature stage [h]')
#ax.set_ylim(-.75,1.75)
#ax.set_xlim(-48,0)
#t = np.arange(-48,1)
##ax.plot(t,avlic,color='k',linewidth=2)
##ax.plot(t,avlie,color='grey',linewidth=2)
#ax.plot(t,cy50,color='red',linewidth=2,zorder=5)
#ax.plot(t,en50,color='dodgerblue',linewidth=2,zorder=6)
#ax.axhline(0,linewidth=1,color='k',zorder=10)
#ax.fill_between(t,tenlic,ninetylic,alpha=0.5,color='red')
#ax.plot(t,cy25,color='red',linewidth=2.,linestyle='--')
#ax.plot(t,cy75,color='red',linewidth=2.,linestyle='--')
#ax.plot(t,en25,color='dodgerblue',linewidth=2.,linestyle='--')
#ax.plot(t,en75,color='dodgerblue',linewidth=2.,linestyle='--')
#ax.fill_between(t,tenlie,ninetylie,alpha=0.5,color='dodgerblue')
#ax.legend(['apvc','apve'],loc='upper left')
##bp = ax.boxplot(boxpv,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops1)
##ax.set_xticks(ticks=range(1,len(t)+1,6))
#ax.set_xticks(ticks=np.arange(-48,1,6))
#ax.tick_params(labelright=False,right=True)
#ax.set_xticklabels(labels=t[0::6])
##fig.savefig('/home/ascherrmann/009-ERA-5/MED/boxwis-high-PV-t-%01d-2-%d.png'%(hcyc,rdis),dpi=300,bbox_inches="tight")
#ax.text(-0.1, 0.98,'b)',transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
#fig.savefig('/home/ascherrmann/009-ERA-5/MED/env-cyc-PV-lines-t-%d-dis-%d.png'%(hcyc,rdis),dpi=300,bbox_inches="tight")
#plt.close('all')
##
#
#fig,ax = plt.subplots()
#ax.set_ylabel(r'PV [PVU]')
#ax.set_xlabel(r'time to mature stage [h]')
#ax.set_ylim(-.750,1.5)
#ax.set_xlim(0,50)
#t = np.arange(-48,1)
#bp = ax.boxplot(cycbox,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops1)
#bp2 = ax.boxplot(envbox,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline2,meanline=True,showmeans=True,showbox=True,showcaps=True,showfliers=False,medianprops=medianprops,capprops=capprops,whiskerprops=whiskerprops,boxprops=boxprops)
#
#ax.set_xticks(ticks=range(1,len(t)+1,6))
#ax.tick_params(labelright=False,right=True)
#ax.set_xticklabels(labels=t[0::6])
##fig.savefig('/home/ascherrmann/009-ERA-5/MED/boxwis-cyc-env-high-PV-all-t-%01d-2-%d.png'%(hcyc,rdis),dpi=300,bbox_inches="tight")
#plt.close('all')
##
##
#yearc = []
#yeare = []
#yeara = []
#
#for mon in yearlyc.keys():
#    yearc.append(np.sort(yearlyc[mon])*100)
#    yeare.append(np.sort(yearlye[mon])*100)
#    yeara.append(np.sort(yearlya[mon])*100)
#
#ye = [yearc,yeare,yeara]
#fig,axes = plt.subplots(3,1,sharex=True)
#axes = axes.flatten()
#for q,ax in enumerate(axes):
#    ax.set_xlim(0,13)
#    ax.set_ylim(-40,100)
#    ax.text(0.03, 0.95, labels[q], transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
#    bp = ax.boxplot(ye[q],whis=(10,90),labels=months,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False,medianprops=medianprops1)
#    ax.set_xticks(ticks=np.arange(1,13))
#    ax.set_ylabel('PV [PVU]')
#
#ax.set_xticklabels(labels=months)
#plt.subplots_adjust(left=0.1,hspace=0,wspace=0)
#
##fig.savefig('/home/ascherrmann/009-ERA-5/MED/ERA5-seasonality-cyc-env-adv-PV-t-%01d-2-%d.png'%(hcyc,rdis),dpi=300,bbox_inches="tight")
#plt.close('all')
#
#
#fig,ax = plt.subplots()
#ax2 = ax.twinx()
#ax.set_ylabel(r'PV [PVU]')
#ax.set_xlabel(r'time entering the cyclone [h]')
#ax.set_ylim(-.25,2.0)
#ax.set_xlim(0,len(tscx)+1)
#t = tscx
#zid = np.where(t==0)[0]
#ax.axvline(zid+1,color='k',alpha=0.4)
#ax.axvline(zid+1-6,color='k',alpha=0.4)
#ax.axvline(zid+1+6,color='k',alpha=0.4)
#ax2.plot(np.arange(1,len(t)+1),ntraj,color='blue',alpha=0.6)
#ax2.set_ylabel('number of trajectories')
#ax2.set_yscale('log')
#firsttick = np.where(t%6==0)[0][0]
#atick = np.arange(1,len(t))
##
##tick = np.append(np.flip(np.flip(np.arange(
#bp = ax.boxplot(tsc,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False)
#ax.set_xticks(ticks=np.arange(atick[firsttick],len(tscx),6))
##ax.tick_params(labelright=False,right=True)
#ax.set_xticklabels(labels=np.arange(t[firsttick],t[-1]+0.000001,6).astype(int))
#fig.savefig('/home/ascherrmann/009-ERA-5/MED/boxwis-PV-enter-cyclone.png',dpi=300,bbox_inches="tight")
#plt.close('all')
