import numpy as np
import xarray as xr
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
import pickle


pload = '/home/ascherrmann/009-ERA-5/MED/traj/use/'

savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()

SLP = var[0]
Clon = var[1]
Clat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
dates = var[5]

SLPmin = np.array([])

avaID = np.array([])
maturedates = np.array([])
SLPminid = np.array([])
end = np.array([])

for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    loc = np.where(hourstoSLPmin[k]==0)[0][0]
    
    maturedates = np.append(maturedates,dates[k][loc])
    SLPminid = np.append(SLPminid,loc)
    end = np.append(end,maturedates[k] + '-ID-%06d.txt'%avaID[k])
    SLPmin = np.append(SLPmin,SLP[k][loc])


Lat = np.round(np.linspace(-90,90,361),2)
Lon = np.round(np.linspace(-180,180,721),2)

pinvals = np.arange(700,975.5,12.5)
pvival = np.array([-100,0.2,0.5,0.75,100])

tid = avaID

perc = dict()
sumedpvdi = dict()
for k in range(len(pvival)-1):
    perc[pvival[k]] = np.zeros(len(tid))
    sumedpvdi[pvival[k]] = np.zeros(len(tid))

sumedpv = np.zeros(len(tid))

for k in range(len(tid[:])):
    t = maturedates[k]
    ids = SLPminid[k].astype(int)
    ana_path='/net/thermo/home/ascherrmann/009-ERA-5/MED/data/'
    sfile = ana_path + 'S' + t

    s = xr.open_dataset(sfile, drop_variables=['P','TH','THE','RH','VORT','PVRCONVT','PVRCONVM','PVRTURBT','PVRTURBM','PVRLS','PVRCOND','PVRSW','PVRLWH','PVRLWC','PVRDEP','PVREVC','PVREVR','PVRSUBI','PVRSUBS','PVRMELTI','PVRMELTS','PVRFRZ','PVRRIME','PVRBF'])
    clat2 = helper.radial_ids_around_center_calc_ERA5(200)[1] + np.where(abs(Lat-Clat[k][ids])==np.min(abs(Lat-Clat[k][ids])))[0][0]
    clon2 = helper.radial_ids_around_center_calc_ERA5(200)[0] + np.where(abs(Lon-Clon[k][ids])==np.min(abs(Lon-Clon[k][ids])))[0][0]

    clat2 = clat2.astype(int)
    clon2 = clon2.astype(int)

    hya = s.hyai.values
    hyb = s.hybi.values

    pt = np.array([])
    plat = np.array([])
    plon = np.array([])
    SLP = s.PS.values[0,clat2,clon2]
    pv = s.PV.values[0,:,clat2,clon2]
    ntot = 0
    testpv = 0
    testtest = np.zeros(len(clat2))
    for l in range(len(clat2)):
        P = helper.modellevel_ERA5(SLP[l],hya,hyb)
        pid = helper.where_greater_smaller(P,700,925)#np.where((P>=700) & (P<=925))[0]
        for i in pid:
            ntot +=1
            sumedpv[k] += pv[l,i]
            
            for m in range(len(pvival)-1):
                if (((helper.where_greater_smaller(pv[l,i],pvival[m],pvival[m+1])).size) !=0):
                    perc[pvival[m]][k] +=1
                    sumedpvdi[pvival[m]][k] += pv[l,i]
    for m in range(len(pvival)-1):
        perc[pvival[m]][k] /=ntot

#legend=['PV < 0.2 PVU', '0.2 < PV < 0.5 PVU', '0.5 < PV < 0.75 PVU', '0.75 PVU < PV','all PV'] 
#fig, axes = plt.subplots(2,1,sharex=True)
#plt.subplots_adjust(left=0.1,bottom=None,top=None,right=None,wspace=0.0,hspace=0.0)
#ax = axes[0]
#ax2 = axes[1]
#marker = ['d','.','*','x']
#color = ['dodgerblue','grey','orange','red']
order = np.argsort(SLPmin[:])
sumedpv = sumedpv[order]
SLPmin = SLPmin[order]

fig, ax = plt.subplots()
ax.scatter(SLPmin,sumedpv,marker='.',color='k')
ax.set_xlabel('min. SLP [hPa]')
ax.set_ylabel(r'$\Sigma$ PV$_{P>700 hPA}$ [PVU]')
fig.savefig(pload + 'SLP-sumedPV-scatter-925.png',dpi=300,bbox_inches='tight')


save = np.zeros((len(order),6))
save[:,0] = SLPmin
for m in range(len(pvival)-1):
    save[:,m+1] = sumedpvdi[pvival[m]]
save[:,5] = sumedpv
np.savetxt(pload + 'summed-pv-data-from-cdf-925.txt',save.astype(float),fmt='%f',delimiter=' ',newline='\n')

#relvort = relvort[order]
#sumedpv = sumedpv[order]
#print(sumedpv)
#print(pearsonr(relvort,sumedpv))
#for m in range(len(pvival)-1):
#    perc[pvival[m]] = perc[pvival[m]][order]
#    sumedpvdi[pvival[m]] = sumedpvdi[pvival[m]][order]
#    ax.plot(relvort,perc[pvival[m]],color=color[m])
#    ax2.plot(relvort,sumedpvdi[pvival[m]],color=color[m],ls='-')
#    print(pearsonr(relvort,sumedpvdi[pvival[m]]))
#    print(sumedpvdi[pvival[m]]/sumedpv)
#ax.set_ylim(0,0.75)
#ax2.set_ylim(-50,1700)
#ax.tick_params(labelright=False,right=True,labelbottom=False,bottom=True)
#ax2.tick_params(labelright=False,right=True,labeltop=False,top=True)
#print(pearsonr(relvort,(sumedpvdi[0.5] + sumedpvdi[0.75])/sumedpv))
#
#ax2.plot(relvort,sumedpv,color='k')
#ax2.legend(legend,fontsize=8,frameon=False,loc='upper left')
#ax2.set_ylabel(r'$\Sigma$ PV [PVU]')
#ax.set_ylabel(r'data points with PV [%]')
#ax2.set_xlabel(r'rel. vort. [$\times 10^{-4}$ s$^{-1}$]')
##fig.savefig(sp + 'pv-anomaly-percentage.png',dpi=300, bbox_inches="tight")
#plt.close('all')

