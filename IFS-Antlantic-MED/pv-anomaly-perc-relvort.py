import numpy as np
import xarray as xr
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
import pickle


relvort = relvort[tid]
MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
MONTHSN = np.arange(1,13,1)

pload = '/home/ascherrmann/010-IFS/traj/MED/use/'
PVdata = 


Lat = np.round(np.linspace(0,90,226),2)
Lon = np.round(np.linspace(-180,180,901),2)

pinvals = np.arange(700,975.5,12.5)
pvival = np.array([-100,0.2,0.5,0.75,100])

perc = dict()
sumedpvdi = dict()
for k in range(len(pvival)-1):
    perc[pvival[k]] = np.zeros(len(tid))
    sumedpvdi[pvival[k]] = np.zeros(len(tid))
sumedpv = np.zeros(len(tid))

for k in range(len(tid[:])):
    a = tid[k]
    t = dates[a]
    yyyy = int(t[0:4])
    MM = int(t[4:6])
    DD = int(t[6:8])
    hh = int(t[9:])
    monthid, = np.where(MONTHSN==MM)
    Month = MONTHS[monthid[0]] + t[2:4]
    if ((yyyy==2018)&(MM==12)):
        Month = 'NOV18'
    if ((MM==2) & (DD<4)):
        Month = 'JAN18'

    ana_path='/net/thermo/atmosdyn/atroman/phd/'+ Month +'/cdf/'
    sfile = ana_path + 'S' + t
    cycl = xr.open_dataset('/net/thermo/atmosdyn/atroman/phd/'+ Month + '/cyclones/CY' + t + '.nc',drop_variables=['CENTRES','WFRONTS','CFRONTS','BBFRONTS'])

    s = xr.open_dataset(sfile, drop_variables=['P','TH','THE','RH','VORT','PVRCONVT','PVRCONVM','PVRTURBT','PVRTURBM','PVRLS','PVRCOND','PVRSW','PVRLWH','PVRLWC','PVRDEP','PVREVC','PVREVR','PVRSUBI','PVRSUBS','PVRMELTI','PVRMELTS','PVRFRZ','PVRRIME','PVRBF'])

    clat2 = helper.radial_ids_around_center_calc(200)[1] + np.mean(clat[a])#).astype(int)
    clon2 = helper.radial_ids_around_center_calc(200)[0] + np.mean(clon[a])#).astype(int)
    clat2 = clat2.astype(int)
    clon2 = clon2.astype(int)

    pt = np.array([])
    plat = np.array([])
    plon = np.array([])
    SLP = s.PS.values[0,0,clat2,clon2]
    pv = s.PV.values[0,:,clat2,clon2]
    ntot = 0
    testpv = 0
    testtest = np.zeros(len(clat2))
    for l in range(len(clat2)):
        P = helper.modellevel_to_pressure(SLP[l])
        pid = helper.where_greater_smaller(P,700,975)#np.where((P>=700) & (P<=925))[0]
        for i in pid:
            ntot +=1
            sumedpv[k] += pv[l,i]
            
            for m in range(len(pvival)-1):
                if (((helper.where_greater_smaller(pv[l,i],pvival[m],pvival[m+1])).size) !=0):
                    perc[pvival[m]][k] +=1
                    sumedpvdi[pvival[m]][k] += pv[l,i]
    for m in range(len(pvival)-1):
        perc[pvival[m]][k] /=ntot

legend=['PV < 0.2 PVU', '0.2 < PV < 0.5 PVU', '0.5 < PV < 0.75 PVU', '0.75 PVU < PV','all PV'] 
fig, axes = plt.subplots(2,1,sharex=True)
plt.subplots_adjust(left=0.1,bottom=None,top=None,right=None,wspace=0.0,hspace=0.0)
ax = axes[0]
ax2 = axes[1]
marker = ['d','.','*','x']
color = ['dodgerblue','grey','orange','red']
order = np.argsort(relvort)

save = np.zeros((len(order),6))
save[:,0] = relvort
for m in range(len(pvival)-1):
    save[:,m+1] = sumedpvdi[pvival[m]]
save[:,5] = sumedpv
#np.savetxt(sp + 'summed-pv-data-from-cdf.txt',save.astype(float),fmt='%f',delimiter=' ',newline='\n')
relvort = relvort[order]
sumedpv = sumedpv[order]
#print(sumedpv)
print(pearsonr(relvort,sumedpv))
for m in range(len(pvival)-1):
    perc[pvival[m]] = perc[pvival[m]][order]
    sumedpvdi[pvival[m]] = sumedpvdi[pvival[m]][order]
    ax.plot(relvort,perc[pvival[m]],color=color[m])
    ax2.plot(relvort,sumedpvdi[pvival[m]],color=color[m],ls='-')
#    print(pearsonr(relvort,sumedpvdi[pvival[m]]))
#    print(sumedpvdi[pvival[m]]/sumedpv)
ax.set_ylim(0,0.75)
ax2.set_ylim(-50,1700)
ax.tick_params(labelright=False,right=True,labelbottom=False,bottom=True)
ax2.tick_params(labelright=False,right=True,labeltop=False,top=True)
#print(pearsonr(relvort,(sumedpvdi[0.5] + sumedpvdi[0.75])/sumedpv))

ax2.plot(relvort,sumedpv,color='k')
ax2.legend(legend,fontsize=8,frameon=False,loc='upper left')
ax2.set_ylabel(r'$\Sigma$ PV [PVU]')
ax.set_ylabel(r'data points with PV [%]')
ax2.set_xlabel(r'rel. vort. [$\times 10^{-4}$ s$^{-1}$]')
#fig.savefig(sp + 'pv-anomaly-percentage.png',dpi=300, bbox_inches="tight")
plt.close('all')

