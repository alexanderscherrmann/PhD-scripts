#!/usr/bin/env bash
# S. Schemm
#--------------------

module 'load nco/5.0.0'



# members 
# 0400 0500 0600 0700 0800 0900 1000 1100 1200 1300 1400 1500
# present day
#inp="/net/meso/climphys/cesm212/b.e212.BHISTcmip6.f09_g17.0900/archive/atm/hist/"
inp='/home/ascherrmann/scripts/CESM/'
# future climate
#inp="/net/meso/climphys/cesm212/b.e212.BSSP370cmip6.f09_g17.0900/archive/atm/hist/
out=${inp}

#out="/net/thermo/atmosdyn2/ascherrmann/015-CESM-WRF/ics/"

# --- input folder
#inp="/atmcirc/cesm212/b.e212.B1850cmip6.f09_g17.200.GRASS_NA/rerun/"
# --- output folder
#out="/atmcirc/cesm212/b.e212.B1850cmip6.f09_g17.200.GRASS_NA/plev/"

# --- create output folder
#mkdir -pv ${out}

# --- create destination grid: simple pure pressure grid (Units: Pa) ---
ncap2 -O -v -s \
	'defdim("plev",37);plev[$plev]={100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100};' \
	vrt_prs.nc
#
#ncap2 -O -v -s \
#    'defdim("plev",2);plev[$plev]={100000,97500};' \
#    vrt_prs.nc
#ncap2 -O -v -s \
#    'defdim("plev",20);plev[$plev]={100000,97500,92500,85000,70000,60000,50000,40000,30000,25000,20000,15000,10000,5000,3000,2000,1000,500,300,100};' \
#    vrt_prs.nc
# --- interpolate from input file to vertical grid in vrt_prs ----
# --- Note: Names on input file must include hyai, hyam, hybi, hybm, ilev, lev, P0, and PS (for E3SM/CESM hybrid grids)
# ---  or lev, lev_2, and lnsp (for ECMWF hybrid grids),
# ---  or plev (for pure-pressure input grids).

#for f in $( ls ${inp}"b."*"cam.h3"*".nc" ); do
#for f in "${inp}/test.nc" #b.e212.BHISTcmip6.f09_g17.0900.cam.h3.1988-01-01-00000.nc
#do

    # write to a new file
#    echo "Vertical interpolate to: " ${out}/$( basename "$f" )
#    time ncremap -t 4 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log ${f} ${out}/"interpolation-$( basename "$f")"

    # add surface pressure which otherwise is lost
#    ncks -A -v PS ${f} ${out}/$( basename "$f" )

    # compress the file [not needed for most]
    #nczip ${out}/$( basename "$f" )

    # overwrite the original file [dangerous]
    #\mv ${out}/$( basename "$f" ) ${f}

#done

# -t: Number of OpenMP Threads
# -v: Variable to interpolate
# --vrt_xtr=mss_val extrapolate to missing value
# --vrt_ntp=log pressure level interpolation logarithmic

# -- ;-)
exit 0

