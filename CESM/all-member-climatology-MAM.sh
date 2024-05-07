#!/usr/bin/env bash

module load nco/5.0.0

# members 
# 0400 0500 0600 0700 0800 0900 1000 1100 1200 1300 1400 1500

HIST="BHISTcmip6.f09_g17."
PROJ="BSSP370cmip6.f09_g17."
op="/net/meso/climphys/cesm212/b.e212."
cpp="/archive/atm/hist/"
in="/net/thermo/atmosdyn2/ascherrmann/015-CESM-WRF/MAM/"
#memb="0900"
mkdir ${in}

for memb in 0900 1000 1100 1200 1300
do
	inp="${in}${memb}/"
	mkdir ${inp}

	cd ${inp}
	echo $(pwd)
	
	ncap2 -O -v -s \
		'defdim("plev",37);plev[$plev]={100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100};' \
		vrt_prs.nc
	
	gnt=0
	for y in {1981..2010}
	do
		fname="n-time-steps-MAM-${y}.txt"
		echo $fname
	
		### link file here
		ln -sf ${op}${HIST}${memb}${cpp}*h3.${y}-01-01-00000.nc b-MAM-${y}.nc
	
		# select time range of interest here DJF
		nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-05-31T23:00:00,2 b-MAM-${y}.nc time-sel-MAM-${y}.nc
	
		rm b-MAM-${y}.nc	
	#	# calculate P use it for RH and set RH everywhere to 100 when RH>100
		nice -5 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-MAM-${y}.nc RH-MAM-${y}.nc	
	
		rm time-sel-MAM-${y}.nc
	        #interpolate to pressure levels defined above in vrt_prs.nc
		nice -5 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-MAM-${y}.nc interpolated-MAM-${y}.nc
	
		# get seasonal and monthly sums
		nice -5 cdo timsum interpolated-MAM-${y}.nc sum-MAM-${y}.nc
		nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-03-31T23:00:00 interpolated-MAM-${y}.nc MAR-${y}.nc
	        nice -5 ncks -d time,${y}-04-01T00:00:00,${y}-04-30T23:00:00 interpolated-MAM-${y}.nc APR-${y}.nc
		nice -5 ncks -d time,${y}-05-01T00:00:00,${y}-05-31T23:00:00 interpolated-MAM-${y}.nc MAY-${y}.nc
	
		# get number of time steps
		MARnt=$(cdo ntime MAR-${y}.nc)
		APRnt=$(cdo ntime APR-${y}.nc)
		MAYnt=$(cdo ntime MAY-${y}.nc)
	
		# get number of time steps in that season and add to total time steps in the climatology and write it in file
	        nt=$(cdo ntime interpolated-MAM-${y}.nc)
	        let "gnt += ${nt}"
	
		# save timesteps
		cat > ${fname} << EOF
	${nt}
	${MARnt}
	${APRnt}
	${MAYnt}
EOF
	        # get time-sums for individual months
		nice -5 cdo timsum MAR-${y}.nc sum-MAR-${y}.nc
		nice -5 cdo timsum APR-${y}.nc sum-APR-${y}.nc
		nice -5 cdo timsum MAY-${y}.nc sum-MAY-${y}.nc
	
	        # clean up
		rm MAR-${y}.nc APR-${y}.nc MAY-${y}.nc interpolated-MAM-${y}.nc RH-MAM-${y}.nc
	
	done
	
	fname="total-n-time-steps-MAM-1981-2010.txt"
	cat > ${fname} << EOF
	${gnt}
EOF
	
	gnt=0
	for y in {2011..2014}
	do
		fname="n-time-steps-MAM-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${HIST}${memb}${cpp}*h3.${y}-01-01-00000.nc b-MAM-${y}.nc
	
	        # select time range of interest here DJF
		nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-05-31T23:00:00,2 b-MAM-${y}.nc time-sel-MAM-${y}.nc

                rm b-MAM-${y}.nc
        #       # calculate P use it for RH and set RH everywhere to 100 when RH>100
                nice -5 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-MAM-${y}.nc RH-MAM-${y}.nc

                rm time-sel-MAM-${y}.nc
                #interpolate to pressure levels defined above in vrt_prs.nc
                nice -5 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-MAM-${y}.nc interpolated-MAM-${y}.nc

                # get seasonal and monthly sums
                nice -5 cdo timsum interpolated-MAM-${y}.nc sum-MAM-${y}.nc
                nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-03-31T23:00:00 interpolated-MAM-${y}.nc MAR-${y}.nc
                nice -5 ncks -d time,${y}-04-01T00:00:00,${y}-04-30T23:00:00 interpolated-MAM-${y}.nc APR-${y}.nc
                nice -5 ncks -d time,${y}-05-01T00:00:00,${y}-05-31T23:00:00 interpolated-MAM-${y}.nc MAY-${y}.nc

                # get number of time steps
                MARnt=$(cdo ntime MAR-${y}.nc)
                APRnt=$(cdo ntime APR-${y}.nc)
                MAYnt=$(cdo ntime MAY-${y}.nc)

                # get number of time steps in that season and add to total time steps in the climatology and write it in file
                nt=$(cdo ntime interpolated-MAM-${y}.nc)
                let "gnt += ${nt}"

                # save timesteps
                cat > ${fname} << EOF
        ${nt}
        ${MARnt}
        ${APRnt}
        ${MAYnt}
EOF
                # get time-sums for individual months
                nice -5 cdo timsum MAR-${y}.nc sum-MAR-${y}.nc
                nice -5 cdo timsum APR-${y}.nc sum-APR-${y}.nc
                nice -5 cdo timsum MAY-${y}.nc sum-MAY-${y}.nc

                # clean up
                rm MAR-${y}.nc APR-${y}.nc MAY-${y}.nc interpolated-MAM-${y}.nc RH-MAM-${y}.nc	
	done
	
	for y in {2015..2040}
	do
		fname="n-time-steps-MAM-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc b-MAM-${y}.nc
	
	        # select time range of interest here DJF
		nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-05-31T23:00:00,2 b-MAM-${y}.nc time-sel-MAM-${y}.nc

                rm b-MAM-${y}.nc
        #       # calculate P use it for RH and set RH everywhere to 100 when RH>100
                nice -5 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-MAM-${y}.nc RH-MAM-${y}.nc

                rm time-sel-MAM-${y}.nc
                #interpolate to pressure levels defined above in vrt_prs.nc
                nice -5 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-MAM-${y}.nc interpolated-MAM-${y}.nc

                # get seasonal and monthly sums
                nice -5 cdo timsum interpolated-MAM-${y}.nc sum-MAM-${y}.nc
                nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-03-31T23:00:00 interpolated-MAM-${y}.nc MAR-${y}.nc
                nice -5 ncks -d time,${y}-04-01T00:00:00,${y}-04-30T23:00:00 interpolated-MAM-${y}.nc APR-${y}.nc
                nice -5 ncks -d time,${y}-05-01T00:00:00,${y}-05-31T23:00:00 interpolated-MAM-${y}.nc MAY-${y}.nc

                # get number of time steps
                MARnt=$(cdo ntime MAR-${y}.nc)
                APRnt=$(cdo ntime APR-${y}.nc)
                MAYnt=$(cdo ntime MAY-${y}.nc)

                # get number of time steps in that season and add to total time steps in the climatology and write it in file
                nt=$(cdo ntime interpolated-MAM-${y}.nc)
                let "gnt += ${nt}"

                # save timesteps
                cat > ${fname} << EOF
        ${nt}
        ${MARnt}
        ${APRnt}
        ${MAYnt}
EOF
                # get time-sums for individual months
                nice -5 cdo timsum MAR-${y}.nc sum-MAR-${y}.nc
                nice -5 cdo timsum APR-${y}.nc sum-APR-${y}.nc
                nice -5 cdo timsum MAY-${y}.nc sum-MAY-${y}.nc

                # clean up
                rm MAR-${y}.nc APR-${y}.nc MAY-${y}.nc interpolated-MAM-${y}.nc RH-MAM-${y}.nc	
	done
	
	fname="total-n-time-steps-MAM-2011-2040.txt"
	cat > ${fname} << EOF
	${gnt}
EOF
	
	
	gnt=0
	for y in {2041..2070}
	do
		fname="n-time-steps-MAM-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc b-MAM-${y}.nc
	
	        # select time range of interest here 
		nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-05-31T23:00:00,2 b-MAM-${y}.nc time-sel-MAM-${y}.nc

                rm b-MAM-${y}.nc
        #       # calculate P use it for RH and set RH everywhere to 100 when RH>100
                nice -5 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-MAM-${y}.nc RH-MAM-${y}.nc

                rm time-sel-MAM-${y}.nc
                #interpolate to pressure levels defined above in vrt_prs.nc
                nice -5 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-MAM-${y}.nc interpolated-MAM-${y}.nc

                # get seasonal and monthly sums
                nice -5 cdo timsum interpolated-MAM-${y}.nc sum-MAM-${y}.nc
                nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-03-31T23:00:00 interpolated-MAM-${y}.nc MAR-${y}.nc
                nice -5 ncks -d time,${y}-04-01T00:00:00,${y}-04-30T23:00:00 interpolated-MAM-${y}.nc APR-${y}.nc
                nice -5 ncks -d time,${y}-05-01T00:00:00,${y}-05-31T23:00:00 interpolated-MAM-${y}.nc MAY-${y}.nc

                # get number of time steps
                MARnt=$(cdo ntime MAR-${y}.nc)
                APRnt=$(cdo ntime APR-${y}.nc)
                MAYnt=$(cdo ntime MAY-${y}.nc)

                # get number of time steps in that season and add to total time steps in the climatology and write it in file
                nt=$(cdo ntime interpolated-MAM-${y}.nc)
                let "gnt += ${nt}"

                # save timesteps
                cat > ${fname} << EOF
        ${nt}
        ${MARnt}
        ${APRnt}
        ${MAYnt}
EOF
                # get time-sums for individual months
                nice -5 cdo timsum MAR-${y}.nc sum-MAR-${y}.nc
                nice -5 cdo timsum APR-${y}.nc sum-APR-${y}.nc
                nice -5 cdo timsum MAY-${y}.nc sum-MAY-${y}.nc

                # clean up
                rm MAR-${y}.nc APR-${y}.nc MAY-${y}.nc interpolated-MAM-${y}.nc RH-MAM-${y}.nc	
	done
	
	fname="total-n-time-steps-MAM-2041-2070.txt"
	cat > ${fname} << EOF
	${gnt}
EOF
	
	gnt=0
	for y in {2071..2100}
	do
		fname="n-time-steps-MAM-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc b-MAM-${y}.nc
	
	        # select time range of interest here DJF
		nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-05-31T23:00:00,2 b-MAM-${y}.nc time-sel-MAM-${y}.nc

                rm b-MAM-${y}.nc
        #       # calculate P use it for RH and set RH everywhere to 100 when RH>100
                nice -5 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-MAM-${y}.nc RH-MAM-${y}.nc

                rm time-sel-MAM-${y}.nc
                #interpolate to pressure levels defined above in vrt_prs.nc
                nice -5 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-MAM-${y}.nc interpolated-MAM-${y}.nc

                # get seasonal and monthly sums
                nice -5 cdo timsum interpolated-MAM-${y}.nc sum-MAM-${y}.nc
                nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-03-31T23:00:00 interpolated-MAM-${y}.nc MAR-${y}.nc
                nice -5 ncks -d time,${y}-04-01T00:00:00,${y}-04-30T23:00:00 interpolated-MAM-${y}.nc APR-${y}.nc
                nice -5 ncks -d time,${y}-05-01T00:00:00,${y}-05-31T23:00:00 interpolated-MAM-${y}.nc MAY-${y}.nc

                # get number of time steps
                MARnt=$(cdo ntime MAR-${y}.nc)
                APRnt=$(cdo ntime APR-${y}.nc)
                MAYnt=$(cdo ntime MAY-${y}.nc)

                # get number of time steps in that season and add to total time steps in the climatology and write it in file
                nt=$(cdo ntime interpolated-MAM-${y}.nc)
                let "gnt += ${nt}"

                # save timesteps
                cat > ${fname} << EOF
        ${nt}
        ${MARnt}
        ${APRnt}
        ${MAYnt}
EOF
                # get time-sums for individual months
                nice -5 cdo timsum MAR-${y}.nc sum-MAR-${y}.nc
                nice -5 cdo timsum APR-${y}.nc sum-APR-${y}.nc
                nice -5 cdo timsum MAY-${y}.nc sum-MAY-${y}.nc

                # clean up
                rm MAR-${y}.nc APR-${y}.nc MAY-${y}.nc interpolated-MAM-${y}.nc RH-MAM-${y}.nc	
	done
	
	fname="total-n-time-steps-MAM-2071-2100.txt"
	cat > ${fname} << EOF
	${gnt}
EOF

done
exit 0

