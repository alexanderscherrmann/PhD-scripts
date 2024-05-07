#!/usr/bin/env bash

module load nco/5.0.0

# members 
# 0400 0500 0600 0700 0800 0900 1000 1100 1200 1300 1400 1500

HIST="BHISTcmip6.f09_g17."
PROJ="BSSP370cmip6.f09_g17."
op="/net/meso/climphys/cesm212/b.e212."
cpp="/archive/atm/hist/"
in="/net/thermo/atmosdyn2/ascherrmann/015-CESM-WRF/SON/"
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
		fname="n-time-steps-SON-${y}.txt"
		echo $fname
	
		### link file here
		ln -sf ${op}${HIST}${memb}${cpp}*h3.${y}-01-01-00000.nc b-SON-${y}.nc
	
		# select time range of interest here DJF
		nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-11-30T23:00:00,2 b-SON-${y}.nc time-sel-SON-${y}.nc
	
		rm b-SON-${y}.nc	
	#	# calculate P use it for RH and set RH everywhere to 100 when RH>100
		nice -5 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-SON-${y}.nc RH-SON-${y}.nc	
	
		rm time-sel-SON-${y}.nc
	        #interpolate to pressure levels defined above in vrt_prs.nc
		nice -5 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-SON-${y}.nc interpolated-SON-${y}.nc
	
		# get seasonal and monthly sums
		nice -5 cdo timsum interpolated-SON-${y}.nc sum-SON-${y}.nc
		nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-09-30T23:00:00 interpolated-SON-${y}.nc SEP-${y}.nc
	        nice -5 ncks -d time,${y}-10-01T00:00:00,${y}-10-30T23:00:00 interpolated-SON-${y}.nc OCT-${y}.nc
		nice -5 ncks -d time,${y}-11-01T00:00:00,${y}-11-30T23:00:00 interpolated-SON-${y}.nc NOV-${y}.nc
	
		# get number of time steps
		SEPnt=$(cdo ntime SEP-${y}.nc)
		OCTnt=$(cdo ntime OCT-${y}.nc)
		NOVnt=$(cdo ntime NOV-${y}.nc)
	
		# get number of time steps in that season and add to total time steps in the climatology and write it in file
	        nt=$(cdo ntime interpolated-SON-${y}.nc)
	        let "gnt += ${nt}"
	
		# save timesteps
		cat > ${fname} << EOF
${nt}
${SEPnt}
${OCTnt}
${NOVnt}
EOF
	        # get time-sums for individual months
		nice -5 cdo timsum SEP-${y}.nc sum-SEP-${y}.nc
		nice -5 cdo timsum OCT-${y}.nc sum-OCT-${y}.nc
		nice -5 cdo timsum NOV-${y}.nc sum-NOV-${y}.nc
	
	        # clean up
		rm SEP-${y}.nc OCT-${y}.nc NOV-${y}.nc interpolated-SON-${y}.nc RH-SON-${y}.nc
	
	done
	
	fname="total-n-time-steps-SON-1981-2010.txt"
	cat > ${fname} << EOF
${gnt}
EOF
	
	gnt=0
	for y in {2011..2014}
	do
		fname="n-time-steps-SON-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${HIST}${memb}${cpp}*h3.${y}-01-01-00000.nc b-SON-${y}.nc
	
	        # select time range of interest here DJF
		nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-11-30T23:00:00,2 b-SON-${y}.nc time-sel-SON-${y}.nc

                rm b-SON-${y}.nc
        #       # calculate P use it for RH and set RH everywhere to 100 when RH>100
                nice -5 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-SON-${y}.nc RH-SON-${y}.nc

                rm time-sel-SON-${y}.nc
                #interpolate to pressure levels defined above in vrt_prs.nc
                nice -5 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-SON-${y}.nc interpolated-SON-${y}.nc

                # get seasonal and monthly sums
                nice -5 cdo timsum interpolated-SON-${y}.nc sum-SON-${y}.nc
                nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-09-30T23:00:00 interpolated-SON-${y}.nc SEP-${y}.nc
                nice -5 ncks -d time,${y}-10-01T00:00:00,${y}-10-30T23:00:00 interpolated-SON-${y}.nc OCT-${y}.nc
                nice -5 ncks -d time,${y}-11-01T00:00:00,${y}-11-30T23:00:00 interpolated-SON-${y}.nc NOV-${y}.nc

                # get number of time steps
                SEPnt=$(cdo ntime SEP-${y}.nc)
                OCTnt=$(cdo ntime OCT-${y}.nc)
                NOVnt=$(cdo ntime NOV-${y}.nc)

                # get number of time steps in that season and add to total time steps in the climatology and write it in file
                nt=$(cdo ntime interpolated-SON-${y}.nc)
                let "gnt += ${nt}"

                # save timesteps
                cat > ${fname} << EOF
${nt}
${SEPnt}
${OCTnt}
${NOVnt}
EOF
                # get time-sums for individual months
                nice -5 cdo timsum SEP-${y}.nc sum-SEP-${y}.nc
                nice -5 cdo timsum OCT-${y}.nc sum-OCT-${y}.nc
                nice -5 cdo timsum NOV-${y}.nc sum-NOV-${y}.nc

                # clean up
                rm SEP-${y}.nc OCT-${y}.nc NOV-${y}.nc interpolated-SON-${y}.nc RH-SON-${y}.nc	
	done
	
	for y in {2015..2040}
	do
		fname="n-time-steps-SON-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc b-SON-${y}.nc
	
	        # select time range of interest here DJF
		nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-11-30T23:00:00,2 b-SON-${y}.nc time-sel-SON-${y}.nc

                rm b-SON-${y}.nc
        #       # calculate P use it for RH and set RH everywhere to 100 when RH>100
                nice -5 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-SON-${y}.nc RH-SON-${y}.nc

                rm time-sel-SON-${y}.nc
                #interpolate to pressure levels defined above in vrt_prs.nc
                nice -5 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-SON-${y}.nc interpolated-SON-${y}.nc

                # get seasonal and monthly sums
                nice -5 cdo timsum interpolated-SON-${y}.nc sum-SON-${y}.nc
                nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-09-30T23:00:00 interpolated-SON-${y}.nc SEP-${y}.nc
                nice -5 ncks -d time,${y}-10-01T00:00:00,${y}-10-30T23:00:00 interpolated-SON-${y}.nc OCT-${y}.nc
                nice -5 ncks -d time,${y}-11-01T00:00:00,${y}-11-30T23:00:00 interpolated-SON-${y}.nc NOV-${y}.nc

                # get number of time steps
                SEPnt=$(cdo ntime SEP-${y}.nc)
                OCTnt=$(cdo ntime OCT-${y}.nc)
                NOVnt=$(cdo ntime NOV-${y}.nc)

                # get number of time steps in that season and add to total time steps in the climatology and write it in file
                nt=$(cdo ntime interpolated-SON-${y}.nc)
                let "gnt += ${nt}"

                # save timesteps
                cat > ${fname} << EOF
${nt}
${SEPnt}
${OCTnt}
${NOVnt}
EOF
                # get time-sums for individual months
                nice -5 cdo timsum SEP-${y}.nc sum-SEP-${y}.nc
                nice -5 cdo timsum OCT-${y}.nc sum-OCT-${y}.nc
                nice -5 cdo timsum NOV-${y}.nc sum-NOV-${y}.nc

                # clean up
                rm SEP-${y}.nc OCT-${y}.nc NOV-${y}.nc interpolated-SON-${y}.nc RH-SON-${y}.nc	
	done
	
	fname="total-n-time-steps-SON-2011-2040.txt"
	cat > ${fname} << EOF
${gnt}
EOF
	
	
	gnt=0
	for y in {2041..2070}
	do
		fname="n-time-steps-SON-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc b-SON-${y}.nc
	
	        # select time range of interest here 
		nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-11-30T23:00:00,2 b-SON-${y}.nc time-sel-SON-${y}.nc

                rm b-SON-${y}.nc
        #       # calculate P use it for RH and set RH everywhere to 100 when RH>100
                nice -5 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-SON-${y}.nc RH-SON-${y}.nc

                rm time-sel-SON-${y}.nc
                #interpolate to pressure levels defined above in vrt_prs.nc
                nice -5 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-SON-${y}.nc interpolated-SON-${y}.nc

                # get seasonal and monthly sums
                nice -5 cdo timsum interpolated-SON-${y}.nc sum-SON-${y}.nc
                nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-09-30T23:00:00 interpolated-SON-${y}.nc SEP-${y}.nc
                nice -5 ncks -d time,${y}-10-01T00:00:00,${y}-10-30T23:00:00 interpolated-SON-${y}.nc OCT-${y}.nc
                nice -5 ncks -d time,${y}-11-01T00:00:00,${y}-11-30T23:00:00 interpolated-SON-${y}.nc NOV-${y}.nc

                # get number of time steps
                SEPnt=$(cdo ntime SEP-${y}.nc)
                OCTnt=$(cdo ntime OCT-${y}.nc)
                NOVnt=$(cdo ntime NOV-${y}.nc)

                # get number of time steps in that season and add to total time steps in the climatology and write it in file
                nt=$(cdo ntime interpolated-SON-${y}.nc)
                let "gnt += ${nt}"

                # save timesteps
                cat > ${fname} << EOF
${nt}
${SEPnt}
${OCTnt}
${NOVnt}
EOF
                # get time-sums for individual months
                nice -5 cdo timsum SEP-${y}.nc sum-SEP-${y}.nc
                nice -5 cdo timsum OCT-${y}.nc sum-OCT-${y}.nc
                nice -5 cdo timsum NOV-${y}.nc sum-NOV-${y}.nc

                # clean up
                rm SEP-${y}.nc OCT-${y}.nc NOV-${y}.nc interpolated-SON-${y}.nc RH-SON-${y}.nc	
	done
	
	fname="total-n-time-steps-SON-2041-2070.txt"
	cat > ${fname} << EOF
${gnt}
EOF
	
	gnt=0
	for y in {2071..2100}
	do
		fname="n-time-steps-SON-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc b-SON-${y}.nc
	
	        # select time range of interest here DJF
		nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-11-30T23:00:00,2 b-SON-${y}.nc time-sel-SON-${y}.nc

                rm b-SON-${y}.nc
        #       # calculate P use it for RH and set RH everywhere to 100 when RH>100
                nice -5 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-SON-${y}.nc RH-SON-${y}.nc

                rm time-sel-SON-${y}.nc
                #interpolate to pressure levels defined above in vrt_prs.nc
                nice -5 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-SON-${y}.nc interpolated-SON-${y}.nc

                # get seasonal and monthly sums
                nice -5 cdo timsum interpolated-SON-${y}.nc sum-SON-${y}.nc
                nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-09-30T23:00:00 interpolated-SON-${y}.nc SEP-${y}.nc
                nice -5 ncks -d time,${y}-10-01T00:00:00,${y}-10-30T23:00:00 interpolated-SON-${y}.nc OCT-${y}.nc
                nice -5 ncks -d time,${y}-11-01T00:00:00,${y}-11-30T23:00:00 interpolated-SON-${y}.nc NOV-${y}.nc

                # get number of time steps
                SEPnt=$(cdo ntime SEP-${y}.nc)
                OCTnt=$(cdo ntime OCT-${y}.nc)
                NOVnt=$(cdo ntime NOV-${y}.nc)

                # get number of time steps in that season and add to total time steps in the climatology and write it in file
                nt=$(cdo ntime interpolated-SON-${y}.nc)
                let "gnt += ${nt}"

                # save timesteps
                cat > ${fname} << EOF
${nt}
${SEPnt}
${OCTnt}
${NOVnt}
EOF
                # get time-sums for individual months
                nice -5 cdo timsum SEP-${y}.nc sum-SEP-${y}.nc
                nice -5 cdo timsum OCT-${y}.nc sum-OCT-${y}.nc
                nice -5 cdo timsum NOV-${y}.nc sum-NOV-${y}.nc

                # clean up
                rm SEP-${y}.nc OCT-${y}.nc NOV-${y}.nc interpolated-SON-${y}.nc RH-SON-${y}.nc	
	done
	
	fname="total-n-time-steps-SON-2071-2100.txt"
	cat > ${fname} << EOF
${gnt}
EOF

done
exit 0

