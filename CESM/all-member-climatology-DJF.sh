#!/usr/bin/env bash

module load nco/5.0.0

# members 
# 0400 0500 0600 0700 0800 0900 1000 1100 1200 1300 1400 1500

HIST="BHISTcmip6.f09_g17."
PROJ="BSSP370cmip6.f09_g17."
op="/net/meso/climphys/cesm212/b.e212."
cpp="/archive/atm/hist/"
in="/net/thermo/atmosdyn2/ascherrmann/015-CESM-WRF/"
#memb="0900"

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
	yy=0
	for y in {1981..2010}
	do
		fname="n-time-steps-${y}.txt"
		echo $fname
	
		### link file here
		ln -sf ${op}${HIST}${memb}${cpp}*h3.${y}-01-01-00000.nc .
	
		# select time range of interest here DJF
		let "yy = ${y}+1"	
		echo $yy
		nice -3 ncks -d time,${y}-12-01T00:00:00,${yy}-01-01T00:00:00,2 -d time,${y}-01-01T00:00:00,${y}-02-28T23:00:00,2 *h3.${y}-01-01-00000.nc time-sel-${y}.nc
	
		rm *h3.${y}-01-01-00000.nc
	
	#	# calculate P use it for RH and set RH everywhere to 100 when RH>100
		nice -3 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-${y}.nc RH-${y}.nc	
	
		rm time-sel-${y}.nc
	        #interpolate to pressure levels defined above in vrt_prs.nc
		nice -3 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-${y}.nc interpolated-${y}.nc
	
		# get seasonal and monthly sums
		nice -3 cdo timsum interpolated-${y}.nc sum-DJF-${y}.nc
		nice -3 ncks -d time,${y}-12-01T00:00:00,${yy}-01-01T00:00:00 interpolated-${y}.nc DEC-${y}.nc
	        nice -3 ncks -d time,${y}-01-01T00:00:00,${y}-01-31T23:00:00 interpolated-${y}.nc JAN-${y}.nc
		nice -3 ncks -d time,${y}-02-01T00:00:00,${y}-02-28T23:00:00 interpolated-${y}.nc FEB-${y}.nc
	
		# get number of time steps
		DECnt=$(cdo ntime DEC-${y}.nc)
		JANnt=$(cdo ntime JAN-${y}.nc)
		FEBnt=$(cdo ntime FEB-${y}.nc)
	
		# get number of time steps in that season and add to total time steps in the climatology and write it in file
	        nt=$(cdo ntime interpolated-${y}.nc)
	        let "gnt += ${nt}"
	
		# save timesteps
		cat > ${fname} << EOF
	${nt}
	${DECnt}
	${JANnt}
	${FEBnt}
EOF
	        # get time-sums for individual months
		nice -3 cdo timsum DEC-${y}.nc sum-DEC-${y}.nc
		nice -3 cdo timsum JAN-${y}.nc sum-JAN-${y}.nc
		nice -3 cdo timsum FEB-${y}.nc sum-FEB-${y}.nc
	
	        # clean up
		rm DEC-${y}.nc JAN-${y}.nc FEB-${y}.nc interpolated-${y}.nc RH-${y}.nc
	
	done
	
	fname="total-n-time-steps-1981-2010.txt"
	cat > ${fname} << EOF
	${gnt}
EOF
	
	gnt=0
	for y in {2011..2014}
	do
		fname="n-time-steps-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${HIST}${memb}${cpp}*h3.${y}-01-01-00000.nc .
	
	        # select time range of interest here DJF
	        let "yy = ${y}+1"
	        echo $yy
	        nice -3 ncks -d time,${y}-12-01T00:00:00,${yy}-01-01T00:00:00,2 -d time,${y}-01-01T00:00:00,${y}-02-28T23:00:00,2 *h3.${y}-01-01-00000.nc time-sel-${y}.nc
	
	        rm *h3.${y}-01-01-00000.nc
	
	#       # calculate P use it for RH and set RH everywhere to 100 when RH>100
	        nice -3 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-${y}.nc RH-${y}.nc
	
	        rm time-sel-${y}.nc
	        #interpolate to pressure levels defined above in vrt_prs.nc
	        nice -3 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-${y}.nc interpolated-${y}.nc
	
	        # get seasonal and monthly sums
	        nice -3 cdo timsum interpolated-${y}.nc sum-DJF-${y}.nc
	        nice -3 ncks -d time,${y}-12-01T00:00:00,${yy}-01-01T00:00:00 interpolated-${y}.nc DEC-${y}.nc
	        nice -3 ncks -d time,${y}-01-01T00:00:00,${y}-01-31T23:00:00 interpolated-${y}.nc JAN-${y}.nc
	        nice -3 ncks -d time,${y}-02-01T00:00:00,${y}-02-28T23:00:00 interpolated-${y}.nc FEB-${y}.nc
	
	        # get number of time steps
	        DECnt=$(cdo ntime DEC-${y}.nc)
	        JANnt=$(cdo ntime JAN-${y}.nc)
	        FEBnt=$(cdo ntime FEB-${y}.nc)
	
	        # get number of time steps in that season and add to total time steps in the climatology and write it in file
	        nt=$(cdo ntime interpolated-${y}.nc)
	        let "gnt += ${nt}"
	
	        # save timesteps
	        cat > ${fname} << EOF
	${nt}
	${DECnt}
	${JANnt}
	${FEBnt}
EOF
	        # get time-sums for individual months
	        nice -3 cdo timsum DEC-${y}.nc sum-DEC-${y}.nc
	        nice -3 cdo timsum JAN-${y}.nc sum-JAN-${y}.nc
	        nice -3 cdo timsum FEB-${y}.nc sum-FEB-${y}.nc
	
	        # clean up
	        rm DEC-${y}.nc JAN-${y}.nc FEB-${y}.nc interpolated-${y}.nc RH-${y}.nc
	
	done
	
	for y in {2015..2040}
	do
		fname="n-time-steps-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc .
	
	        # select time range of interest here DJF
	        let "yy = ${y}+1"
	        echo $yy
	        nice -3 ncks -d time,${y}-12-01T00:00:00,${yy}-01-01T00:00:00,2 -d time,${y}-01-01T00:00:00,${y}-02-28T23:00:00,2 *h3.${y}-01-01-00000.nc time-sel-${y}.nc
	
	        rm *h3.${y}-01-01-00000.nc
	
	#       # calculate P use it for RH and set RH everywhere to 100 when RH>100
	        nice -3 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-${y}.nc RH-${y}.nc
	
	        rm time-sel-${y}.nc
	        #interpolate to pressure levels defined above in vrt_prs.nc
	        nice -3 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-${y}.nc interpolated-${y}.nc
	
	        # get seasonal and monthly sums
	        nice -3 cdo timsum interpolated-${y}.nc sum-DJF-${y}.nc
	        nice -3 ncks -d time,${y}-12-01T00:00:00,${yy}-01-01T00:00:00 interpolated-${y}.nc DEC-${y}.nc
	        nice -3 ncks -d time,${y}-01-01T00:00:00,${y}-01-31T23:00:00 interpolated-${y}.nc JAN-${y}.nc
	        nice -3 ncks -d time,${y}-02-01T00:00:00,${y}-02-28T23:00:00 interpolated-${y}.nc FEB-${y}.nc
	
	        # get number of time steps
	        DECnt=$(cdo ntime DEC-${y}.nc)
	        JANnt=$(cdo ntime JAN-${y}.nc)
	        FEBnt=$(cdo ntime FEB-${y}.nc)
	
	        # get number of time steps in that season and add to total time steps in the climatology and write it in file
	        nt=$(cdo ntime interpolated-${y}.nc)
	        let "gnt += ${nt}"
	
	        # save timesteps
	        cat > ${fname} << EOF
	${nt}
	${DECnt}
	${JANnt}
	${FEBnt}
EOF
	        # get time-sums for individual months
	        nice -3 cdo timsum DEC-${y}.nc sum-DEC-${y}.nc
	        nice -3 cdo timsum JAN-${y}.nc sum-JAN-${y}.nc
	        nice -3 cdo timsum FEB-${y}.nc sum-FEB-${y}.nc
	
	        # clean up
	        rm DEC-${y}.nc JAN-${y}.nc FEB-${y}.nc interpolated-${y}.nc RH-${y}.nc
	
	done
	
	fname="total-n-time-steps-2011-2040.txt"
	cat > ${fname} << EOF
	${gnt}
EOF
	
	
	gnt=0
	for y in {2041..2070}
	do
		fname="n-time-steps-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc .
	
	        # select time range of interest here DJF
	        let "yy = ${y}+1"
	        echo $yy
	        nice -3 ncks -d time,${y}-12-01T00:00:00,${yy}-01-01T00:00:00,2 -d time,${y}-01-01T00:00:00,${y}-02-28T23:00:00,2 *h3.${y}-01-01-00000.nc time-sel-${y}.nc
	
	        rm *h3.${y}-01-01-00000.nc
	
	#       # calculate P use it for RH and set RH everywhere to 100 when RH>100
	        nice -3 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000 *100+ hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-${y}.nc RH-${y}.nc
	
	        rm time-sel-${y}.nc
	        #interpolate to pressure levels defined above in vrt_prs.nc
	        nice -3 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-${y}.nc interpolated-${y}.nc
	
	        # get seasonal and monthly sums
	        nice -3 cdo timsum interpolated-${y}.nc sum-DJF-${y}.nc
	        nice -3 ncks -d time,${y}-12-01T00:00:00,${yy}-01-01T00:00:00 interpolated-${y}.nc DEC-${y}.nc
	        nice -3 ncks -d time,${y}-01-01T00:00:00,${y}-01-31T23:00:00 interpolated-${y}.nc JAN-${y}.nc
	        nice -3 ncks -d time,${y}-02-01T00:00:00,${y}-02-28T23:00:00 interpolated-${y}.nc FEB-${y}.nc
	
	        # get number of time steps
	        DECnt=$(cdo ntime DEC-${y}.nc)
	        JANnt=$(cdo ntime JAN-${y}.nc)
	        FEBnt=$(cdo ntime FEB-${y}.nc)
	
	        # get number of time steps in that season and add to total time steps in the climatology and write it in file
	        nt=$(cdo ntime interpolated-${y}.nc)
	        let "gnt += ${nt}"
	
	        # save timesteps
	        cat > ${fname} << EOF
	${nt}
	${DECnt}
	${JANnt}
	${FEBnt}
EOF
	        # get time-sums for individual months
	        nice -3 cdo timsum DEC-${y}.nc sum-DEC-${y}.nc
	        nice -3 cdo timsum JAN-${y}.nc sum-JAN-${y}.nc
	        nice -3 cdo timsum FEB-${y}.nc sum-FEB-${y}.nc
	
	        # clean up
	        rm DEC-${y}.nc JAN-${y}.nc FEB-${y}.nc interpolated-${y}.nc RH-${y}.nc
	
	done
	
	fname="total-n-time-steps-2041-2070.txt"
	cat > ${fname} << EOF
	${gnt}
EOF
	
	gnt=0
	for y in {2071..2100}
	do
		fname="n-time-steps-${y}.txt"
	        echo $fname
	
	        ### link file here
	        ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc .
	
	        # select time range of interest here DJF
	        let "yy = ${y}+1"
	        echo $yy
	        nice -3 ncks -d time,${y}-12-01T00:00:00,${yy}-01-01T00:00:00,2 -d time,${y}-01-01T00:00:00,${y}-02-28T23:00:00,2 *h3.${y}-01-01-00000.nc time-sel-${y}.nc
	
	        rm *h3.${y}-01-01-00000.nc
	
	#       # calculate P use it for RH and set RH everywhere to 100 when RH>100
	        nice -3 ncap2 -t 2 -s 'P[time,lev,lat,lon]=(hyam*1000*100 + hybm*PS); RH=(0.263 * P * Q/(exp(17.67 * (T-273.16)/(T-29.65)))); where(RH>100) RH=100;' time-sel-${y}.nc RH-${y}.nc
	
	        rm time-sel-${y}.nc
	        #interpolate to pressure levels defined above in vrt_prs.nc
	        nice -3 ncremap -t 2 --vrt_fl=vrt_prs.nc --vrt_xtr=nrs_ngh --vrt_ntp=log -v U,V,T,RH,Z3 RH-${y}.nc interpolated-${y}.nc
	
	        # get seasonal and monthly sums
	        nice -3 cdo timsum interpolated-${y}.nc sum-DJF-${y}.nc
	        nice -3 ncks -d time,${y}-12-01T00:00:00,${yy}-01-01T00:00:00 interpolated-${y}.nc DEC-${y}.nc
	        nice -3 ncks -d time,${y}-01-01T00:00:00,${y}-01-31T23:00:00 interpolated-${y}.nc JAN-${y}.nc
	        nice -3 ncks -d time,${y}-02-01T00:00:00,${y}-02-28T23:00:00 interpolated-${y}.nc FEB-${y}.nc
	
	        # get number of time steps
	        DECnt=$(cdo ntime DEC-${y}.nc)
	        JANnt=$(cdo ntime JAN-${y}.nc)
	        FEBnt=$(cdo ntime FEB-${y}.nc)
	
	        # get number of time steps in that season and add to total time steps in the climatology and write it in file
	        nt=$(cdo ntime interpolated-${y}.nc)
	        let "gnt += ${nt}"
	
	        # save timesteps
	        cat > ${fname} << EOF
	${nt}
	${DECnt}
	${JANnt}
	${FEBnt}
EOF
	        # get time-sums for individual months
	        nice -3 cdo timsum DEC-${y}.nc sum-DEC-${y}.nc
	        nice -3 cdo timsum JAN-${y}.nc sum-JAN-${y}.nc
	        nice -3 cdo timsum FEB-${y}.nc sum-FEB-${y}.nc
	
	        # clean up
	        rm DEC-${y}.nc JAN-${y}.nc FEB-${y}.nc interpolated-${y}.nc RH-${y}.nc
	
	done
	
	fname="total-n-time-steps-2071-2100.txt"
	cat > ${fname} << EOF
	${gnt}
EOF

done
exit 0

