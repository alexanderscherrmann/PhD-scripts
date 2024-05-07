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
mkdir ${in}

for memb in 0900 1000 1100 1200 1300
do
	inp="${in}"

	cd ${inp}
	echo $(pwd)
	
	#for y in {1981..2010}
	#do
	#	### link file here

	#	ln -sf ${op}${HIST}${memb}${cpp}*h3.${y}-01-01-00000.nc b-${y}.nc
	#	cdo selvar,PS,PSL,SST,TS b-${y}.nc surface-fields-${y}.nc
	#	rm b-${y}.nc

#	#	ln -sf ${op}${HIST}${memb}${cpp}*h4.${y}-01-01-00000.nc b-${y}.nc
#	#	cdo selvar,RHREFHT,TREFHT b-${y}.nc surface2-${y}.nc
#	#	rm b-${y}.nc
#	#	cdo mergetime surface1-${y}.nc surface2-${y}.nc surface-fields-${y}.nc
#	#	rm surface1-${y}.nc surface2-${y}.nc
#
	#	nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-03-31T23:00:00,2 surface-fields-${y}.nc surface-MAR-${y}.nc
        #        nice -5 ncks -d time,${y}-04-01T00:00:00,${y}-04-30T23:00:00,2 surface-fields-${y}.nc surface-APR-${y}.nc
        #        nice -5 ncks -d time,${y}-05-01T00:00:00,${y}-05-31T23:00:00,2 surface-fields-${y}.nc surface-MAY-${y}.nc
	#	nice -5 ncks -d time,${y}-01-01T00:00:00,${y}-01-31T23:00:00,2 surface-fields-${y}.nc surface-JAN-${y}.nc
        #        nice -5 ncks -d time,${y}-02-01T00:00:00,${y}-02-28T23:00:00,2 surface-fields-${y}.nc surface-FEB-${y}.nc
        #        nice -5 ncks -d time,${y}-12-01T00:00:00,${y}-12-31T23:00:00,2 surface-fields-${y}.nc surface-DEC-${y}.nc
	#	nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-09-30T23:00:00,2 surface-fields-${y}.nc surface-SEP-${y}.nc
        #        nice -5 ncks -d time,${y}-10-01T00:00:00,${y}-10-31T23:00:00,2 surface-fields-${y}.nc surface-OCT-${y}.nc
        #        nice -5 ncks -d time,${y}-11-01T00:00:00,${y}-11-30T23:00:00,2 surface-fields-${y}.nc surface-NOV-${y}.nc
#	#		
#	#        # get time-sums for individual months
	#	nice -5 cdo timsum surface-MAR-${y}.nc surface-sum-MAR-${y}.nc
	#	nice -5 cdo timsum surface-APR-${y}.nc surface-sum-APR-${y}.nc
	#	nice -5 cdo timsum surface-MAY-${y}.nc surface-sum-MAY-${y}.nc
	#	nice -5 cdo timsum surface-JAN-${y}.nc surface-sum-JAN-${y}.nc
        #        nice -5 cdo timsum surface-FEB-${y}.nc surface-sum-FEB-${y}.nc
        #        nice -5 cdo timsum surface-DEC-${y}.nc surface-sum-DEC-${y}.nc
	#	nice -5 cdo timsum surface-SEP-${y}.nc surface-sum-SEP-${y}.nc
        #        nice -5 cdo timsum surface-OCT-${y}.nc surface-sum-OCT-${y}.nc
        #        nice -5 cdo timsum surface-NOV-${y}.nc surface-sum-NOV-${y}.nc
#	#	
	#	cdo enssum surface-sum-MAR-${y}.nc surface-sum-APR-${y}.nc surface-sum-MAY-${y}.nc surface-sum-MAM-${y}.nc
	#	cdo enssum surface-sum-JAN-${y}.nc surface-sum-FEB-${y}.nc surface-sum-DEC-${y}.nc surface-sum-DJF-${y}.nc
	#	cdo enssum surface-sum-SEP-${y}.nc surface-sum-OCT-${y}.nc surface-sum-NOV-${y}.nc surface-sum-SON-${y}.nc
#
	#	rm surface-fields-${y}.nc surface-MAR-${y}.nc surface-APR-${y}.nc surface-MAY-${y}.nc surface-JAN-${y}.nc surface-FEB-${y}.nc surface-DEC-${y}.nc surface-SEP-${y}.nc surface-OCT-${y}.nc surface-NOV-${y}.nc
	#	mv surface-sum-MAR-${y}.nc surface-sum-APR-${y}.nc surface-sum-MAY-${y}.nc surface-sum-MAM-${y}.nc ${inp}MAM/${memb}/
	#        mv surface-sum-JAN-${y}.nc surface-sum-FEB-${y}.nc surface-sum-DEC-${y}.nc surface-sum-DJF-${y}.nc ${inp}DJF/${memb}/
	#	mv surface-sum-SEP-${y}.nc surface-sum-OCT-${y}.nc surface-sum-NOV-${y}.nc surface-sum-SON-${y}.nc ${inp}SON/${memb}/	
	#done
	#
	#for y in {2011..2014}
	#do
	#	ln -sf ${op}${HIST}${memb}${cpp}*h3.${y}-01-01-00000.nc b-${y}.nc
	#	cdo selvar,PS,PSL,SST,TS b-${y}.nc surface-fields-${y}.nc
        #        rm b-${y}.nc

#       #        ln -sf ${op}${HIST}${memb}${cpp}*h4.${y}-01-01-00000.nc b-${y}.nc
#       #        cdo selvar,RHREFHT,TREFHT b-${y}.nc surface2-${y}.nc
#       #        rm b-${y}.nc
#       #        cdo mergetime surface1-${y}.nc surface2-${y}.nc surface-fields-${y}.nc
#       #        rm surface1-${y}.nc surface2-${y}.nc
#

        #        nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-03-31T23:00:00,2 surface-fields-${y}.nc surface-MAR-${y}.nc
        #        nice -5 ncks -d time,${y}-04-01T00:00:00,${y}-04-30T23:00:00,2 surface-fields-${y}.nc surface-APR-${y}.nc
        #        nice -5 ncks -d time,${y}-05-01T00:00:00,${y}-05-31T23:00:00,2 surface-fields-${y}.nc surface-MAY-${y}.nc
        #        nice -5 ncks -d time,${y}-01-01T00:00:00,${y}-01-31T23:00:00,2 surface-fields-${y}.nc surface-JAN-${y}.nc
        #        nice -5 ncks -d time,${y}-02-01T00:00:00,${y}-02-28T23:00:00,2 surface-fields-${y}.nc surface-FEB-${y}.nc
        #        nice -5 ncks -d time,${y}-12-01T00:00:00,${y}-12-31T23:00:00,2 surface-fields-${y}.nc surface-DEC-${y}.nc
        #        nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-09-30T23:00:00,2 surface-fields-${y}.nc surface-SEP-${y}.nc
        #        nice -5 ncks -d time,${y}-10-01T00:00:00,${y}-10-31T23:00:00,2 surface-fields-${y}.nc surface-OCT-${y}.nc
        #        nice -5 ncks -d time,${y}-11-01T00:00:00,${y}-11-30T23:00:00,2 surface-fields-${y}.nc surface-NOV-${y}.nc
#
#       #        # get time-sums for individual months
        #        nice -5 cdo timsum surface-MAR-${y}.nc surface-sum-MAR-${y}.nc
        #        nice -5 cdo timsum surface-APR-${y}.nc surface-sum-APR-${y}.nc
        #        nice -5 cdo timsum surface-MAY-${y}.nc surface-sum-MAY-${y}.nc
        #        nice -5 cdo timsum surface-JAN-${y}.nc surface-sum-JAN-${y}.nc
        #        nice -5 cdo timsum surface-FEB-${y}.nc surface-sum-FEB-${y}.nc
        #        nice -5 cdo timsum surface-DEC-${y}.nc surface-sum-DEC-${y}.nc
        #        nice -5 cdo timsum surface-SEP-${y}.nc surface-sum-SEP-${y}.nc
        #        nice -5 cdo timsum surface-OCT-${y}.nc surface-sum-OCT-${y}.nc
        #        nice -5 cdo timsum surface-NOV-${y}.nc surface-sum-NOV-${y}.nc
#
        #        cdo enssum surface-sum-MAR-${y}.nc surface-sum-APR-${y}.nc surface-sum-MAY-${y}.nc surface-sum-MAM-${y}.nc
        #        cdo enssum surface-sum-JAN-${y}.nc surface-sum-FEB-${y}.nc surface-sum-DEC-${y}.nc surface-sum-DJF-${y}.nc
        #        cdo enssum surface-sum-SEP-${y}.nc surface-sum-OCT-${y}.nc surface-sum-NOV-${y}.nc surface-sum-SON-${y}.nc
#
        #        rm surface-fields-${y}.nc surface-MAR-${y}.nc surface-APR-${y}.nc surface-MAY-${y}.nc surface-JAN-${y}.nc surface-FEB-${y}.nc surface-DEC-${y}.nc surface-SEP-${y}.nc surface-OCT-${y}.nc surface-NOV-${y}.nc
        #        mv surface-sum-MAR-${y}.nc surface-sum-APR-${y}.nc surface-sum-MAY-${y}.nc surface-sum-MAM-${y}.nc ${inp}MAM/${memb}/
        #        mv surface-sum-JAN-${y}.nc surface-sum-FEB-${y}.nc surface-sum-DEC-${y}.nc surface-sum-DJF-${y}.nc ${inp}DJF/${memb}/
        #        mv surface-sum-SEP-${y}.nc surface-sum-OCT-${y}.nc surface-sum-NOV-${y}.nc surface-sum-SON-${y}.nc ${inp}SON/${memb}/
	#done
	
	for y in {2015..2040}
	do
	        ### link file here
	        ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc b-${y}.nc
		cdo selvar,PS,PSL,SST,TS b-${y}.nc surface-fields-${y}.nc
                rm b-${y}.nc

#               ln -sf ${op}${HIST}${memb}${cpp}*h4.${y}-01-01-00000.nc b-${y}.nc
#               cdo selvar,RHREFHT,TREFHT b-${y}.nc surface2-${y}.nc
#               rm b-${y}.nc
#               cdo mergetime surface1-${y}.nc surface2-${y}.nc surface-fields-${y}.nc
#               rm surface1-${y}.nc surface2-${y}.nc
#

                nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-03-31T23:00:00,2 surface-fields-${y}.nc surface-MAR-${y}.nc
                nice -5 ncks -d time,${y}-04-01T00:00:00,${y}-04-30T23:00:00,2 surface-fields-${y}.nc surface-APR-${y}.nc
                nice -5 ncks -d time,${y}-05-01T00:00:00,${y}-05-31T23:00:00,2 surface-fields-${y}.nc surface-MAY-${y}.nc
                nice -5 ncks -d time,${y}-01-01T00:00:00,${y}-01-31T23:00:00,2 surface-fields-${y}.nc surface-JAN-${y}.nc
                nice -5 ncks -d time,${y}-02-01T00:00:00,${y}-02-28T23:00:00,2 surface-fields-${y}.nc surface-FEB-${y}.nc
                nice -5 ncks -d time,${y}-12-01T00:00:00,${y}-12-31T23:00:00,2 surface-fields-${y}.nc surface-DEC-${y}.nc
                nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-09-30T23:00:00,2 surface-fields-${y}.nc surface-SEP-${y}.nc
                nice -5 ncks -d time,${y}-10-01T00:00:00,${y}-10-31T23:00:00,2 surface-fields-${y}.nc surface-OCT-${y}.nc
                nice -5 ncks -d time,${y}-11-01T00:00:00,${y}-11-30T23:00:00,2 surface-fields-${y}.nc surface-NOV-${y}.nc
#
#               # get time-sums for individual months
                nice -5 cdo timsum surface-MAR-${y}.nc surface-sum-MAR-${y}.nc
                nice -5 cdo timsum surface-APR-${y}.nc surface-sum-APR-${y}.nc
                nice -5 cdo timsum surface-MAY-${y}.nc surface-sum-MAY-${y}.nc
                nice -5 cdo timsum surface-JAN-${y}.nc surface-sum-JAN-${y}.nc
                nice -5 cdo timsum surface-FEB-${y}.nc surface-sum-FEB-${y}.nc
                nice -5 cdo timsum surface-DEC-${y}.nc surface-sum-DEC-${y}.nc
                nice -5 cdo timsum surface-SEP-${y}.nc surface-sum-SEP-${y}.nc
                nice -5 cdo timsum surface-OCT-${y}.nc surface-sum-OCT-${y}.nc
                nice -5 cdo timsum surface-NOV-${y}.nc surface-sum-NOV-${y}.nc
#
                cdo enssum surface-sum-MAR-${y}.nc surface-sum-APR-${y}.nc surface-sum-MAY-${y}.nc surface-sum-MAM-${y}.nc
                cdo enssum surface-sum-JAN-${y}.nc surface-sum-FEB-${y}.nc surface-sum-DEC-${y}.nc surface-sum-DJF-${y}.nc
                cdo enssum surface-sum-SEP-${y}.nc surface-sum-OCT-${y}.nc surface-sum-NOV-${y}.nc surface-sum-SON-${y}.nc
#
                rm surface-fields-${y}.nc surface-MAR-${y}.nc surface-APR-${y}.nc surface-MAY-${y}.nc surface-JAN-${y}.nc surface-FEB-${y}.nc surface-DEC-${y}.nc surface-SEP-${y}.nc surface-OCT-${y}.nc surface-NOV-${y}.nc
                mv surface-sum-MAR-${y}.nc surface-sum-APR-${y}.nc surface-sum-MAY-${y}.nc surface-sum-MAM-${y}.nc ${inp}MAM/${memb}/
                mv surface-sum-JAN-${y}.nc surface-sum-FEB-${y}.nc surface-sum-DEC-${y}.nc surface-sum-DJF-${y}.nc ${inp}DJF/${memb}/
                mv surface-sum-SEP-${y}.nc surface-sum-OCT-${y}.nc surface-sum-NOV-${y}.nc surface-sum-SON-${y}.nc ${inp}SON/${memb}/	
	done
	
	for y in {2041..2070}
	do
		ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc b-${y}.nc
		cdo selvar,PS,PSL,SST,TS b-${y}.nc surface-fields-${y}.nc
                rm b-${y}.nc

#               ln -sf ${op}${HIST}${memb}${cpp}*h4.${y}-01-01-00000.nc b-${y}.nc
#               cdo selvar,RHREFHT,TREFHT b-${y}.nc surface2-${y}.nc
#               rm b-${y}.nc
#               cdo mergetime surface1-${y}.nc surface2-${y}.nc surface-fields-${y}.nc
#               rm surface1-${y}.nc surface2-${y}.nc
#

                nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-03-31T23:00:00,2 surface-fields-${y}.nc surface-MAR-${y}.nc
                nice -5 ncks -d time,${y}-04-01T00:00:00,${y}-04-30T23:00:00,2 surface-fields-${y}.nc surface-APR-${y}.nc
                nice -5 ncks -d time,${y}-05-01T00:00:00,${y}-05-31T23:00:00,2 surface-fields-${y}.nc surface-MAY-${y}.nc
                nice -5 ncks -d time,${y}-01-01T00:00:00,${y}-01-31T23:00:00,2 surface-fields-${y}.nc surface-JAN-${y}.nc
                nice -5 ncks -d time,${y}-02-01T00:00:00,${y}-02-28T23:00:00,2 surface-fields-${y}.nc surface-FEB-${y}.nc
                nice -5 ncks -d time,${y}-12-01T00:00:00,${y}-12-31T23:00:00,2 surface-fields-${y}.nc surface-DEC-${y}.nc
                nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-09-30T23:00:00,2 surface-fields-${y}.nc surface-SEP-${y}.nc
                nice -5 ncks -d time,${y}-10-01T00:00:00,${y}-10-31T23:00:00,2 surface-fields-${y}.nc surface-OCT-${y}.nc
                nice -5 ncks -d time,${y}-11-01T00:00:00,${y}-11-30T23:00:00,2 surface-fields-${y}.nc surface-NOV-${y}.nc
#
#               # get time-sums for individual months
                nice -5 cdo timsum surface-MAR-${y}.nc surface-sum-MAR-${y}.nc
                nice -5 cdo timsum surface-APR-${y}.nc surface-sum-APR-${y}.nc
                nice -5 cdo timsum surface-MAY-${y}.nc surface-sum-MAY-${y}.nc
                nice -5 cdo timsum surface-JAN-${y}.nc surface-sum-JAN-${y}.nc
                nice -5 cdo timsum surface-FEB-${y}.nc surface-sum-FEB-${y}.nc
                nice -5 cdo timsum surface-DEC-${y}.nc surface-sum-DEC-${y}.nc
                nice -5 cdo timsum surface-SEP-${y}.nc surface-sum-SEP-${y}.nc
                nice -5 cdo timsum surface-OCT-${y}.nc surface-sum-OCT-${y}.nc
                nice -5 cdo timsum surface-NOV-${y}.nc surface-sum-NOV-${y}.nc
#
                cdo enssum surface-sum-MAR-${y}.nc surface-sum-APR-${y}.nc surface-sum-MAY-${y}.nc surface-sum-MAM-${y}.nc
                cdo enssum surface-sum-JAN-${y}.nc surface-sum-FEB-${y}.nc surface-sum-DEC-${y}.nc surface-sum-DJF-${y}.nc
                cdo enssum surface-sum-SEP-${y}.nc surface-sum-OCT-${y}.nc surface-sum-NOV-${y}.nc surface-sum-SON-${y}.nc
#
                rm surface-fields-${y}.nc surface-MAR-${y}.nc surface-APR-${y}.nc surface-MAY-${y}.nc surface-JAN-${y}.nc surface-FEB-${y}.nc surface-DEC-${y}.nc surface-SEP-${y}.nc surface-OCT-${y}.nc surface-NOV-${y}.nc
                mv surface-sum-MAR-${y}.nc surface-sum-APR-${y}.nc surface-sum-MAY-${y}.nc surface-sum-MAM-${y}.nc ${inp}MAM/${memb}/
                mv surface-sum-JAN-${y}.nc surface-sum-FEB-${y}.nc surface-sum-DEC-${y}.nc surface-sum-DJF-${y}.nc ${inp}DJF/${memb}/
                mv surface-sum-SEP-${y}.nc surface-sum-OCT-${y}.nc surface-sum-NOV-${y}.nc surface-sum-SON-${y}.nc ${inp}SON/${memb}/
	done
	
	for y in {2071..2100}
	do
		ln -sf ${op}${PROJ}${memb}${cpp}*h3.${y}-01-01-00000.nc b-${y}.nc
		cdo selvar,PS,PSL,SST,TS b-${y}.nc surface-fields-${y}.nc
                rm b-${y}.nc

#               ln -sf ${op}${HIST}${memb}${cpp}*h4.${y}-01-01-00000.nc b-${y}.nc
#               cdo selvar,RHREFHT,TREFHT b-${y}.nc surface2-${y}.nc
#               rm b-${y}.nc
#               cdo mergetime surface1-${y}.nc surface2-${y}.nc surface-fields-${y}.nc
#               rm surface1-${y}.nc surface2-${y}.nc
#

                nice -5 ncks -d time,${y}-03-01T00:00:00,${y}-03-31T23:00:00,2 surface-fields-${y}.nc surface-MAR-${y}.nc
                nice -5 ncks -d time,${y}-04-01T00:00:00,${y}-04-30T23:00:00,2 surface-fields-${y}.nc surface-APR-${y}.nc
                nice -5 ncks -d time,${y}-05-01T00:00:00,${y}-05-31T23:00:00,2 surface-fields-${y}.nc surface-MAY-${y}.nc
                nice -5 ncks -d time,${y}-01-01T00:00:00,${y}-01-31T23:00:00,2 surface-fields-${y}.nc surface-JAN-${y}.nc
                nice -5 ncks -d time,${y}-02-01T00:00:00,${y}-02-28T23:00:00,2 surface-fields-${y}.nc surface-FEB-${y}.nc
                nice -5 ncks -d time,${y}-12-01T00:00:00,${y}-12-31T23:00:00,2 surface-fields-${y}.nc surface-DEC-${y}.nc
                nice -5 ncks -d time,${y}-09-01T00:00:00,${y}-09-30T23:00:00,2 surface-fields-${y}.nc surface-SEP-${y}.nc
                nice -5 ncks -d time,${y}-10-01T00:00:00,${y}-10-31T23:00:00,2 surface-fields-${y}.nc surface-OCT-${y}.nc
                nice -5 ncks -d time,${y}-11-01T00:00:00,${y}-11-30T23:00:00,2 surface-fields-${y}.nc surface-NOV-${y}.nc
#
#               # get time-sums for individual months
                nice -5 cdo timsum surface-MAR-${y}.nc surface-sum-MAR-${y}.nc
                nice -5 cdo timsum surface-APR-${y}.nc surface-sum-APR-${y}.nc
                nice -5 cdo timsum surface-MAY-${y}.nc surface-sum-MAY-${y}.nc
                nice -5 cdo timsum surface-JAN-${y}.nc surface-sum-JAN-${y}.nc
                nice -5 cdo timsum surface-FEB-${y}.nc surface-sum-FEB-${y}.nc
                nice -5 cdo timsum surface-DEC-${y}.nc surface-sum-DEC-${y}.nc
                nice -5 cdo timsum surface-SEP-${y}.nc surface-sum-SEP-${y}.nc
                nice -5 cdo timsum surface-OCT-${y}.nc surface-sum-OCT-${y}.nc
                nice -5 cdo timsum surface-NOV-${y}.nc surface-sum-NOV-${y}.nc
#
                cdo enssum surface-sum-MAR-${y}.nc surface-sum-APR-${y}.nc surface-sum-MAY-${y}.nc surface-sum-MAM-${y}.nc
                cdo enssum surface-sum-JAN-${y}.nc surface-sum-FEB-${y}.nc surface-sum-DEC-${y}.nc surface-sum-DJF-${y}.nc
                cdo enssum surface-sum-SEP-${y}.nc surface-sum-OCT-${y}.nc surface-sum-NOV-${y}.nc surface-sum-SON-${y}.nc
#
                rm surface-fields-${y}.nc surface-MAR-${y}.nc surface-APR-${y}.nc surface-MAY-${y}.nc surface-JAN-${y}.nc surface-FEB-${y}.nc surface-DEC-${y}.nc surface-SEP-${y}.nc surface-OCT-${y}.nc surface-NOV-${y}.nc
                mv surface-sum-MAR-${y}.nc surface-sum-APR-${y}.nc surface-sum-MAY-${y}.nc surface-sum-MAM-${y}.nc ${inp}MAM/${memb}/
                mv surface-sum-JAN-${y}.nc surface-sum-FEB-${y}.nc surface-sum-DEC-${y}.nc surface-sum-DJF-${y}.nc ${inp}DJF/${memb}/
                mv surface-sum-SEP-${y}.nc surface-sum-OCT-${y}.nc surface-sum-NOV-${y}.nc surface-sum-SON-${y}.nc ${inp}SON/${memb}/
	done
	
done
exit 0

