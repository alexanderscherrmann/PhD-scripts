#!bin/bash

export LAGRANTO=/usr/local/dyn_tools/lagranto.wrf/
export MODEL=wrf
export NETCDF_FORMAT=CF
dwrf=/atmosdyn2/ascherrmann/013-WRF-sim/

t="01_00"
lon1=-75
lon2=-50
lat1=35
lat2=55
for sim in DJF-clim-max-U-at-300-hPa-0.7-QGPV DJF-clim-max-U-at-300-hPa-1.4-QGPV DJF-clim-max-U-at-300-hPa-2.1-QGPV
       #	DJF-clim-max-U-at-300-hPa-1.4-QGPV
do
	# create start files in lat lon
	python /home/ascherrmann/scripts/Paper2/generate-WCB-trajectories.py ${sim} ${t} ${lon1} ${lon2} ${lat1} ${lat2}
	# chage to simulation dir
	#
	cd ${dwrf}${sim}
	cp /atmosdyn2/ascherrmann/013-WRF-sim/DJF-clim-max-U-at-300-hPa-1.4-QGPV/tracevars .

	#create coordinate conversion map
	/usr/local/dyn_tools/lagranto.wrf/goodies/wrfmap.sh -create wrfout_d01_2000-12-01_00:00:00
#
#	# convert start points to xy coordinates
	/usr/local/dyn_tools/lagranto.wrf/goodies/wrfmap.sh -ll2xy wcb_start.ll wcb_startf.xy
#
#	# add PV to the wrfoutput
	python /home/ascherrmann/scripts/WRF/add-PV-to-wrf-out.py ${sim}
#
#	# add variables to P files
	bash /home/ascherrmann/scripts/WRF/link-wrfout-to-Pfiles.sh
#
#	# calc traj
	/usr/local/dyn_tools/lagranto.wrf/bin/caltra.sh 200012${t} 20001207_12 wcb_startf.xy wcb_trajectories.xy
#
#	# trace PV along trajectories
	trace wcb_trajectories.xy wcb_trace.xy
#
	/usr/local/dyn_tools/lagranto.wrf/goodies/wrfmap.sh -xy2ll wcb_trace.xy wcb_trace.ll
#	python /home/ascherrmann/scripts/Paper2/plot-wcb-trajectories.py ${sim}
#
	python /home/ascherrmann/scripts/Paper2/wcb-in-evolution.py $sim


done

