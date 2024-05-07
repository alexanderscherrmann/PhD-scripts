#!bin/bash

export LAGRANTO=/usr/local/dyn_tools/lagranto.wrf/
export MODEL=wrf

dwrf=/atmosdyn2/ascherrmann/013-WRF-sim/

t=04_12
lon1=-10
lon2=15
lat1=25
lat2=48

#for sim in DJF-clim-max-U-at-300-hPa-2.1-QGPV DJF-clim-max-U-at-300-hPa-0.7-QGPV DJF-clim-max-U-at-300-hPa-0.9-QGPV DJF-clim-max-U-at-300-hPa-1.1-QGPV DJF-clim-max-U-at-300-hPa-1.4-QGPV DJF-clim-max-U-at-300-hPa-1.7-QGPV DJF-clim-max-U-at-300-hPa-2.8-QGPV DJF-200-km-east-from-max-300-hPa-0.7-QGPV   DJF-200-km-south-from-max-300-hPa-0.7-QGPV DJF-200-km-east-from-max-300-hPa-1.4-QGPV   DJF-200-km-south-from-max-300-hPa-1.4-QGPV DJF-200-km-east-from-max-300-hPa-2.1-QGPV   DJF-200-km-south-from-max-300-hPa-2.1-QGPV DJF-200-km-north-from-max-300-hPa-0.7-QGPV  DJF-200-km-west-from-max-300-hPa-0.7-QGPV DJF-200-km-north-from-max-300-hPa-1.4-QGPV  DJF-200-km-west-from-max-300-hPa-1.4-QGPV DJF-200-km-north-from-max-300-hPa-2.1-QGPV  DJF-200-km-west-from-max-300-hPa-2.1-QGPV DJF-400-km-east-from-max-300-hPa-0.7-QGPV   DJF-400-km-south-from-max-300-hPa-0.7-QGPV DJF-400-km-east-from-max-300-hPa-1.4-QGPV   DJF-400-km-south-from-max-300-hPa-1.4-QGPV DJF-400-km-east-from-max-300-hPa-2.1-QGPV   DJF-400-km-south-from-max-300-hPa-2.1-QGPV DJF-400-km-north-from-max-300-hPa-0.7-QGPV  DJF-400-km-west-from-max-300-hPa-0.7-QGPV DJF-400-km-north-from-max-300-hPa-1.4-QGPV  DJF-400-km-west-from-max-300-hPa-1.4-QGPV DJF-400-km-north-from-max-300-hPa-2.1-QGPV  DJF-400-km-west-from-max-300-hPa-2.1-QGPV
#for sim in DJF-clim-max-U-at-300-hPa-hourly-2.1-QGPV
for sim in DJF-clim-max-U-at-300-hPa-0.3-QGPV-check  DJF-clim-max-U-at-300-hPa-0.9-QGPV-check  DJF-clim-max-U-at-300-hPa-1.7-QGPV-check DJF-clim-max-U-at-300-hPa-0.5-QGPV-check  DJF-clim-max-U-at-300-hPa-1.1-QGPV-check  DJF-clim-max-U-at-300-hPa-2.1-QGPV-check DJF-clim-max-U-at-300-hPa-0.7-QGPV-check  DJF-clim-max-U-at-300-hPa-1.4-QGPV-check
do
	# create start files in lat lon
	python /home/ascherrmann/scripts/WRF/generate-wrf-trajectories-start-file.py ${sim} ${t} ${lon1} ${lon2} ${lat1} ${lat2}
	# chage to simulation dir
	cd ${dwrf}${sim}
#	cp /atmosdyn2/ascherrmann/013-WRF-sim/DJF-clim-max-U-at-300-hPa-0.7-QGPV/tracevars .

	#create coordinate conversion map
	/usr/local/dyn_tools/lagranto.wrf/goodies/wrfmap.sh -create wrfout_d01_2000-12-01_00:00:00

	# convert start points to xy coordinates
	/usr/local/dyn_tools/lagranto.wrf/goodies/wrfmap.sh -ll2xy startf.ll startf.xy

	# add PV to the wrfoutput
	python /home/ascherrmann/scripts/WRF/add-PV-to-wrf-out.py ${sim}

	# add variables to P files
	bash /home/ascherrmann/scripts/WRF/link-wrfout-to-Pfiles.sh

	# calc traj
	/usr/local/dyn_tools/lagranto.wrf/bin/caltra.sh 200012${t} 20001201_00 startf.xy trajectories.xy

	# trace PV along trajectories
	trace trajectories.xy trace.xy

	/usr/local/dyn_tools/lagranto.wrf/goodies/wrfmap.sh -xy2ll trace.xy trace.ll
	python /home/ascherrmann/scripts/Paper2/plot-trajectories-wrf-sim.py ${sim}
done

