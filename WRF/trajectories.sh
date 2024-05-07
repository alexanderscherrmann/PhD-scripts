#!/bin/bash
# these are 96 hours from mature stage
idt=$(pwd)
td=/atmosdyn2/ascherrmann/013-WRF-sim/data/PV-med-traj/
cd $td
hd=$(pwd)
t=-192
for k in trajectories*.txt
do
	start=$(ls $k | cut -c 14-24)
	time caltra $k $t $k -i 60 -o 60 -ts 5 -ref $start -cdf /atmosdyn2/ascherrmann/009-ERA-5/MED/data/	
done
