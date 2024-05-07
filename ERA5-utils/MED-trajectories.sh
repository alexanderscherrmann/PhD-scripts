#!/bin/bash
# these are 48 hours from mature stage
a=0
t=-48
for k in trastart-mature*.txt
do
	start=$(ls $k | cut -b 17-27)
	year=$(ls $k | cut -b 17-20)
	mon=$(ls $k | cut -b 21-22)
	name=$(ls $k | cut -b 17-37)
        nd="trajectories-mature-$name.txt"
	en=${end[$a]}
	let "a+=1"
        nohup caltra $k $t $nd -i 60 -o 60 -ts 5 -ref $start -cdf /atmosdyn/era5/cdf/$year/$mon/ &
	sleep 240s
done
