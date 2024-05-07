#!/bin/bash
# these are 48 hours from mature stage
t=-48
for k in $(ls -rt trastart-mature-* | head -206 | tail -53)
do
	start=$(ls $k | cut -b 17-27)
	name=$(ls $k | cut -b 17-37)
        nd="trajectories-mature-$name.txt"
        nohup caltra $k $t $nd -i 60 -o 60 -ts 5 -ref $start -cdf /home/ascherrmann/009-ERA-5/MED/data/ &
	sleep 230s
done
