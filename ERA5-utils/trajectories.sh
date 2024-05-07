#!/bin/bash
# these are 48 hours from mature stage
a=0
end=($(ls traend-*.txt | cut -b 8-18))
for k in trastart-mature-2full*.txt
do
	start=$(ls $k | cut -b 22-32)
	name=$(ls $k | cut -b 17-42)
        nd="trajectories-mature-$name.txt"
	en=${end[$a]}
	let "a+=1"
        nohup caltra $start $en $k $nd &
	sleep 120s
done

