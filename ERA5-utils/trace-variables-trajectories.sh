#!/bin/bash/
for k in trajectories-mature-2full*.txt
do
        name=$(ls $k | cut -b 21-46)
        traced="traced-vars-S-$name.txt"
	nohup trace $k $traced &
	sleep 60s
done
