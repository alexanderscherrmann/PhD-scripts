#!/bin/bash 

for k in $(ls wrfout*)
do
	y=$(ls $k |cut -c 12-15)
	m=$(ls $k | cut -c 17-18)
	d=$(ls $k | cut -c 20-21)
	h=$(ls $k | cut -c 23-24)
	ln -sf $k "P${y}${m}${d}_${h}"
done
