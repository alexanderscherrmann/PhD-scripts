#!/bin/bash
p=$(pwd)
for k in 1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2012 2013 2014 2015 2016 2017 2018 2019 2020
do
	for q in 01 02 12
	do
#		if [[ "$k" = "2001" ]] 
#		then
#			if [[ "$q" = "12" ]]
#			then
#				continue
#			fi
#
#		fi	
#		if [[ "$k" = "2018" ]]
#                then
#                        if [[ "$q" = "12" ]]
#                        then
#                                continue
#                        fi
#
#                fi		
		nohup python get-single-data-files-sf.py 80 -150 80 -20 $k $q
	done
	sleep 1200	
done

