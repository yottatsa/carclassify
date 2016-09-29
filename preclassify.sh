#!/bin/sh -x
car="$1/car"
notcar="$1/notcar"
twocars="$1/twocars"
mkdir -p "$car" "$notcar" "$twocars"
shift
for f
do
	b="$(basename "$f" .jpg)"
	for x in 0 45 98 147 202 284 359 399 465 519 574
	do
		convert "$f" -crop 50x40+${x}+26 -colorspace Gray "$car/${b}-${x}-1.jpg"
		convert "$f" -crop 50x40+${x}+39 -colorspace Gray "$car/${b}-${x}-2.jpg"
		convert "$f" -crop 50x40+${x}+10 -colorspace Gray "$notcar/${b}-${x}.jpg"
		convert "$f" -crop 50x40+$((x+15))+26 -colorspace Gray "$twocars/${b}-$((x+15)).jpg"
		convert "$f" -crop 50x40+$((x+35))+39 -colorspace Gray "$twocars/${b}-$((x+35)).jpg"
	done
done
