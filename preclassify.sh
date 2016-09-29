#!/bin/sh -x
d="$1"
mkdir -p "$d"
shift
for f
do
	b="$(basename "$f" .jpg)"
	for x in 0 37 86 153 213 265 318 380 428 479 519 579
	do
		convert "$f" -crop 50x40+${x}+36 -colorspace Gray "$d/0-${b}-${x}.jpg"
		convert "$f" -crop 50x40+$((x+10))+10 -colorspace Gray "$d/2-${b}-$((x+10)).jpg"
		convert "$f" -crop 50x40+$((x+20))+0 -colorspace Gray "$d/2-${b}-$((x)).jpg"
		convert "$f" -crop 50x40+$((x+25))+36 -colorspace Gray "$d/1-${b}-$((x+25)).jpg"
	done
done
