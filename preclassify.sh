#!/bin/sh -x
d="$1"
mkdir -p "$d"
shift
for f
do
	b="$(basename "$f" .jpg)"
	for x in 225x38
	do
		convert "$f" -crop 50x40+${x} -colorspace Gray "$d/130-${b}-${x}.jpg"
	done
done
