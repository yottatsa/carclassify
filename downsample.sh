#!/bin/sh -x
d="$1"
mkdir -p "$d"
shift
for f
do
	convert "$f" -distort Perspective "90,165,0,165 90,270,0,270 580,130,640,165 580,190,640,270" -crop 640x80+0+210 -colorspace Gray  "$d/$(basename "${f}")"
done
