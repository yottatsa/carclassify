Find a car
==========

    find ~/img/ -type f | xargs ./downsample.sh downsample
    find downsample/ -type f | xargs ./preclassify.sh preclassify
    python classify.py
    
