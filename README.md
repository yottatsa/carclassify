Find a car
==========

![demo](https://raw.githubusercontent.com/yottatsa/carclassify/master/demo.jpg)

    find ~/img/ -type f | xargs ./downsample.sh downsample
    find downsample/ -type f | xargs ./preclassify.sh preclassify
    python classify.py
    
