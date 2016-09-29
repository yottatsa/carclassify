#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path
import sys
import glob

import numpy as np
import math
import skimage.io
from skimage.draw import line
from sklearn.externals import joblib
from skimage.feature import hog


from sklearn import svm
from skimage import exposure, img_as_float

def adjust(img):
    img = exposure.adjust_gamma(img, 2)
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))

def extract(img):
    feature = adjust(img).flatten()
    return feature.tolist()

def vector():
    vec = []

    def append(*args):
        vec.append(args)
        return vec
    return append, vec

def load_knowndata(filenames, shape):
    vec, res = vector()
    for index, filename in enumerate(filenames):
        target = os.path.splitext(os.path.basename(filename))[0]
        target = int(target.split('-')[0])
        image = skimage.io.imread(filename)
        if image.shape == shape:
            print filename, target
            res = vec(filename, target, extract(image))
    names, targets, datas = zip(*res)
    return names, np.array(targets), np.array(datas)

def train(filenames, shape):
    _, targets, datas = load_knowndata(filenames, shape)
    classifier = svm.SVC(gamma=1e-8, decision_function_shape='ovo')
    classifier.fit(datas, targets)
    return classifier

def walk_image(image, shape, classifier):
    dx, dy = shape
    vec, res = vector()
    h, w = image.shape
    for x in range(0, h, dx/10):
        for y in range(0, w, dy/10):
            probe = image[x:x + dx, y:y + dy]
            if probe.shape == shape:
                res = vec((x, y, dx, dy), probe, extract(probe))

    names, probes, datas = zip(*res)
    predicted = classifier.predict(np.array(datas))
    return names, predicted, probes

def mark_region(image, x, y, h, w, mark=None):
    if not mark:
        return
    image[line(x, y, x + h, y + w)] = mark
    image[line(x + h, y, x, y + w)] = mark

def deduplicator():
    hits = []
    def func(x, y, r):
        for lx, ly, lr in hits:
            distance = math.sqrt((lx-x) * (lx-x) + (ly-y)*(ly-y))
            if distance < r+lr:
                return False
        hits.append((x, y, r))	
        return True
    return func

def predict(filenames, shape, classifier):
    for name in filenames:
        show = deduplicator()

        result = os.path.join('result', os.path.basename(name))
        image = skimage.io.imread(name)

        coords, predicted, probes = walk_image(image, shape, classifier)
        image = adjust(image)

        for pred, coords, probe in zip(predicted, coords, probes):
            if pred > 16:
                x, y, dx, dy = coords
                r = min(dx, dy)/3
                if show(x+dx/2, y+dy/2, r):
                    mark_region(image, x+5, y+5, dx-10, dy-10, mark=pred)
                    #pr = os.path.join(
                    #    'probe', 
                    #    '{}-{}x{}-{}'.format(pred, x, y, os.path.basename(name)))
                    #skimage.io.imsave(pr, probe)

        skimage.io.imsave(result, image)
        print result


if __name__ == '__main__':
    shape = dx, dy = 40, 50

    if len(sys.argv) == 2:
        model = sys.argv[1]
    else:
        model = 'model.pkl'

    if os.path.exists(model):
        classifier = joblib.load(model)
        print model
    else:
        filenames = sorted(glob.glob('train/*.jpg'))
        classifier = train(filenames, shape)
        joblib.dump(classifier, model)
        print model

    filenames = sorted(glob.glob('test/*.jpg'))
    predict(filenames, shape, classifier)
