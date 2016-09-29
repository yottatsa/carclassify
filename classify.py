#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Francois Boulogne

import os
import os.path
import sys
import glob

import numpy as np
import skimage.io
from skimage.draw import line
from sklearn.externals import joblib


def load_knowndata(filenames, shape):
    unknown = {'targets': [], 'data': [], 'name': []}

    for index, filename in enumerate(filenames):
        target = os.path.splitext(os.path.basename(filename))[0]
        target = int(target.split('-')[0])
        image = skimage.io.imread(filename)
        if image.shape != shape:
            continue
        unknown['targets'].append(target)
        unknown['name'].append(filename)
        unknown['data'].append(image.flatten().tolist())
    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    unknown['targets'] = np.array(unknown['targets'])
    unknown['data'] = np.array(unknown['data'])
    return unknown


def train(filenames, shape):
    from sklearn import svm
    #from sklearn import linear_model
    unknown = load_knowndata(filenames, shape)
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=1e-8)
    # classifier = linear_model.LogisticRegression()
    # We learn the digits on the first half of the digits
    classifier.fit(unknown['data'], unknown['targets'])
    return classifier


def mark_region(image, x, y, h, w, mark=None):
    if not mark:
        return
    image[line(x, y, x, y + w)] = mark
    image[line(x, y + w, x + h, y + w)] = mark
    image[line(x + h, y + w, x + h, y)] = mark
    image[line(x + h, y, x, y)] = mark


if __name__ == '__main__':
    shape = dx, dy = 40, 50

    if len(sys.argv) == 2:
        model = sys.argv[1]
    else:
        model = 'model.pkl'

    if os.path.exists(model):
        classifier = joblib.load('filename.pkl')
    else:
        filenames = sorted(glob.glob('train/*.jpg'))
        classifier = train(filenames, shape)
        joblib.dump(classifier, model)

    filenames = sorted(glob.glob('test/*.jpg'))
    for name in filenames:
        unknown = {'targets': [], 'data': [], 'name': []}
        result = os.path.join('result', os.path.basename(name))
        image = skimage.io.imread(name)
        h, w = image.shape
        for x in range(0, h - dx, dx / 3):
            for y in range(0, w - dy, dy / 3):
                probe = image[x:x + dx, y:y + dy]
                if probe.shape != (dx, dy):
                    continue
                name = (x, y, dx, dy)
                unknown['targets'].append(-1)  # Target = -1: unkown
                unknown['name'].append(name)
                unknown['data'].append(probe.flatten().tolist())

        unknown['targets'] = np.array(unknown['targets'])
        unknown['data'] = np.array(unknown['data'])

        predicted = classifier.predict(unknown['data'])

        cx, cy = 0, 0
        for pred, name in zip(predicted, unknown['name']):
            x, y, dx, dy = name

            if cx <= x <= cx + dx and cy <= y <= cy + dy:
                continue

            if pred == 0:
                cx, cy = x, y

            mark_region(image, x + 5, y + 5, dx - 10, dy - 10, mark=[200, 50, None][pred])

        skimage.io.imsave(result, image)
