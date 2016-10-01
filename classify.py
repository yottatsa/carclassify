#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import os.path
import sys

import numpy as np
import skimage.io
from skimage import exposure, img_as_float, img_as_ubyte
from skimage.draw import line
from skimage.transform import rescale
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


def adjust(img):
    img = img_as_float(img)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_as_ubyte(img)


def extract(img):
    feature = rescale(img, 0.5)
    feature = feature.flatten()
    return feature.tolist()


def load_knowndata(filenames, shape):
    X = []
    y = []
    for index, filename in enumerate(filenames):
        target = os.path.splitext(os.path.basename(filename))[0]
        target = int(target.split('-')[0])
        image = skimage.io.imread(filename)
        if image.shape == shape:
            print filename, target
            X.append(extract(adjust(image)))
            y.append(target)
    return np.array(X), np.array(y)


def train(filenames, shape):
    Cs = np.logspace(-1, 3, 10)
    gammas = np.logspace(-5, -1, 8)
    svc = svm.SVC(decision_function_shape='ovo')
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs, gamma=gammas),
                       n_jobs=-1)
    X, y = load_knowndata(filenames, shape)
    clf.fit(X, y)
    print clf.best_score_
    print clf.best_estimator_.C
    print clf.best_estimator_.gamma
    return clf.best_estimator_


def walk_image(image, shape, classifier):
    dx, dy = shape
    coords = []
    X = []
    h, w = image.shape
    for x in range(0, h, dx / 2):
        for y in range(0, w, dy / 4):
            probe = image[x:x + dx, y:y + dy]
            if probe.shape == shape:
                coords.append((x, y, dx, dy))
                X.append(extract(probe))

    predicted = classifier.predict(np.array(X))
    return coords, predicted


def mark_region(image, x, y, h, w, mark=None):
    if not mark:
        return
    image[line(x, y, x + h, y + w)] = mark
    image[line(x + h, y, x, y + w)] = mark


def predict(filenames, shape, classifier):
    for name in filenames:

        result = os.path.join('result', os.path.basename(name))
        image = skimage.io.imread(name)

        image = adjust(image)
        coords, predicted = walk_image(image, shape, classifier)

        marks = []
        maxmark = 0
        for coord, pred in zip(coords, predicted):
            maxmark = max(maxmark, pred)
            if pred > 64:
                x, y, dx, dy = coord
                mark_region(image, x + 10, y + 10, dx - 20, dy - 20, pred)

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
