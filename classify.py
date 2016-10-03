#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
except:
    plt = None

import sys
import glob
import os.path

import numpy as np
import skimage.io
from skimage import exposure, img_as_float, img_as_ubyte
from skimage.draw import line
from skimage.color import rgb2gray
from skimage.feature import canny, hog
from skimage.transform import rescale, ProjectiveTransform, warp
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

from scipy import ndimage

import datetime


def adjust(img, **kwargs):
    img = img_as_float(img)
    #p2, p98 = np.percentile(img, (2, 98))
    #img = exposure.rescale_intensity(img, in_range=(p2, p98))

    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img


def extract(img):
    features = []
    features = features + rescale(img, 0.5).flatten().tolist()
    features = features + rescale(canny(img, sigma=2), 0.5).flatten().tolist()
    return features


def load_knowndata(filenames, shape):
    dx, dy = shape
    X = []
    y = []
    for filename in filenames:
        image = skimage.io.imread(filename)
        for point in open(filename + '.txt').read().split():
            iy, ix, target = map(int, point.split('x'))
            probe = image[ix - dx / 2:ix + dx / 2, iy - dy / 2:iy + dy / 2]
            if probe.shape == shape:
                print filename, ix, iy, target
                X.append(extract(probe))
                y.append(target)
    return np.array(X), np.array(y)


def train(filenames, shape):
    Cs = np.logspace(-12, 3, 10)
    gammas = np.logspace(-8, -1, 8)
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

    if classifier:
        h, w = image.shape
        for x in range(0, h-dx+1, dx/10):
            for y in range(0, w-dy+1, dy/3):
                probe = image[x:x + dx, y:y + dy]
                if probe.shape == shape:
                    coords.append((x, y, dx, dy))
                    X.append(extract(probe))

        predicted = classifier.predict(np.array(X))
    else:
        predicted = []
    return coords, predicted


def mark_region(image, x, y, h, w, mark=None):
    if not mark:
        return
    image[line(x, y, x + h, y + w)] = mark
    image[line(x + h, y, x, y + w)] = mark


def download_series(url, num, filter_func, **kwargs):
    images = [
        filter_func(skimage.io.imread(url),
                    **kwargs)
        for x in range(num - 1)]
    series = skimage.io.concatenate_images(images)
    return series


def open_series(pattern, filter_func=lambda a: a, **kwargs):
    def load_func(f, *args, **kwargs):
        return filter_func(skimage.io.imread(f), **kwargs)

    ic = skimage.io.ImageCollection(pattern, load_func=load_func, **kwargs)
    series = ic.concatenate()
    return series


def combine(imgs):
    tf = ProjectiveTransform()
    src = np.array((
        (0, 0),
        (0, 110),
        (525, 110),
        (525, 0)
    ))
    dst = np.array((
        (169, 169),
        (177, 280),
        (604, 203),
        (594, 129)
    ))
    of = (110, 525)
    tf.estimate(src, dst)

    img = np.median(imgs, axis=0)
    img = ndimage.median_filter(img, 2)
    img = warp(img, tf, output_shape=of)
    img = rgb2gray(img)

    return img


if __name__ == '__main__':
    shape = dx, dy = 40, 50
    model = 'model.pkl'
    if os.path.exists(model):
        classifier = joblib.load(model)
        print model
    else:
        filenames = sorted(glob.glob('train/*.jpg'))
        try:
            classifier = train(filenames, shape)
            joblib.dump(classifier, model)
        except Exception, e:
            print e
            classifier = None
        print model

    try:
        url = sys.argv[1]
        img = download_series(url, 16, adjust)
    except  IndexError:
        img = open_series('test/*.jpg', adjust)
    img = combine(img)

    coords, predicted = walk_image(img, shape, classifier)
    img = img_as_ubyte(img)
    img_to_show = np.copy(img)

    marks = []
    for coord, pred in zip(coords, predicted):
        if pred > 128:
            x, y, dx, dy = coord
            mark_region(img_to_show, x + 15, y + 15, dx - 30, dy - 30, pred)

    try:
        out = sys.argv[2]
    except  IndexError:
        out = None

    if out:
        skimage.io.imsave(out, img_to_show)
        exit(0)

    name = os.path.join("train", datetime.datetime.now().isoformat())
    log = None


    def onclick(event):
        global log, name
        if event and event.xdata:
            if not log:
                log = open('{}.jpg.txt'.format(name), 'a+')
                log.write('\n\n')
            log.write('{}x{}x{}\n'.format(int(event.xdata), int(event.ydata),
                                          int(255/event.button)))
            print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  (event.button, event.x, event.y, event.xdata, event.ydata))


    fig = plt.figure()
    ax = fig.add_subplot()

    plt.imshow(img_to_show, cmap='Greys_r')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    if log:
        skimage.io.imsave('{}.jpg'.format(name), img)
