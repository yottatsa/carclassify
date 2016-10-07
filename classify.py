#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
except:
    plt = None

import datetime
import glob
import logging
import os.path
import sys

import numpy as np
import skimage.io
from influxdb import InfluxDBClient
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray
from skimage.draw import line
from skimage.exposure import equalize_adapthist
from skimage.feature import canny
from skimage.morphology import label
from skimage.restoration import denoise_bilateral
from skimage.transform import rescale, ProjectiveTransform, warp
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


def adjust(img, tf, of, **kwargs):
    img = img_as_float(img)
    img = equalize_adapthist(img, clip_limit=0.03)
    img = warp(img, tf, output_shape=of)
    return img


def combine(imgs):
    img = np.mean(imgs, axis=0)
    img = denoise_bilateral(img, multichannel=True)
    img = rgb2gray(img)
    img = img_as_ubyte(img)
    return img


def extract(img):
    features = []
    features = features + rescale(canny(rescale(img, 2), sigma=3), 0.1).flatten().tolist()
    features = features + rescale(canny(img, sigma=1), 0.5).flatten().tolist()
    features = features + rescale(img, 0.5).flatten().tolist()
    return features


def get_probe(shape, image, x, y):
    """
    Make crop and feature extract
    :param shape: Probe shape
    :param image: Source image
    :param x: X coord of probe
    :param y: Y coord of probe
    :return: adjusted coords and feature vector of probe
    """
    h, w = image.shape
    dx, dy = shape
    x = x - max(min(x, 0), x + dx - h)
    y = y - max(min(y, 0), y + dy - w)
    return x, y, extract(image[x:x + dx, y:y + dy])


def load_knowndata(filenames, shape):
    """
    Load train data and return samples
    :param filenames: Files to read
    :param shape: Probe shape
    :return: X,y
    """
    dx, dy = shape
    X = []
    y = []
    for filename in filenames:
        image = skimage.io.imread(filename)
        for point in open(filename + '.txt').read().split():
            if point.startswith('#'):
                continue
            ty, tx, target = map(int, point.split('x'))
            tx, ty, probe = get_probe(shape, image, tx - dx / 2, ty - dy / 2)
            logging.debug('Datapoint: %s %d %d %d', filename, tx, ty, target)
            X.append(probe)
            y.append(target)
    return np.array(X), np.array(y)


def train(filenames, shape):
    Cs = np.logspace(-1, 5, 12)
    gammas = np.logspace(-8, -1, 7)
    X, y = load_knowndata(filenames, shape)
    svc = svm.SVC(decision_function_shape='ovo')
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs, gamma=gammas),
                       cv=10, n_jobs=-1)
    clf.fit(X, y)
    logging.info('Model: %f C=%f', clf.best_score_, clf.best_estimator_.C)
    return clf.best_estimator_


def walk_image(image, shape, classifier):
    dx, dy = shape
    coords = []
    X = []

    if classifier:
        h, w = image.shape
        for x in range(0, h - dx + 1, dx / 4):
            for y in range(0, w - dy + 1, dy / 4):
                x, y, probe = get_probe(shape, image, x, y)
                coords.append((x, y, dx, dy))
                X.append(probe)

        predicted = classifier.predict(np.array(X))
    else:
        predicted = []
    return coords, predicted


def mark_region(image, holes, x, y, h, w, mark):
    image[line(x - 1, y, x + h - 1, y + w)] = 255 - mark
    image[line(x + h, y - 1, x, y + w - 1)] = 255 - mark
    image[line(x, y, x + h, y + w)] = mark
    image[line(x + h, y, x, y + w)] = mark
    holes[x:x + h, y:y + w] = mark


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


def calc_segments(arr, lpf):
    labels = label(arr == 0)
    bands = np.unique(labels, return_index=True, return_counts=True)
    return [i for i in zip(bands[1], bands[2])[1:] if i[1] >= lpf]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    shape = dx, dy = 40, 40
    model = 'model.pkl'

    if True:
        api = {'images': []}
        l = []
        r = []
        filenames = sorted(glob.glob('train/*.jpg'))
        for filename in filenames:
            image = skimage.io.imread(filename)
            img = {'name': '../'+filename, 'probes': []}
            api['images'].append(img)
            for point in open(filename + '.txt').read().split():
                if point.startswith('#'):
                    continue
                ty, tx, target = map(int, point.split('x'))
                if target > 127:
                    t=l
                    tt='l'
                else:
                    t=r
                    tt='r'
                if target not in t:
                    t.append(target)
                tx, ty, probe = get_probe(shape, image, tx - dx / 2, ty - dy / 2)
                prb = {'x': ty, 'y': tx, 'label': '{}{}'.format(tt,t.index(target)+1)}
                img['probes'].append(prb)

        import json
        json.dump( api, open('api/images.json', 'w'),indent=4)
        print api
        exit(0)

    try:
        out = sys.argv[2]
    except  IndexError:
        out = None

    if os.path.exists(model):
        classifier = joblib.load(model)
        logging.info(model)
    else:
        filenames = sorted(glob.glob('train/*.jpg'))
        try:
            classifier = train(filenames, shape)
            joblib.dump(classifier, model)
            logging.info(model)
        except Exception, e:
            logging.exception(e)
            classifier = None

    of = (120, 680)
    tf = ProjectiveTransform()
    src = np.array((
        (0, 0),
        (0, of[0]),
        (of[1], of[0]),
        (of[1], 0)
    ))
    dst = np.array((
        (117, 163),
        (145, 279),
        (613, 189),
        (610, 135)
    ))
    tf.estimate(src, dst)

    try:
        url = sys.argv[1]
        img = download_series(url, 16, adjust, tf=tf, of=of)
    except  IndexError:
        img = open_series('test/*.jpg', adjust, tf=tf, of=of)
    img = combine(img)

    coords, predicted = walk_image(img, shape, classifier)
    img_to_show = np.copy(img)
    holes = np.copy(img)
    holes[0:of[0], 0:of[1]] = 0
    for coord, pred in zip(coords, predicted):
        if pred > 127:
            x, y, dx, dy = coord
            mark_region(img_to_show, holes, x + dx / 3, y + dy / 3, dx / 3 * 2,
                        dy / 3 * 2, pred)
            # holes[x:x + dx, y:y + dy] = pred

    segments = calc_segments(holes[108:109, 0:of[1]], dy / 3)
    logging.info(segments)
    p_segments = ["%0.1f slot near %d place" % (float(p[1]) / dy, p[0] / dy + 1)
                  for p in segments]
    if p_segments:
        print ', '.join(p_segments)
    else:
        print "No spaces"


    if out:
        skimage.io.imsave(out, img_to_show)
        logging.info(out)
        try:
            j_segments = {
                "measurement": 'free',
                "tags": {
                    "pos": 108
                },
                "fields": {
                    "value": sum(map(lambda x: round(float(x[1])/dy), segments))
                }
            }
            logging.info(j_segments)
            client = InfluxDBClient('localhost', 8086, 'root', 'root', 'cars')
            client.write_points([j_segments])
        except Exception, e:
            logging.exception(e)


    else:
        name = os.path.join("train", datetime.datetime.now().isoformat())
        log = None


        def onclick(event):
            global log, name
            if event and event.xdata:
                try:
                    log.close()
                except:
                    pass
                log = open('{}.jpg.txt'.format(name), 'a+')
                log.write(
                    '{}x{}x{}\n'.format(int(event.xdata), int(event.ydata),
                                        int(255 / event.button)))


        fig = plt.figure()
        ax = fig.add_subplot()

        plt.imshow(img_to_show, cmap='Greys_r')

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()

        if log:
            skimage.io.imsave('{}.jpg'.format(name), img)
