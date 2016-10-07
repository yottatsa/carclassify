# -*- coding: utf-8 -*-

import datetime
import glob
import json
import logging
import os.path

import numpy as np
import skimage.io
from influxdb import InfluxDBClient
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist
from skimage.feature import canny
from skimage.morphology import label
from skimage.restoration import denoise_bilateral
from skimage.transform import rescale, ProjectiveTransform, warp
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

from nms import nms


def calc_segments(labels, lines=[90], dy=40):
    segments = []
    for line in lines:
        arr = labels[line, :]
        bands = np.unique(label(arr == 0),
                          return_index=True,
                          return_counts=True)
        spaces = [i for i in zip(bands[1], bands[2])[1:] if i[1] >= dy]
        total = sum(map(lambda x: round(float(x[1]) / dy), spaces))
        segments.append({'y': line, 'spaces': spaces, 'total': int(total)})
    return segments


class Probe(dict):
    def __init__(self, image, x, y, label):
        self.image = image
        x, y, self.data = self.image.model.get_probe(image.data, int(x), int(y),
                                                     name=image['name'])
        self['x'] = x
        self['y'] = y
        self['label'] = label


class Image(dict):
    def __init__(self, model, filename):
        self.model = model
        self['name'] = filename
        self['probes'] = []
        self.data = self.model.images_cache.get(filename,
                                                skimage.io.imread(filename))

    def add_probe(self, x, y, label):
        probe = Probe(self, x, y, label)
        self['probes'].append(probe)
        return probe


class Model(list):
    def __init__(self, db='train', shape=(50, 50), cutoff=0.8, overlap=0.3):
        self.db = db
        self.shape = shape
        self.modelfile = os.path.join(db, 'images.json')
        self.clffile = os.path.join(db, 'model.pkl')
        self.clf = None
        self.metadata = {}
        self.classes = {}
        self.labels = {}
        self.images_cache = {}
        self.probes_cache = {}
        self.cutoff = cutoff
        self.overlap = overlap

    def get(self):
        return self

    def get_probe(self, image, x, y, name=None):
        dy, dx = self.shape
        h, w = image.shape
        x = x - max(min(x, 0), x + dx - w)
        y = y - max(min(y, 0), y + dy - h)
        if name:
            data = self.probes_cache.get("{}-{}x{}".format(name, x, y),
                                         self.extract(
                                             image[y:y + dy, x:x + dx]))
        else:
            data = self.extract(image[y:y + dy, x:x + dx])
        return x, y, data

    def extract(self, img):
        features = []
        features = features + rescale(canny(rescale(img, 2), sigma=2),
                                      0.1).flatten().tolist()
        features = features + rescale(canny(img, sigma=1),
                                      0.2).flatten().tolist()
        features = features + rescale(img, 0.5).flatten().tolist()
        return features

    def add_image(self, data):
        if not data['probes']:
            return
        image = Image(self, data['name'])
        self.append(image)
        for probe_dict in data['probes']:
            image.add_probe(probe_dict['x'], probe_dict['y'],
                            probe_dict['label'])
        return image

    def update(self, model):
        for image in self:
            del image['probes'][:]
        del self[:]

        for image_dict in model['images']:
            self.add_image(image_dict)
        self.clf = None

    def load(self, skip_model=False):
        if os.path.exists(self.modelfile):
            model = json.load(open(self.modelfile))
            if not skip_model:
                self.update(model)

            self.classes = model.get('classes', {})
            self.metadata.update(model.get('metadata', {}))
            self.labels = dict([(v, k) for k, v in self.classes.items()])

            if os.path.exists(self.clffile):
                self.clf = joblib.load(self.clffile)

    def dump(self):
        if not self.clf:
            self.fit()
        json.dump({
            'images': self,
            'classes': self.classes,
            'metadata': self.metadata,
        }, open(self.modelfile, 'w'), indent=4)
        joblib.dump(self.clf, self.clffile)

    def cleanup(self):
        storage = set(glob.glob(os.path.join(self.db, '*.jpg')))
        model = set([image['name'] for image in self])
        for f in storage - model:
            os.unlink(f)

    def _get_labeler(self, start=255, step=-1):
        i = dict()
        i['i'] = start

        def f(label):
            if label in self.classes:
                return self.classes[label]
            i['i'] = i['i'] + step
            self.classes[label] = i['i']
            self.labels[i['i']] = label
            return i['i']

        return f

    def _learn_data(self, image=None, label=None, mode=None):
        self.classes = {}
        self.labels = {}
        labeler = {'l': self._get_labeler(), 'r': self._get_labeler(0, 1)}

        X = []
        y = []
        images = []
        labels = []
        for image in self:
            for probe in image['probes']:
                X.append(probe.data)
                y.append(labeler[probe['label'][0]](probe['label']))
        return np.array(X), np.array(y)

    def fit(self):
        if len(self) == 0:
            self.clf = None
            return

        Cs = np.logspace(-1, 5, 12)
        gammas = np.logspace(-8, -1, 7)
        X, y = self._learn_data()
        svc = svm.SVC(decision_function_shape='ovo', probability=True)
        clf = GridSearchCV(estimator=svc,
                           param_grid=dict(C=Cs, gamma=gammas),
                           cv=10, n_jobs=-1)
        clf.fit(X, y)

        self.clf = clf.best_estimator_
        self.metadata['score'] = clf.best_score_
        self.metadata['C'] = clf.best_estimator_.C
        self.metadata['gamma'] = clf.best_estimator_.gamma

        logging.info('Model: %s', self.metadata)

        return self.clf

    def predict(self, image):
        if not self.clf:
            if not self.fit():
                return []

        dy, dx = self.shape
        coords = []
        X = []

        h, w = image.shape
        for x in range(0, w - dx + 1, dx / 10):
            for y in range(0, h - dy + 1, dy / 10):
                x, y, probe = self.get_probe(image, x, y)
                coords.append((x, y, dx, dy))
                X.append(probe)

        T = self.clf.predict_proba(np.array(X))
        indices, classes = zip(
            *[(i, c) for i, c in enumerate(self.clf.classes_) if c > 127])
        P = T[:, (indices)]
        probability = [(P[l][c], classes[c]) for l, c in
                       enumerate(np.argmax(P, 1))]
        predicted = np.column_stack((coords, probability))
        predicted = [i for i in predicted if i[4] > self.cutoff]
        predicted = nms(predicted, self.overlap)
        return predicted


class ImageFabric():
    def __init__(self, model, url=None, of=(120, 680), dst=(
            (117, 163),
            (145, 279),
            (613, 190),
            (610, 130)
    )):
        self.model = model
        self.url = url
        self.of = of
        self.tf = ProjectiveTransform()
        src = np.array((
            (0, 0),
            (0, self.of[0]),
            (self.of[1], self.of[0]),
            (self.of[1], 0)
        ))
        dst = np.array(dst)
        self.tf.estimate(src, dst)

    def adjust(self, img, **kwargs):
        img = img_as_float(img)
        img = equalize_adapthist(img, clip_limit=0.03)  # gives more details
        return img

    def combine(self, imgs):
        img = np.mean(imgs, axis=0)
        img = denoise_bilateral(img, multichannel=True)  # smooth jpeg
        img = rgb2gray(img)
        img = img_as_ubyte(img)
        img = warp(img, self.tf, output_shape=self.of)
        return img

    def download_series(self, url, num, **kwargs):
        images = [
            self.adjust(skimage.io.imread(url),
                        **kwargs)
            for x in range(num - 1)]
        series = skimage.io.concatenate_images(images)
        return series

    def open_series(self, pattern, **kwargs):
        def load_func(f, *args, **kwargs):
            return self.adjust(skimage.io.imread(f), **kwargs)

        ic = skimage.io.ImageCollection(pattern, load_func=load_func, **kwargs)
        series = ic.concatenate()
        return series

    def fetch(self):
        if self.url:
            img = self.download_series(self.url, 8)
        else:
            img = self.open_series('test/*.jpg')
        return self.combine(img)

    def get(self, name=None):
        img = self.fetch()

        labels = np.zeros(shape=img.shape)
        probes = []

        for x, y, dx, dy, confidence, label in self.model.predict(img):
            labels[y:y + dy, x:x + dx] = 1
            probes.append({
                'x': x,
                'y': y,
                'label': self.model.labels[label],
                'confidence': int(confidence * 100)})

        segments = calc_segments(labels)
        logging.info(segments)

        if not name:
            name = os.path.join(self.model.db,
                                datetime.datetime.now().isoformat()) + ".jpg"
        skimage.io.imsave(name, img)

        return {
            'name': name,
            'probes': probes,
            'segments': segments,
        }


def oneshot(url, out, dy=40):
    model = Model()
    images = ImageFabric(model, url)
    model.load(skip_model=True)
    probe = images.get(out)

    data = []
    for segments in probe['segments']:
        data.append({
            "measurement": 'free',
            "tags": {
                "pos": segments['y']
            },
            "fields": {
                "value": segments['total']
            }
        })
        if segments['total'] != 0:
            spaces = [
                "%0.1f slot near %d place" % (float(p[1]) / dy, p[0] / dy + 1)
                for p in segments['spaces']]
            print ', '.join(spaces), ', total:', segments['total']
        else:
            print "No spaces"

    logging.info(data)

    try:
        client = InfluxDBClient('localhost', 8086, 'root', 'root', 'cars')
        client.write_points(data)
    except Exception, e:
        logging.exception(e)
