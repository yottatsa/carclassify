# -*- coding: utf-8 -*-

import datetime
import glob
import json
import logging
import os.path
import math

import imageio
import numpy as np
import skimage.io
from influxdb import InfluxDBClient
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
from skimage.draw import line
from skimage.exposure import equalize_adapthist
from skimage.morphology import label
from skimage.restoration import denoise_bilateral
from skimage.transform import rescale, ProjectiveTransform, warp
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from nms import nms

SCALE=0.5
SERIES=30

def greenify(img):
    if len(img.shape) > 2:
        return img
    img = gray2rgb(img)
    img[:, :, 0] = 0
    img[:, :, 2] = 0
    return img

def calc_segments(labels, lines=[4.0/5], dy=30):
    segments = []
    for l in lines:
        h, w = labels.shape
        line = int(l*h)
        arr = labels[line, :]
        bands = np.unique(label(arr == 0),
                          return_index=True,
                          return_counts=True)
        spaces = [i for i in zip(bands[1], bands[2])[1:] if i[1] >= dy / 2]
        total = sum(map(lambda x: round(float(x[1]) / dy), spaces))
        segments.append({
            'y': line,
            'w': w,
            'spaces': spaces,
            'total': int(total)})
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
    def __init__(self, model, filename, h=0, w=0):
        self.model = model
        self['name'] = filename
        self['probes'] = []
        self.data = self.model.images_cache.get(filename,
                                                skimage.io.imread(filename))
        self['h'], self['w'] = self.data.shape[:2]

    def add_probe(self, x, y, label):
        probe = Probe(self, x, y, label)
        self['probes'].append(probe)
        return probe


class Model(list):
    def __init__(self, db='train', shape=(35, 35), cutoff=0.9, overlap=0.3):
        self.db = db
        self.shape = shape
        self.modelfile = os.path.join(db, 'images.json')
        self.mode = 'mlp'
        self.clffile = os.path.join(db, self.mode + 'model.pkl')
        self.scalerfile = os.path.join(db, self.mode + 'scaler.pkl')
        self.clf = None
        self.scaler = None
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
        h, w, d = image.shape
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
        return rescale(img, SCALE).flatten().tolist()

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

            if os.path.exists(self.clffile) and os.path.exists(self.scalerfile):
                self.clf = joblib.load(self.clffile)
                self.scaler = joblib.load(self.scalerfile)

    def dump(self):
        if not self.clf:
            self.fit()
        json.dump({
            'images': self,
            'classes': self.classes,
            'metadata': self.metadata,
        }, open(self.modelfile, 'w'), indent=4)
        joblib.dump(self.clf, self.clffile)
        joblib.dump(self.scaler, self.scalerfile)

    def cleanup(self):
        storage = set(glob.glob(os.path.join(self.db, '*.png')))
        model = set([image['name'] for image in self])
        for f in storage - model:
            os.unlink(f)

    def _get_labeler(self, start=0, step=1):
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

    def _learn_data(self, image=None):
        self.classes = {}
        self.labels = {}
        labeler = {'l': self._get_labeler(), 'r': self._get_labeler(0, -1)}

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
        X, y = self._learn_data()
        self.clf, score, self.scaler = self._fit_mlp(X, y)
        self.metadata['score'] = score
        self.metadata['model'] = repr(self.clf)
        return self.clf

    def _fit_mlp(self, X, y):
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        size = (int(0.6 * (self.shape[0] * self.shape[1] * SCALE * SCALE)), ) * 2
        clf = MLPClassifier(solver='lbfgs', random_state=1,
                            hidden_layer_sizes=size, alpha=0.1,
                            activation='relu',
                            early_stopping=False)
        clf.fit(X, y)
        #score = 1
        score = max(cross_val_score(clf, X, y, cv=5))
        logging.info('Model: %s %f', clf, score)
        return clf, score, scaler

    def predict(self, image):
        if not self.clf:
            if not self.fit():
                return []

        dy, dx = self.shape
        coords = []
        X = []

        h, w, d = image.shape
        for x in range(0, w - dx + 1, dx / 10):
            for y in range(0, h - dy + 1, dy / 10):
                x, y, probe = self.get_probe(image, x, y)
                coords.append((x, y, dx, dy))
                X.append(probe)
        X = np.array(X)
        X = self.scaler.transform(X)
        T = self.clf.predict_proba(X)
        indices, classes = zip(
            *[(i, c) for i, c in enumerate(self.clf.classes_) if c > 0])
        P = T[:, (indices)]
        probability = [(P[l][c], classes[c]) for l, c in
                       enumerate(np.argmax(P, 1))]
        predicted = np.column_stack((coords, probability))
        predicted = [i for i in predicted if i[4] > self.cutoff]
        predicted = nms(predicted, self.overlap)
        return predicted


class ImageFabric():
    def __init__(self, model, url=None, dst=(
            (79, 116),
            (91, 274),
            (631, 194),
            (630, 118)
    )):
        def d(v):
            a, b = v
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        sizes = sorted(map(d, zip(dst, dst[1:] + dst[:1])))
        self.of = (
            int(math.floor(sizes[0])),
            int(math.floor(sizes[2]))
        )
        logging.debug('Image size: %s, %s', sizes, self.of)

        self.model = model
        self.url = url
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
        img = rgb2lab(img)
        return img

    def combine_gray(self, imgs, _warp):
        l = np.mean(imgs[..., 0], axis=0)
        a = np.mean(imgs[..., 1], axis=0)
        b = np.mean(imgs[..., 2], axis=0)
        img=img_as_float((l*1.7+(a+b+256)*0.16).astype(np.uint8))
        if _warp:
            img = warp(img, self.tf, output_shape=self.of)
        img = equalize_adapthist(img, clip_limit=0.01)
        return img

    def combine(self, imgs, _warp):
        l = np.mean(imgs[..., 0], axis=0)
        a = np.mean(imgs[..., 1], axis=0)
        b = np.mean(imgs[..., 2], axis=0)
        img = np.zeros(l.shape+(3,), np.float)
        img[..., 0] = l
        img[..., 1] = a
        img[..., 2] = b
        img = lab2rgb(img)
        if _warp:
            img = warp(img, self.tf, output_shape=self.of)
        img = equalize_adapthist(img, clip_limit=0.01)
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

    def fetch(self, warp=True):
        if self.url:
            img = self.download_series(self.url, SERIES)
        else:
            img = self.open_series('test/*.png')
        return self.combine(img, warp)

    def raw(self):
        img = self.fetch(warp=False)
        img = greenify(img)
        return imageio.imwrite(imageio.RETURN_BYTES, img, format='png')

    def get(self, name=None, draw=False):
        img = self.fetch()

        labels = np.zeros(shape=img.shape[:2])
        probes = []

        for x, y, dx, dy, confidence, label in self.model.predict(img):
            labels[y:y + dy, x:x + dx] = 1
            probes.append({
                'x': x,
                'y': y,
                'label': self.model.labels[label],
                'confidence': int(confidence * 100)})
            if draw:
                x = int(x)
                y = int(y)
                dx = int(dx)
                dy = int(dy)
                img[line(y - 2, x - 1, y + dy - 2, x + dx - 1)] = 1
                img[line(y + dy - 1, x - 2, y - 1, x + dx - 2)] = 1
                img[line(y - 1, x - 1, y + dy - 1, x + dx - 1)] = -1
                img[line(y + dy - 1, x - 1, y - 1, x + dx - 1)] = -1

        segments = calc_segments(labels)
        logging.info(segments)

        if not name:
            name = os.path.join(self.model.db,
                                datetime.datetime.now().isoformat()) + ".png"

        skimage.io.imsave(name, greenify(img))

        return {
            'name': name,
            'h': self.of[0],
            'w': self.of[1],
            'probes': probes,
            'segments': segments,
        }


def oneshot(url, out, dy=40):
    model = Model()
    images = ImageFabric(model, url)
    model.load(skip_model=True)
    probe = images.get(out, draw=True)

    data = []
    for segments in probe['segments']:
        data.append({
            "measurement": 'free',
            "tags": {
                "pos": segments['y']
            },
            "fields": {
                "value": float(segments['total'])
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
