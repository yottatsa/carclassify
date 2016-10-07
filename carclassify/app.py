#!/usr/bin/env python
# -*- coding: utf-8 -*-

import flask

from classify import Model, ImageFabric


def App(name, url=None):
    app = flask.Flask(name)
    app.model = Model()
    app.images = ImageFabric(app.model, url)
    app.model.load()
    app.model.cleanup()

    @app.route('/')
    def hello():
        return flask.redirect(flask.url_for('static', filename='index.html'),
                              code=302)

    @app.route('/api/images.json', methods=['GET'])
    def get_images():
        return flask.json.jsonify({
            'images': app.model.get(),
            'metadata': app.model.metadata
        })

    @app.route('/api/images.json', methods=['PUT'])
    def put():
        app.model.update(flask.request.json)
        app.model.dump()
        app.model.cleanup()
        return flask.json.jsonify({'metadata': app.model.metadata})

    @app.route('/api/live.json')
    def create_image():
        return flask.json.jsonify({'image': app.images.get()})

    return app
