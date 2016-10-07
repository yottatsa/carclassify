#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

import app
import classify

logging.basicConfig(level=logging.DEBUG)

try:
    url = sys.argv[1]
except IndexError:
    url = None

try:
    out = sys.argv[2]
except IndexError:
    out = None

if not out:
    wsgiapp = app.App(__name__, url).run()
else:
    classify.oneshot(url, out)
