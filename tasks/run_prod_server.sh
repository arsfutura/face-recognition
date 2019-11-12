#!/usr/bin/env bash

uwsgi --http 0.0.0.0:5000 --wsgi-file api/app.py --callable app --processes 5 --threads 2