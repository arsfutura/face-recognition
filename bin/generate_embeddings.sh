#!/usr/bin/env bash

rm -rf data/aligned/*
rm -rf data/features/*

../util/align-dlib.py ../data/images align outerEyesAndNose ../data/aligned --size 96
./batch-represent/main.lua -outDir ../data/features -data ../data/aligned
