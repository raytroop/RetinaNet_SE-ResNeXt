#!/usr/bin/env bash

python dataGen/setup.py build_ext --inplace
rm -rf build dataGen/compute_overlap.c .eggs
