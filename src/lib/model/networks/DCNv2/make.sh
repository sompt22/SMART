#!/usr/bin/env bash
rm -f *.so
rm -rf build/
python setup.py build_ext --inplace
