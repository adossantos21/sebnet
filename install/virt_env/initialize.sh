#!/bin/bash

# This script installs mmpretrain and mmsegmentation as editable packages
# to make them importable and ensure modules are properly registered for use.

# Install dependencies
pip install scipy packaging ftfy regex cityscapesscripts

# Ensure openmim is installed
pip install -U openmim

# Install mmpretrain
cd mmpretrain
mim install -e .
cd ..

# Install mmsegmentation
cd mmsegmentation
pip install -v -e .
cd ..

# Install py-edge-eval forked from adossantos21 - this version is packaged.
pip install git+ssh://git@github.com/adossantos21/py-edge-eval.git@sebnet

echo "Installation complete. You can now use the software seamlessly."
