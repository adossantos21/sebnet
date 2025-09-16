#!/bin/bash

# This script installs mmpretrain and mmsegmentation as editable packages
# to make them importable and ensure modules are properly registered for use.

# First, ensure openmim is installed
pip install -U openmim

# Install mmpretrain
cd mmpretrain
mim install -e .
cd ..

# Install mmsegmentation
cd mmsegmentation
pip install -v -e .
cd ..

# Resolve MMCV conflict between mmpretrain and mmsegmentation
pip uninstall mmcv -y
mim install mmengine
mim install "mmcv>=2.0.0"

# Install py-edge-eval forked from adossantos21 - this version is packaged.
pip install git+ssh://git@github.com/adossantos21/py-edge-eval.git@sebnet

echo "Installation complete. You can now use the software seamlessly."
