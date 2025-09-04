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
mim install -e .
cd ..

echo "Installation complete. You can now use the software seamlessly."
