#!/bin/bash

# Compile cpp subsampling
cd /content/ADL4CV/4D-PLS/cpp_wrappers/cpp_subsampling
python3 /content/ADL4CV/4D-PLS/cpp_wrappers/cpp_subsampling/setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd /content/ADL4CV/4D-PLS/cpp_wrappers/cpp_neighbors
python3 /content/ADL4CV/4D-PLS/cpp_wrappers/cpp_neighbors/setup.py build_ext --inplace
cd ..