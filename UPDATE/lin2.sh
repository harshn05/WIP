#!/bin/bash
export PATH="/media/Harsh/python64/lin2/bin:$PATH"
conda update anaconda
conda update conda
conda update --all
conda install mpmath pyqtgraph
conda install -c https://conda.anaconda.org/menpo mayavi
conda install -c https://conda.binstar.org/menpo opencv3
conda install -c asmeurer gcc=4.8.5
