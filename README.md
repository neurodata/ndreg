# ndreg
[![Travis](https://travis-ci.org/neurodata/ndreg.svg?branch=master)](https://travis-ci.org/#)
[![Documentation Status](https://readthedocs.org/projects/ndreg/badge/?version=latest)](http://ndreg.readthedocs.io/en/latest/?badge=latest)
[![DockerHub](https://img.shields.io/docker/pulls/neurodata/ndreg.svg)](https://hub.docker.com/r/neurodata/ndreg)
[![arXiv Tag](https://img.shields.io/badge/arXiv-1605.02060-brightgreen.svg)](https://arxiv.org/abs/1605.02060)<br/>


This is Neurodata's open-source Python package that performs affine and deformable (LDDMM) image registration.   <br/>

## Sytem Requirements


The recommended way to use this package is to install [Docker](https://store.docker.com/search?offering=community&type=edition). Docker is currently available on Mac OS X El Capitan 10.11 and newer macOS releases, the following Ubuntu versions: Zesty 17.04 (LTS), Yakkety 16.10, Xenial 16.04 (LTS), Trusty 14.04 (LTS), and Windows 10.

Note: Certain optimizations made within the package may require a 64-bit machine with an Intel processor

### Software Dependencies (with version numbers)

The only software dependency needed if using the recommended method is Docker. However the main dependency
External libraries: <br/>
- Insight Segmentation and Registration Toolkit (ITK) -- 5.0 

### Versions tested on
We have tested the Docker image and build on macOS High Sierra (on MacBook Pro with 2.9 GHz Intel Core i7 and 16 GB RAM) and Ubuntu Xenial 16.04.3 LTS (with 64 GB RAM).

## Installation Guide

Once Docker is installed on your machine, pull the `neurodata/ndreg` image from Docker Hub [here](https://hub.docker.com/r/neurodata/ndreg) as follows: <br/>

`docker pull neurodata/ndreg` <br/>

It will typically take a few minutes to pull the entire Docker image.

## Demo

### Instructions to run demo

In order to use the functionality built into this Docker image, you need to run the Docker image:

`docker run -p 8888:8888 neurodata/ndreg` <br/>

This should print a link to the terminal console that looks like this: <br/>

`http://0.0.0.0:8888/?token=SOME_TOKEN` <br/>

Go to this link in your browser by copying and pasting it. <br/>

Next, click on `ndreg_demo_real_data.ipynb`. Once the notebook opens, you can run all cells by clicking on 'Cell' and then 'Run All'.

The expected run time for this demo is ~30 minutes.

### Expected Output

In order to identify ensure you have the correct output, refer to the Jupyter notebook [here](https://github.com/neurodata/ndreg/blob/master/ndreg_demo_real_data.ipynb).

## Use on your own data

In order to run `ndreg` on data you have on a local machine, use the following command in your terminal:

`docker run -v path/to/local/dir:/run/data/ -p 8888:8888 neurodata/ndreg` <br/>

`-v` is passed as an argument in order to mount a local volume to a Docker container. Replace `path/to/local/dir` with the absolute (or relative) path to your data locally. You do not need to modify the `:/run/data/` portion of the command; that portion of the command will mount your local volume to the `/run/data/`  directory in the Docker container. <br/>

If all goes well, a link similar to the one above should be printed. Go to this link in your browser by copying and pasting it. You should see a list of folders (including a `data` folder) and the demo Jupyter notebook `ndreg_demo_real_data.ipynb`. Your local data will appear in the `data` folder <br/>

In order to reuse the demo Jupyter notebook, modify the cell that contains the `params` variable and replace the path to the image data (and path to atlas data if you are not registering to the Allen Reference Atlas), voxel spacing variables, and brain orientations within that cell. (To understand more about the 3-letter orientation scheme see [here](http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm). (Note: all voxel spacing is recorded in millimeters (mm)). Once these variables are modified, the rest of the notebook can be run as is.
