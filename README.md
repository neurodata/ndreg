# clareg
Package that performs affine and LDDMM registration *easily* <br/>

# Installation

## Easy way

Pull image from Docker Hub (link [here](https://hub.docker.com/r/vikramc/simple-elastix/)) <br/>

`docker pull vikramc/simple-elastix` <br/>

## Hard way

Build image from Dockerfile <br\>

First clone this repository: `git clone git@github.com:vikramc1/clareg.git` <br/>
From inside the `clareg` directory: `docker build -t simple-elastix .` <br/>

# Running

In order to use the functionality built into this Docker image, you need to mount your local data volume as follows:

`docker run --rm -v /path/to/your/data:/run/data/ -p 8888:8888 simple-elastix` <br/>

This should print a link to the terminal console that looks like this: <br/>

`http://0.0.0.0:8888/?token=SOME_TOKEN` <br/>

Go to this link in your browser by copying and pasting it. <br/>

All of the data you mounted should be in the `/run/data` directory in the Docker image. Keep in mind that anything added to/deleted from that directory is also added to/deleted from the local machine (not just the Docker image) <br/>

You can look at the sample notebook in this repository to see how to use the two packages `Preprocessor` and `Registerer`




