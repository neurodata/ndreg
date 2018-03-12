FROM ubuntu:16.04
LABEL maintainer="Vikram Chandrashekhar"
RUN apt-get -y upgrade

RUN apt-get update && apt-get -y --no-install-recommends install \
  build-essential \
  python-pip \
  python-all-dev \
  zlib1g-dev \
  libjpeg8-dev \
  libtiff5-dev \
  libfreetype6-dev \
  liblcms2-dev \
  libwebp-dev \
  tcl8.5-dev \
  tk8.5-dev \
  python-tk \
  libhdf5-dev \
  libinsighttoolkit4-dev \
  libfftw3-dev \
  libopenblas-base \
  libopenblas-dev \ 
  python \
  python-dev \
  git \
  build-essential \
  cmake \
  gcc \
  vim

# delete apt-get lists to free up space

RUN rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools
#RUN pip install matplotlib SimpleITK numpy psutil pytest tifffile

# We currently following 'master' to incorporate many recent bug fixes.
#WORKDIR /work
#RUN git clone https://github.com/jhuapl-boss/intern.git /work/intern --single-branch
#WORKDIR /work/intern
#RUN python setup.py install

# Build ndreg. Cache based on last commit.
WORKDIR /work
ADD https://api.github.com/repos/neurodata/ndreg/git/refs/heads/master version.json
RUN git clone https://github.com/neurodata/ndreg.git /work/ndreg --branch master --single-branch
WORKDIR /work/ndreg
RUN pip install -r requirements.txt
RUN cmake -DCMAKE_CXX_FLAGS="-O3" . && make -j16 && make install

WORKDIR /run
RUN cp /work/ndreg/ndreg_demo_real_data.ipynb ./

# Set up ipython
RUN pip install ipython[all] jupyter 

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
