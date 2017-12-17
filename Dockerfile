FROM ubuntu:16.04

RUN apt-get -y upgrade

RUN apt-get update && apt-get -y install \
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
  gcc
#  icu-devtools \
#  libicu-dev
#   vim

#RUN git clone https://github.com/SuperElastix/SimpleElastix
#RUN mkdir build
#
#WORKDIR build
#
#RUN cmake ../SimpleElastix/SuperBuild
#RUN make -j16   
#
#WORKDIR /build/SimpleITK-build/Wrapping/Python/Packaging
#RUN python setup.py install 
#
#WORKDIR /run

#RUN git clone https://github.com/xianyi/OpenBLAS
#WORKDIR OpenBLAS/
#RUN make -j16 FC=gfortran
#RUN make PREFIX=/opt/openblas install

RUN pip install --upgrade pip
RUN pip install matplotlib SimpleITK numpy psutil pytest tifffile

# We currently following 'master' to incorperate many recent bug fixes.
# When stable, use the following instead:
#RUN pip install intern
WORKDIR /work
RUN git clone https://github.com/jhuapl-boss/intern.git /work/intern --single-branch
WORKDIR /work/intern
RUN python setup.py install

# Set up ipython
RUN pip install ipython[all] jupyter scikit-image scikit-learn

# Build ndreg. Cache based on last commit.
WORKDIR /work
ADD https://api.github.com/repos/neurodata/ndreg/git/refs/heads/master version.json
RUN git clone https://github.com/neurodata/ndreg.git /work/ndreg --branch master --single-branch
WORKDIR /work/ndreg
RUN cmake -DCMAKE_CXX_FLAGS="-O3" . && make -j16 && make install
#RUN cmake . && make && make install

# Clone the registration package repo
WORKDIR /run
#RUN git clone https://github.com/neurodata/ndreg.git

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
