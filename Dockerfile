FROM robbyjo/ubuntu-mkl:16.04-2018.1

RUN apt-get update && apt-get install --no-install-recommends -y \
    apt-utils \
    build-essential \
    curl \
    git \
    python \
    python-dev \
    software-properties-common \
    libtbb-dev && \
    rm -rf /var/lib/apt/lists/*

# Install the latest CMake release
WORKDIR /tmp/
RUN curl  -L https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    pip install cmake

# install latest gcc version
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install --no-install-recommends -y gcc-7 g++-7 && \
    rm -rf /var/lib/apt/lists/*

# install ITK
# be sure to use the MKL libraries
WORKDIR /home/itk/
RUN git clone https://github.com/InsightSoftwareConsortium/ITK.git && \
    mkdir ITK-build && cd ITK-build/ && cmake -DITK_USE_SYSTEM_FFTW=ON -DITK_USE_FFTWF=ON -DITK_USE_FFTWD=ON -DITK_USE_MKL=ON ../ITK && \
    make -j16 && make install

# install ndreg
# Build ndreg. Cache based on last commit.
WORKDIR /work
ADD https://api.github.com/repos/neurodata/ndreg/git/refs/heads/vik-optimize version.json
RUN git clone https://github.com/neurodata/ndreg.git /work/ndreg --branch vik-optimize --single-branch
WORKDIR /work/ndreg
RUN pip install -r requirements.txt
RUN cmake -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_FLAGS="-O3" . && make -j16 && make install

WORKDIR /run
RUN cp /work/ndreg/ndreg_demo_real_data.ipynb ./

RUN rm -rf /home/itk/

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
