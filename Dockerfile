FROM vikramc/itk
# install ndreg
# Build ndreg. Cache based on last commit.
WORKDIR /work
ADD https://api.github.com/repos/neurodata/ndreg/git/refs/heads/master version.json
RUN git clone https://github.com/neurodata/ndreg.git /work/ndreg --branch master --single-branch
#COPY . /work/ndreg/
WORKDIR /work/ndreg/
RUN pip install -r requirements.txt
RUN cmake -DCMAKE_CXX_FLAGS="-O3" . && make -j4 && make install

WORKDIR /run
RUN cp /work/ndreg/ndreg_demo_real_data.ipynb ./ && \
    cp /work/ndreg/data/Thy1eYFP_Control_9.tiff ./ && \
    cp /work/ndreg/data/ARA_50um.tiff ./

RUN rm -rf /home/itk/
RUN apt-get install --no-install-recommends -y python-tk
#    rm -rf /var/lib/apt/lists/* && pip3 install git+git://github.com/vikramc1/ndpull.git

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
