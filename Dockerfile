FROM ubuntu:16.04

# FROM: https://hub.docker.com/r/rafaelmonteiro/deep-learning-toolbox/~/dockerfile/
# FROM: https://hub.docker.com/r/windj007/jupyter-keras-tools/~/dockerfile/

RUN apt-get clean && apt-get update
RUN apt-get install -yqq python python-pip python-dev build-essential \
    git wget gfortran libatlas-base-dev libatlas-dev libatlas3-base libhdf5-dev \
    libfreetype6-dev libpng12-dev pkg-config libxml2-dev libxslt-dev \
    libboost-program-options-dev zlib1g-dev libboost-python-dev

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8888
VOLUME ["/notebook", "/jupyter/certs"]
WORKDIR /notebook

ADD test_scripts /test_scripts
ADD jupyter /jupyter

ENV JUPYTER_CONFIG_DIR="/jupyter"

CMD ["jupyter", "notebook", "--ip=0.0.0.0"]
