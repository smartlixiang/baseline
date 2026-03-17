FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        ca-certificates \
        curl \
        git \
        locales \
        python3 \
        python3-dev \
        python3-pip \
        rsync \
        software-properties-common \
        tree \
        unzip \
        vim \
        wget && \
    sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --upgrade pip && \
    pip install --upgrade \
        numpy \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        tqdm \
        Cython \
        jupyter \
        torch \
        torchvision \
        jax \
        flax \
        neural-tangents && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . /data_diet
WORKDIR /data_diet

EXPOSE 2222
