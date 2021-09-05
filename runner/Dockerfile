FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
        build-essential \
        apt-utils \
        apt-transport-https \
        ca-certificates \
        software-properties-common \
        pkg-config \
        curl \
        wget \
        unzip \
        gpg-agent \
        sudo \
        locales && \
    locale-gen en_US.UTF-8 && \
    locale-gen vi_VN.UTF-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN sudo add-apt-repository universe
RUN sudo apt-get update
RUN sudo apt-get install -y python3 python3-pip
# Set the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


RUN pip3 install torch==1.8.2+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
RUN pip3 install transformers==4.4.2
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install tensorboardX
RUN pip3 install wheel
RUN pip3 install flask
RUN pip3 install fastapi
RUN pip3 install scikit-build
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python>=4.1.2

RUN ln -s /usr/bin/python3 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip