FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

## install basics
RUN apt-get update -y
RUN apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ openssh-server vim wget
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg

## install python
RUN apt-get install -y python3-dev
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

## run server for ssh
RUN mkdir /var/run/sshd
RUN echo 'root:sgt' | chpasswd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN mkdir /root/.ssh
EXPOSE 22
CMD /usr/sbin/sshd -D

## install required packages
RUN pip install opencv-python
RUN pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install tensorboard einops
RUN pip install cloudpickle tabulate click shapely
RUN pip install Cython
RUN pip install cython-bbox
RUN pip install motmetrics lap
RUN pip install --no-index torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.10.2+cu113.html
RUN pip install torch-geometric==2.1.0
RUN pip install fvcore==0.1.5.post20220512

## install detectron2 with gpu support
RUN nvcc --version
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
# install from detectron2 repo
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
