#!/bin/bash

echo 'install tensorflow on Jetson Nano'

# official
https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

# the rigth tutorial ?
https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770

Python 3.6+JetPack4.6.3

$ sudo apt-get update
$ sudo apt-get install -y python3-pip pkg-config
$ sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
$ sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
$ sudo pip3 install --verbose 'protobuf<4' 'Cython<3'
$ sudo wget --no-check-certificate https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
$ sudo pip3 install --verbose tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl


===============================================================

sudo apt update
sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo pip3 install -U testresources setuptools

# $ sudo pip3 install -U numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.7.0

#WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

pip3 install -U numpy future mock keras_preprocessing keras_applications gast protobuf pybind11 cython pkgconfig packaging h5py


#$ sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512 tensorflow==2.12.0+nv23.06
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow==2.7.0+nv21.12
