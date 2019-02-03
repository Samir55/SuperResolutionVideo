# SR VIDEO
[![Build Status](https://travis-ci.org/Samir55/SuperResolutionVideo.svg?branch=master)](https://travis-ci.org/Samir55/SuperResolutionVideo)

This project aims to get a super resolution video version out of a low resolution one. It makes use of the famous H.264 codec as it enhances the I-Frames.
 
This is WIP and in its early stages.

The implementation is based on "Image Super-Resolution Using Deep Convolutional Networks" paper by "Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang
"  

## Dependencies
### From apt-get

```Console 
sudo apt-get install zlib1g zlib1g-dev bzip2 liblzma-dev cmake yasm python3-dev python3-pip python3-wheel python3-numpy libopencv-dev
```
### SDL2
```Console
hg clone https://hg.libsdl.org/SDL
cd SDL 
mkdir build && cd build
../configure
make && sudo make install
```
### FFMPEG
Change directory to the repository SuperVideoResolution directory
#### For CPU version
```Console 
cd external/ffmpeg
./configure --extra-cflags="-fPIC"  --enable-nonfree --enable-libnpp --enable-shared  --enable-pic
make
```
#### For CUDA compatible vesion
```Console
cd external/ffmpeg
./configure --extra-cflags="-fPIC" --enable-cuda --enable-cuvid --enable-nvenc --enable-nonfree --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --enable-shared  --enable-pic
make
```
### Tensorflow
#### For CPU version
```Console
wget -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz"
tar -x -f libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz 

cp lib/libtensorflow.so external/tensorflow/lib/
cp lib/libtensorflow_framework.so external/tensorflow/lib/

cp include/tensorflow/c/c_api.h external/tensorflow/include/tensorflow/c/c_api.h
cp include/tensorflow/c/c_api_experimental.h external/tensorflow/include/tensorflow/c/c_api_experimental.h
cp include/tensorflow/c/eager/c_api.h external/tensorflow/include/tensorflow/c/eager/c_api.h
```
#### For CUDA compatible vesion
You have to download tensorflow from the sources and after **configuring** bazel, do the following
```Console
bazel build //tensorflow:libtensorflow.so

cp tensorflow/bazel-bin/tensorflow/libtensorflow.so external/tensorflow/lib/
cp tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so external/tensorflow/lib/

cp tensorflow/tensorflow/c/c_api.h external/tensorflow/include/tensorflow/c/c_api.h
cp tensorflow/tensorflow/c/c_api_experimental.h external/tensorflow/include/tensorflow/c/c_api_experimental.h
cp tensorflow/tensorflow/c/eager/c_api.h external/tensorflow/include/tensorflow/c/eager/c_api.h
```

## Building
```Console
mkdir build 
cd build 
cmake ..
make
```
