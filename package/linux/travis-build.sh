# Build tensorflow.
echo "Adding tensorflow."
wget -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz"
tar -x -f libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz 

cp lib/libtensorflow.so $TRAVIS_BUILD_DIR/src/libs/tensorflow/lib/
cp lib/libtensorflow_framework.so $TRAVIS_BUILD_DIR/src/libs/tensorflow/lib/

# Copy libs and necessary dirs
cp include/tensorflow/c/c_api.h $TRAVIS_BUILD_DIR/src/libs/tensorflow/include/tensorflow/c/c_api.h
cp include/tensorflow/c/c_api_experimental.h $TRAVIS_BUILD_DIR/src/libs/tensorflow/include/tensorflow/c/c_api_experimental.h
cp include/tensorflow/c/eager/c_api.h $TRAVIS_BUILD_DIR/src/libs/tensorflow/include/tensorflow/c/eager/c_api.h

echo "Installing SDL2 library."
hg clone https://hg.libsdl.org/SDL
(cd SDL && mkdir build && cd build && ../configure && make && sudo make install)

# echo "Installing OpenCv 4.0"
# (wget -L "https://github.com/opencv/opencv/archive/4.0.1.zip" && unzip -q 4.0.1.zip && cd opencv-4.0.1 && mkdir build && cd build && cmake .. && make && sudo make install)

echo "Building FFmpeg."
(cd $TRAVIS_BUILD_DIR/src/libs/ffmpeg && ./configure --extra-cflags="-fPIC" --enable-nonfree --enable-shared  --enable-pic && make)

echo "Building project."
cd $TRAVIS_BUILD_DIR/
mkdir build  && cd build
cmake ..
make