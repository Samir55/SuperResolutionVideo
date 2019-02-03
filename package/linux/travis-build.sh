echo "Installing tensorflow_cc"
cd $TRAVIS_BUILD_DIR/external/tensorflow_cc/tensorflow_cc && mkdir build && cd build && cmake .. && make && sudo make install

echo "Installing SDL2 Library."
hg clone https://hg.libsdl.org/SDL
cd SDL && mkdir build && cd build && ../configure && make && sudo make install

echo "Installing OpenCv 4.0"
wget -L "https://github.com/opencv/opencv/archive/4.0.1.zip" && unzip 4.0.1.zip && cd opencv-4.0.1 && mkdir build && cd build && cmake .. && make && sudo make install

echo "Building ffmpeg"
cd $TRAVIS_BUILD_DIR/external/ffmpeg && ./configure && make

echo "Starting building"
cd $TRAVIS_BUILD_DIR/
mkdir build  && cd build
cmake ..
make