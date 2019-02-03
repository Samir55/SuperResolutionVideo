CMAKE_MINIMUM_REQUIRED(VERSION 3.2 FATAL_ERROR)

PROJECT(SUPER_RES_VIDEO)

SET(CMAKE_CXX_STANDARD 11)

#############################
# Project Source and Header Files
#############################

SET(HEADER_FILES
        Model.h
        )

SET(SOURCE_FILES
        src/main.cpp
        src/${HEADER_FILES}
        )

#############################
# Dependencies
#############################

# FFmpeg dependencies.
FIND_PACKAGE(SDL2 REQUIRED)
FIND_PACKAGE(ZLIB REQUIRED)
FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(LibLZMA REQUIRED)

# OpenCV and TensorFlow.
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(TensorflowCC COMPONENTS SHARED)

# CUDA.
FIND_PACKAGE(CUDA)

#############################
# Include Necessary Headers
#############################

SET(FFMPEG_DIR ${PROJECT_SOURCE_DIR}/external/ffmpeg)

SET(FFMPEG_DIRS
        ${FFMPEG_DIR}
        ${FFMPEG_DIR}/libavformat
        ${FFMPEG_DIR}/libavcodec
        ${FFMPEG_DIR}/libavdevice
        ${FFMPEG_DIR}/libavfilter
        ${FFMPEG_DIR}/libswresample
        ${FFMPEG_DIR}/libswscale
        )

SET(INCLUDE_DIRS
        ${SDL2_INCLUDE_DIRS}
        ${ZLIB_INCLUDE_DIRS}
        ${BZIP2_INCLUDE_DIR}
        ${LIBLZMA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${FFMPEG_DIRS}
        )

INCLUDE_DIRECTORIES(
        ${INCLUDE_DIRS}
)

#############################
# Libraries
#############################

SET(FFMPEG_LIBRARIES
        ${FFMPEG_DIR}/libavformat/libavformat.a
        ${FFMPEG_DIR}/libavcodec/libavcodec.a
        ${FFMPEG_DIR}/libavutil/libavutil.a
        ${FFMPEG_DIR}/libavdevice/libavdevice.a
        ${FFMPEG_DIR}/libavfilter/libavfilter.a
        ${FFMPEG_DIR}/libswresample/libswresample.a
        ${FFMPEG_DIR}/libswscale/libswscale.a
        )

SET(DEPENDENCIES
        -lSDL2
        ${ZLIB_LIBRARIES}
        ${BZIP2_LIBRARIES}
        ${LIBLZMA_LIBRARIES}
        ${OpenCV_LIBS}
        )

#############################
# Add Executables
#############################

ADD_EXECUTABLE(SUPER_RES_VIDEO ${SOURCE_FILES})

#############################
# Link Libraries.
#############################

TARGET_LINK_LIBRARIES(SUPER_RES_VIDEO ${FFMPEG_LIBRARIES} ${DEPENDENCIES})

# Try the shared tensorflow library first.
IF (TensorflowCC_Shared_FOUND)
    TARGET_LINK_LIBRARIES(SUPER_RES_VIDEO TensorflowCC::Shared)
ELSE ()
    FIND_PACKAGE(TensorflowCC REQUIRED COMPONENTS Static)
    TARGET_LINK_LIBRARIES(SUPER_RES_VIDEO TensorflowCC::Static)
ENDIF ()


IF (CUDA_FOUND)
    TARGET_LINK_LIBRARIES(SUPER_RES_VIDEO ${CUDA_LIBRARIES})
ENDIF ()
