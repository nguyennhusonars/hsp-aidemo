cmake_minimum_required(VERSION 3.10)

PROJECT(aidemo LANGUAGES CXX)

if(BUILD_X86)
    set(CMAKE_C_COMPILER /usr/bin/clang)
    set(CMAKE_CXX_COMPILER /usr/bin/clang++)
elseif(BUILD_AARCH64)
    set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc-9)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++-9)
endif()

set(CMAKE_BUILD_TYPE Debug)
if(BUILD_X86)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/libs/opencv-4.9.0/x86)
elseif(BUILD_AARCH64)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/libs/opencv-4.9.0/aarch64)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -fexceptions")

INCLUDE_DIRECTORIES(
    ${OpenCV_DIR}/include/opencv4
    ${CMAKE_SOURCE_DIR}/include/nlohmann
    ${CMAKE_SOURCE_DIR}/include/SNPE2.19
    ${CMAKE_SOURCE_DIR}/src
)

# Find the GStreamer package
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0)
include_directories(${GSTREAMER_INCLUDE_DIRS})

file (GLOB SRC_FILES 
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

if(BUILD_X86)
    LINK_DIRECTORIES(${PROJECT_NAME}
        ${OpenCV_DIR}/lib
        ${CMAKE_SOURCE_DIR}/libs/SNPE2.19/x86_64-linux-clang
    )
elseif(BUILD_AARCH64)
    LINK_DIRECTORIES(${PROJECT_NAME}
        ${OpenCV_DIR}/lib
        ${CMAKE_SOURCE_DIR}/libs/SNPE2.19/aarch64-ubuntu-gcc9.4
    )
endif()

ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_FILES})

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    -lm -ldl -lpthread
    -lopencv_core
    -lopencv_imgproc
    -lopencv_imgcodecs
    -lopencv_videoio
    -lopencv_highgui
    -lSNPE -lX11
    ${GSTREAMER_LIBRARIES}
) 

add_definitions(-O2 -pthread -frtti)
