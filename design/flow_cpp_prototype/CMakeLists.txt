cmake_minimum_required(VERSION 3.12)
project(flow_cpp_prototype)

set(CMAKE_CXX_STANDARD 14)

include( /usr/local/Cellar/opencv/4.0.1/lib/cmake/opencv4/OpenCVConfig.cmake )
# include( /usr/local/opt/opencv@3/share/OpenCV/OpenCVConfig.cmake ) # find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(flow_cpp_prototype main.cpp)

# set(OpenCV_STATIC ON) # Does not work..
find_package(OpenCV REQUIRED)

target_link_libraries( flow_cpp_prototype ${OpenCV_LIBS} -static)