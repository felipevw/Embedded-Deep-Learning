cmake_minimum_required(VERSION 3.0)
project( opencvgpu )

find_package( OpenCV 4.3 REQUIRED)
#find_package(CUDA REQUIRED)


#include_directories( ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( superres superres.cpp )
#target_link_libraries( DisplayImage ${OpenCV_LIBS} ${CUDA_LIBRARIES} )


target_link_libraries( superres ${OpenCV_LIBS} )


