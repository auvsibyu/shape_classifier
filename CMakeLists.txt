cmake_minimum_required(VERSION 2.8)
project( ShapeClassifier )
find_package( OpenCV REQUIRED )
add_executable( ShapeClassifier src/ShapeClassifier.cpp
src/blob_detector.cpp
src/shape_detector.cpp
include/blob_detector.h
include/shape_detector.h )
target_link_libraries( ShapeClassifier ${OpenCV_LIBS} )
