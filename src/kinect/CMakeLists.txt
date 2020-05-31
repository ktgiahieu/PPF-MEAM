project(BinPicking)

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV REQUIRED)

add_library(kinect SHARED STATIC kinect.cpp kinect.h)
target_link_libraries (kinect ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(kinect PUBLIC ${DATATYPE_INCLUDE_DIR})

