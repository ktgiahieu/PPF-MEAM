project(BinPicking)

set(B2BTL_MEAM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "")

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED )

add_library(B2BTL_MEAM SHARED STATIC B2BTL_MEAM.cpp B2BTL_MEAM.h)
target_link_libraries(B2BTL_MEAM ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_include_directories(B2BTL_MEAM PUBLIC ${DATATYPE_INCLUDE_DIR})



