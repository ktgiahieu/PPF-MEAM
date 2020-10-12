cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(PPF_MEAM)

add_subdirectory(include)

add_executable (PPF_MEAM demo.cpp)
target_link_libraries (PPF_MEAM B2BTL_MEAM)
target_include_directories(PPF_MEAM PUBLIC ${PCL_INCLUDE_DIRS} ${PCLFUNCTION_INCLUDE_DIR} ${B2BTL_MEAM_INCLUDE_DIR})
