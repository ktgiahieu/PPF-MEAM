project(BinPicking)

add_library(kinect SHARED STATIC kinect.cpp kinect.h)
target_link_libraries (kinect pclFunction B2BTL_MEAM meshSampling HPR)
target_include_directories(kinect PUBLIC ${PCLFUNCTION_INCLUDE_DIR} ${MESHSAMPLING_INCLUDE_DIR} ${HPR_INCLUDE_DIR} ${B2BTL_MEAM_INCLUDE_DIR})
