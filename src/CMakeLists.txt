project(BinPicking)

find_package(PCL 1.2 REQUIRED)

add_subdirectory(kinect)
add_subdirectory(robotArm)
add_subdirectory(socket)

add_executable (BinPicking binPicking.cpp)
target_link_libraries (BinPicking kinect robotArm socket SHOT PPF B2BTL_MEAM)
target_include_directories(BinPicking PUBLIC ${PCLFUNCTION_INCLUDE_DIR} ${SHOT_INCLUDE_DIR} ${PPF_INCLUDE_DIR} ${B2BTL_MEAM_INCLUDE_DIR})
