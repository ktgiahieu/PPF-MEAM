project(BinPicking)

find_package(PCL 1.2 REQUIRED)

add_subdirectory(kinect)
add_subdirectory(robotArm)
add_subdirectory(socket)

add_executable (BinPicking binPicking.cpp)
target_link_libraries (BinPicking kinect robotArm socket)