project(BinPicking)

add_library(robotArm SHARED STATIC robotArm.cpp robotArm.h)

target_link_libraries (robotArm pclFunction)

