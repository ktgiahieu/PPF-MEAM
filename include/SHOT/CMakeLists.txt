project(BinPicking)

set(SHOT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "")

add_library(SHOT SHARED STATIC SHOT.cpp SHOT.h)
target_link_libraries(SHOT pclFunction meshSampling HPR)
target_include_directories(SHOT PUBLIC ${PCLFUNCTION_INCLUDE_DIR} ${MESHSAMPLING_INCLUDE_DIR} ${HPR_INCLUDE_DIR})



