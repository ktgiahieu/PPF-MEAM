project(BinPicking)

set(PPF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "")

add_library(PPF SHARED STATIC PPF.cpp PPF.h)
target_link_libraries(PPF pclFunction meshSampling HPR)
target_include_directories(PPF PUBLIC ${PCLFUNCTION_INCLUDE_DIR} ${MESHSAMPLING_INCLUDE_DIR} ${HPR_INCLUDE_DIR})



