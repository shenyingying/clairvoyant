cmake_minimum_required(VERSION 3.9)
project(cfc)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

find_package(OpenCV 3.4 REQUIRED)
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE__FLAGS} ${OpenMP__FLAGS}")
    MESSAGE("OpenMP FOUND")
endif ()

aux_source_directory(src SRC_LIST)
add_executable(cfc main.cpp ${SRC_LIST})

target_link_libraries(cfc ${OpenCV_LIBS}  -lm)
#target_link_libraries(cfc ${OpenCV_LIBS}  -fopenmp -lm)