cmake_minimum_required(VERSION 3.9)
project(SampleCode)

# 告知当前使用的交叉编译方式
#SET(CMAKE_SYSTEM_NAME Linux)
# 制定编译工具
#SET(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")

set(CMAKE_CXX_STANDARD 11)
include_directories(/home/sy/code/project/snpe-1.22.0.212/include/zdl)

aux_source_directory(./jni SRC_LIST)
add_executable(SampleCode jni/main.cpp ${SRC_LIST})

target_link_libraries(SampleCode
#        /home/sy/code/project/snpe-1.22.0.212/lib/aarch64-linux-gcc4.9/libSNPE.so
#        /home/sy/code/project/snpe-1.22.0.212/lib/aarch64-linux-gcc4.9/libsymphony-cpu.so
        /home/sy/code/project/snpe-1.22.0.212/lib/x86_64-linux-clang/libSNPE.so
        /home/sy/code/project/snpe-1.22.0.212/lib/x86_64-linux-clang/libsymphony-cpu.so)
