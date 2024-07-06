cmake_minimum_required(VERSION 3.1)
project(cnncpp LANGUAGES CXX)

set(LIB_TARGET cnncpp)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV)

add_subdirectory(tests)
add_subdirectory(runner)

#sources
set(LIB_SRCS src/tensor.cpp 
	     src/utils.cpp
	     src/ops.cpp)

add_library(${LIB_TARGET} ${LIB_SRCS})
target_include_directories(${LIB_TARGET} PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LIB_TARGET} PUBLIC  ${OpenCV_LIBS})
#tests
enable_testing()
add_test(NAME cnncpp_test COMMAND $<TARGET_FILE:cnncpp_test>)
