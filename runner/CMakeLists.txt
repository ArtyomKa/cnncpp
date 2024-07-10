cmake_minimum_required(VERSION 3.20)

set(TARGET cnncpp_run)
add_subdirectory(HighFive)
add_executable(${TARGET} main.cpp)

target_link_libraries(${TARGET} PRIVATE ${OpenCV_LIBS} cnncpp HighFive)
target_include_directories(${TARGET} PRIVATE ${OpenCV_INCLUDE_DIRS})
