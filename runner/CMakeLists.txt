cmake_minimum_required(VERSION 3.20)

set(TARGET cnncpp_run)
add_subdirectory(HighFive)
add_executable(${TARGET} main.cpp)
add_executable(hd5test test_keras_hd5.cpp)

target_link_libraries(${TARGET} PRIVATE ${OpenCV_LIBS} cnncpp HighFive)
target_link_libraries(hd5test PRIVATE HighFive)
target_include_directories(${TARGET} PRIVATE ${OpenCV_INCLUDE_DIRS})
