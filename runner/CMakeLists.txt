cmake_minimum_required(VERSION 3.20)

set(TARGET cnncpp_run)

set(HIGHFIVE_USE_BOOST 0)
include(FetchContent)
FetchContent_Declare(
	HighFive
    GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
    GIT_TAG v2.9.0
)

FetchContent_MakeAvailable(HighFive)

add_executable(${TARGET} main.cpp)
add_executable(hd5test test_keras_hd5.cpp)

target_link_libraries(${TARGET} PRIVATE ${OpenCV_LIBS} cnncpp HighFive)
target_link_libraries(hd5test PRIVATE HighFive)
target_include_directories(${TARGET} PRIVATE ${OpenCV_INCLUDE_DIRS})
