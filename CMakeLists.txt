cmake_minimum_required(VERSION 3.1)
project(cnncpp LANGUAGES CXX)

set(LIB_TARGET cnncpp)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV)

include(FetchContent)
set(HIGHFIVE_USE_BOOST 0)
FetchContent_Declare(
  HighFive
  GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
  GIT_TAG v2.9.0)

FetchContent_MakeAvailable(HighFive)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

add_subdirectory(tests)
add_subdirectory(runner)

# sources
set(LIB_SRCS src/tensor.cpp src/utils.cpp src/layers.cpp src/convolution.cpp
             src/pooling.cpp src/network.cpp)

add_library(${LIB_TARGET} ${LIB_SRCS})
target_include_directories(${LIB_TARGET} PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
  ${LIB_TARGET} PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json HighFive)
# tests
enable_testing()
add_test(NAME cnncpp_test COMMAND $<TARGET_FILE:cnncpp_test>)
