project(cnncpp_gtests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(gtestlib)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(cnncpp_test tensor_test.cpp 
			   kernel_test.cpp 
			   convolution_test.cpp 
			   utils_test.cpp
                           pooling_test.cpp)

target_link_libraries(cnncpp_test cnncpp)
target_link_libraries(cnncpp_test gtest gtest_main)


