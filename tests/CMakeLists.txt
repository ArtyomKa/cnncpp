project(cnncpp_gtests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(cnncpp_test tensor_test.cpp 
			   kernel_test.cpp 
			   convolution_test.cpp 
			   utils_test.cpp
			   fully_connected_test.cpp
                           pooling_test.cpp)

target_link_libraries(cnncpp_test cnncpp)
target_link_libraries(cnncpp_test gtest gtest_main)


