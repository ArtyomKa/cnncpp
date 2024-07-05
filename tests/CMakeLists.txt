project(cnncpp_gtests)

add_subdirectory(gtestlib)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(cnncpp_test tensor_test.cpp)

target_link_libraries(cnncpp_test cnncpp)
target_link_libraries(cnncpp_test gtest gtest_main)


