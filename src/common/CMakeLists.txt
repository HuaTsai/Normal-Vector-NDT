cmake_minimum_required(VERSION 2.8.3)
project(common)

set(CMAKE_CXX_FLAGS "-Wall -Wextra --coverage")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    tf2_eigen
    message_generation
    geometry_msgs
)

add_message_files(
  FILES
    PointCloudSensor.msg
    EgoPointClouds.msg
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
    geometry_msgs
    std_msgs
)

catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
    message_runtime
    tf2_eigen
  LIBRARIES
    ${PROJECT_NAME}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

set(ALL_SRCS
  src/eigen_utils.cc
  src/geometry_utils.cc
  src/other_utils.cc
)

add_library(${PROJECT_NAME} ${ALL_SRCS})

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(eigen_utils_test src/eigen_utils_test.cc)
add_executable(other_utils_test src/other_utils_test.cc)
target_link_libraries(eigen_utils_test ${PROJECT_NAME} gtest_main)
target_link_libraries(other_utils_test ${PROJECT_NAME} gtest_main)
enable_testing()
include(GoogleTest)
gtest_discover_tests(eigen_utils_test)
gtest_discover_tests(other_utils_test)
