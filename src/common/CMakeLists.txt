cmake_minimum_required(VERSION 2.8.3)
project(common)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    tf2_eigen
)

catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
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

enable_testing()
include(GoogleTest)
file(GLOB_RECURSE ALL_TESTS "*_test.cc")
foreach(FILE_PATH ${ALL_TESTS})
  get_filename_component(FILE_NAME ${FILE_PATH} NAME_WE)
  add_executable(${FILE_NAME} ${FILE_PATH})
  target_link_libraries(${FILE_NAME} ${PROJECT_NAME} gtest_main)
  gtest_discover_tests(${FILE_NAME})
endforeach(FILE_PATH)
