cmake_minimum_required(VERSION 3.0.2)
project(pcl_exec)

set(CMAKE_CXX_FLAGS "-Wall -Wextra --coverage")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(PCL 1.12.1 EXACT REQUIRED
  COMPONENTS
    registration
    io
)

find_package(catkin REQUIRED
  COMPONENTS
    common
    roscpp
    metric
)

catkin_package()

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

set(ALLLIBS
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  boost_program_options
)

add_executable(exp6 src/exp6.cc)
target_link_libraries(exp6 ${ALLLIBS})

enable_testing()
include(GoogleTest)
file(GLOB_RECURSE ALL_TESTS "*_test.cc")
foreach(FILE_PATH ${ALL_TESTS})
  get_filename_component(FILENAME ${FILE_PATH} NAME_WE)
  add_executable(${FILENAME} ${FILE_PATH})
  target_link_libraries(${FILENAME} ${ALLLIBS} gtest_main)
  gtest_discover_tests(${FILENAME})
endforeach(FILE_PATH)
