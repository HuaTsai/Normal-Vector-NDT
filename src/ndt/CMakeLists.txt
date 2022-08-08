cmake_minimum_required(VERSION 3.0.2)
project(ndt)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
# find_package(OpenMP REQUIRED) link OpenMP::OpenMP_CXX
find_package(Ceres REQUIRED)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    common
    pcl_ros
)

catkin_package(
  CATKIN_DEPENDS
    common
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

set(ALL_SRCS
  src/cell.cc
  src/cell2d.cc
  src/nmap.cc
  src/nmap2d.cc
  src/matcher.cc
  src/matcher2d.cc
  src/opt.cc
  src/orj.cc
  src/timer.cc
  src/visuals.cc
)

set(ALLLIBS
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
)

add_library(${PROJECT_NAME} ${ALL_SRCS})

target_link_libraries(${PROJECT_NAME} ${ALLLIBS})

enable_testing()
include(GoogleTest)
file(GLOB_RECURSE ALL_TESTS "*_test.cc")
foreach(FILE_PATH ${ALL_TESTS})
  get_filename_component(FILE_NAME ${FILE_PATH} NAME_WE)
  add_executable(${FILE_NAME} ${FILE_PATH})
  target_link_libraries(${FILE_NAME} ${PROJECT_NAME} gtest_main)
  gtest_discover_tests(${FILE_NAME})
endforeach(FILE_PATH)
