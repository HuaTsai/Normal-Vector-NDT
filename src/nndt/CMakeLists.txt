cmake_minimum_required(VERSION 3.0.2)
project(nndt)

set(CMAKE_CXX_FLAGS "-Wall -Wextra --coverage")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

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
  src/nmap.cc
  src/opt.cc
  src/matcher.cc
  src/timer.cc
  src/orj.cc
)

set(ALLLIBS
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
)

add_library(${PROJECT_NAME} ${ALL_SRCS})

target_link_libraries(${PROJECT_NAME} ${ALLLIBS})

add_executable(cell_test src/cell_test.cc)
add_executable(nmap_test src/nmap_test.cc)
add_executable(matcher_test src/matcher_test.cc)
add_executable(timer_test src/timer_test.cc)
target_link_libraries(cell_test ${PROJECT_NAME} gtest_main)
target_link_libraries(nmap_test ${PROJECT_NAME} gtest_main)
target_link_libraries(matcher_test ${PROJECT_NAME} gtest_main)
target_link_libraries(timer_test ${PROJECT_NAME} gtest_main)
enable_testing()
include(GoogleTest)
gtest_discover_tests(cell_test)
gtest_discover_tests(nmap_test)
gtest_discover_tests(matcher_test)
gtest_discover_tests(timer_test)
