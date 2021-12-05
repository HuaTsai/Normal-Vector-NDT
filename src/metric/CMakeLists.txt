cmake_minimum_required(VERSION 3.0.2)
project(metric)

set(CMAKE_CXX_FLAGS "-Wall --coverage")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(Ceres REQUIRED)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    common
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
  src/metric.cc
)

set(ALLLIBS
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
)

add_library(${PROJECT_NAME} ${ALL_SRCS})

target_link_libraries(${PROJECT_NAME} ${ALLLIBS})

add_executable(metric_test src/metric_test.cc)

target_link_libraries(metric_test ${PROJECT_NAME})
