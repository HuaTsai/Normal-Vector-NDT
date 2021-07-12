cmake_minimum_required(VERSION 3.0.2)
project(sndt)

set(CMAKE_CXX_FLAGS "-Wall --coverage")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(Ceres REQUIRED)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    pcl_ros
    sensor_msgs
    common
    visualization_msgs
    tf2_sensor_msgs
    normal2d
    tf2_ros
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    common
    normal2d
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# Library

set (${PROJECT_NAME}_LIB_SRCS
  src/ndt_cell.cc
  src/ndt_grid.cc
  src/ndt_map.cc
  src/ndt_matcher.cc
  src/ndt_visualizations.cc
)

set(ALLLIBS ${catkin_LIBRARIES} ${CERES_LIBRARIES})

add_library(${PROJECT_NAME} ${${PROJECT_NAME}_LIB_SRCS})

target_link_libraries(${PROJECT_NAME} ${ALLLIBS})
