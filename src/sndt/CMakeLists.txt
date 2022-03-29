cmake_minimum_required(VERSION 3.0.2)
project(sndt)

set(CMAKE_CXX_FLAGS "-Wall -Wextra --coverage")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

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

set(ALL_SRCS
  src/ndt_cell.cc
  src/ndt_map.cc
  src/sndt_cell.cc
  src/sndt_map.cc
  src/matcher.cc
  src/matcher_pcl.cc
  src/visuals.cc
  src/outlier_reject.cc
  src/used_time.cc
  src/optimizer.cc
)

set(ALLLIBS
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
)

add_library(${PROJECT_NAME} ${ALL_SRCS})

target_link_libraries(${PROJECT_NAME} ${ALLLIBS})
