cmake_minimum_required(VERSION 2.8.3)
project(common)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

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
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
