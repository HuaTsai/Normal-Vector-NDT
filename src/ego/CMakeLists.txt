cmake_minimum_required(VERSION 3.0.2)
project(ego)

set(CMAKE_CXX_FLAGS "-Wall --coverage")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    pcl_ros
    sensor_msgs
    common
    visualization_msgs
    conti_radar
    tf2_sensor_msgs
    tf2_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(ego src/ego.cc)

target_link_libraries(ego ${catkin_LIBRARIES})
