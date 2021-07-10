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
    message_generation
    visualization_msgs
    conti_radar
    tf2_sensor_msgs
    normal2d
    tf2_ros
)

add_message_files(
  FILES
    NDTCellMsg.msg
    NDTMapMsg.msg  
    PointCloudSensor.msg
    EgoPointClouds.msg
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    sensor_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    roscpp
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

add_library(${PROJECT_NAME} src/dummy.cpp)

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_generate_messages_cpp)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

# Executables

set(ALLLIBS ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PROJECT_NAME} config++)

file(GLOB_RECURSE ALL_SRCS "*.cc")
foreach(FILE_PATH ${ALL_SRCS})
  get_filename_component(FILENAME ${FILE_PATH} NAME_WE)
  add_executable(${FILENAME} ${FILE_PATH})
  target_link_libraries(${FILENAME} ${ALLLIBS})
endforeach(FILE_PATH)
