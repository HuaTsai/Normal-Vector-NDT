cmake_minimum_required(VERSION 3.0.2)
project(normal2d)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED
  COMPONENTS
    pcl_ros
)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} src/normal2d.cc src/pca2d.cc)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(normal2dtest src/normal2dtest.cc)
target_link_libraries(normal2dtest ${PROJECT_NAME} pcl_visualization)
