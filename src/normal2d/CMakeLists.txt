cmake_minimum_required(VERSION 3.0.2)
project(normal2d)

set(CMAKE_CXX_FLAGS "-Wall -Wextra --coverage")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP REQUIRED)

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

add_library(${PROJECT_NAME} src/normal2d.cc)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(nmtest src/normal2dtest.cc)
target_link_libraries(nmtest ${PROJECT_NAME} pcl_visualization)
