cmake_minimum_required(VERSION 3.0.2)
project(trimesh2)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED
 COMPONENTS
  roscpp
)

file(GLOB_RECURSE ALL_LIBS "*.cc")

catkin_package(
 INCLUDE_DIRS
  include
 LIBRARIES
  trimesh2
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(trimesh2 ${ALL_LIBS})

# target_link_libraries(trimesh2 ${catkin_LIBRARIES})

