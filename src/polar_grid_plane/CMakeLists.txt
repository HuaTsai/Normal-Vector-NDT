cmake_minimum_required(VERSION 2.8.3)
project(polar_grid_plane)

set(CMAKE_CXX_FLAGS "-Wall -Wextra --coverage")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    pcl_ros
    common
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(groundfitnode src/groundfitnode.cc)
target_link_libraries(groundfitnode ${catkin_LIBRARIES})

add_executable(groundfit src/groundfit.cc)
target_link_libraries(groundfit ${catkin_LIBRARIES})
