cmake_minimum_required(VERSION 3.0.2)
project(ndt_exec)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    ndt
    pcl_ros
    metric
    normal2d
)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

set(ALLLIBS ${catkin_LIBRARIES})

file(GLOB_RECURSE ALL_SRCS "*.cc")
foreach(FILE_PATH ${ALL_SRCS})
  get_filename_component(FILENAME ${FILE_PATH} NAME_WE)
  add_executable(${FILENAME} ${FILE_PATH})
  target_link_libraries(${FILENAME} ${ALLLIBS})
endforeach(FILE_PATH)
