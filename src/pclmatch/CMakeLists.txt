cmake_minimum_required(VERSION 3.0.2)
project(pclmatch)

set(CMAKE_CXX_FLAGS "-Wall --coverage")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenMP REQUIRED)
find_package(PCL 1.11.1 EXACT REQUIRED
 COMPONENTS
  registration
  io
)
 
find_package(catkin REQUIRED
  COMPONENTS
  roscpp
  common
  trimesh2
)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

set(ALLLIBS ${catkin_LIBRARIES} ${PCL_LIBRARIES} OpenMP::OpenMP_CXX config++)

file(GLOB_RECURSE ALL_SRCS "*.cc")
foreach(FILE_PATH ${ALL_SRCS})
  get_filename_component(FILENAME ${FILE_PATH} NAME_WE)
  add_executable(${FILENAME} ${FILE_PATH})
  target_link_libraries(${FILENAME} ${ALLLIBS})
endforeach(FILE_PATH)
