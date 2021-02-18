cmake_minimum_required(VERSION 2.8.3)
project(common)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    tf2_eigen
    gsl
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
    tf2_eigen
    gsl
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
 
add_executable(dummy src/dummy.cc)
  
target_link_libraries(dummy ${catkin_LIBRARIES})
