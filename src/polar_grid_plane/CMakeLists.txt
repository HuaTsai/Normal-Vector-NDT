cmake_minimum_required(VERSION 2.8.3)
project(polar_grid_plane)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  pcl_ros
  pcl_conversions
  tf_conversions
  eigen_conversions
  common
)

catkin_package(CATKIN_DEPENDS
	sensor_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(Eigen3 REQUIRED)

#SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

link_directories(${PCL_LIBRARY_DIRS})

#Ground Plane Fitter
find_package(PCL 1.7 REQUIRED)
add_definitions(${PCL_DEFINITIONS})

add_executable(groundplanfit 
	nodes/ground_filter/groundplanfit.cpp
)

target_include_directories(groundplanfit PRIVATE
	${PCL_INCLUDE_DIRS}
)

target_link_libraries(groundplanfit
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
)

add_executable(groundfit 
	src/groundfit.cc
)

target_include_directories(groundfit PRIVATE
	${PCL_INCLUDE_DIRS}
)

target_link_libraries(groundfit
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
)
