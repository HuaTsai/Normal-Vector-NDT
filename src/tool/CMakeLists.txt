cmake_minimum_required(VERSION 3.0.2)
project(tool)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    pcl_ros
    sensor_msgs
    common
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(bagtoply src/bagtoply.cc)
target_link_libraries(bagtoply ${catkin_LIBRARIES})

add_executable(showpath src/showpath.cc)
target_link_libraries(showpath ${catkin_LIBRARIES})

add_executable(pathtotext src/pathtotext.cc)
target_link_libraries(pathtotext ${catkin_LIBRARIES})

add_executable(gtpath src/gtpath.cc)
target_link_libraries(gtpath ${catkin_LIBRARIES})

add_executable(gtdis src/gtdis.cc)
target_link_libraries(gtdis ${catkin_LIBRARIES})
