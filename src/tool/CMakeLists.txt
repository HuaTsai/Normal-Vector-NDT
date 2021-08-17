cmake_minimum_required(VERSION 3.0.2)
project(tool)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

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

add_executable(pathtotext src/pathtotext.cc)
target_link_libraries(pathtotext ${catkin_LIBRARIES})

add_executable(getgtpath src/getgtpath.cc)
target_link_libraries(getgtpath ${catkin_LIBRARIES})

add_executable(getimages src/getimages.cc)
target_link_libraries(getimages ${catkin_LIBRARIES})

add_executable(getlidars src/getlidars.cc)
target_link_libraries(getlidars ${catkin_LIBRARIES})

add_executable(seetfs src/seetfs.cc)
target_link_libraries(seetfs ${catkin_LIBRARIES})

add_executable(seelidar src/seelidar.cc)
target_link_libraries(seelidar ${catkin_LIBRARIES})
