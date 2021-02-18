cmake_minimum_required(VERSION 3.0.2)
project(sndt)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(Ceres REQUIRED)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    pcl_ros
    sensor_msgs
    common
    message_generation
    visualization_msgs
    conti_radar
    tf2_sensor_msgs
    normal2d
)

add_message_files(
  FILES
    NDTCellMsg.msg
    NDTMapMsg.msg  
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    roscpp
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# Library

add_library(${PROJECT_NAME} src/dummy.cc)

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_generate_messages_cpp)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

# Executables

set(ALLLIBS ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PROJECT_NAME})

add_executable(main src/main.cc)
target_link_libraries(main ${ALLLIBS})

add_executable(ceres_test src/ceres_test.cc)
target_link_libraries(ceres_test ${ALLLIBS} )

add_executable(ego src/ego.cc)
target_link_libraries(ego ${ALLLIBS})

add_executable(savepc src/savepc.cc)
target_link_libraries(savepc ${ALLLIBS})

add_executable(pairmatch src/pairmatch.cc)
target_link_libraries(pairmatch ${ALLLIBS})

add_executable(pcl_version src/pcl_version.cc)
target_link_libraries(pcl_version ${ALLLIBS})

add_executable(test0128 src/test0128.cc)
target_link_libraries(test0128 ${ALLLIBS})

add_executable(test0129 src/test0129.cc)
target_link_libraries(test0129 ${ALLLIBS})
