cmake_minimum_required(VERSION 3.0.2)
project(normal2d)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)  # Release infers BUG

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package (catkin REQUIRED)

catkin_package(
  DEPENDS
    EIGEN3
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
)

include_directories(include)

# PCL Package ##############################################################
find_package(PCL 1.8.1 EXACT REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Boost Package ############################################################
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.58.0 COMPONENTS program_options filesystem REQUIRED)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

add_library(${PROJECT_NAME} src/Normal2dEstimation.cpp src/PCA2D.cpp)

#PointXYNormal.cpp PointXYNormal.h
add_executable(norm_2d main.cpp)
target_link_libraries(norm_2d ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${PROJECT_NAME})
