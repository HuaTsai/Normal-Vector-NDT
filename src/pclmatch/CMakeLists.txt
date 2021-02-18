cmake_minimum_required(VERSION 3.0.2)
project(pclmatch)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

# find_package(PCL 1.8.1 EXACT REQUIRED
find_package(PCL 1.11.1 EXACT REQUIRED
 COMPONENTS
  registration
  io
)

find_package(catkin REQUIRED
 COMPONENTS
  roscpp
  common
)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  "/home/ee904/Desktop/HuaTsai/NormalNDT/trimesh2/include"
)

link_directories(
  "/home/ee904/Desktop/HuaTsai/NormalNDT/trimesh2/lib.Linux64"
)

set(ALLLIBS ${catkin_LIBRARIES} ${PCL_LIBRARIES} trimesh config++ omp)

add_executable(pcl_version src/pcl_version.cc)

target_link_libraries(pcl_version ${ALLLIBS})

add_executable(main src/main.cc)

target_link_libraries(main ${ALLLIBS})

add_executable(test0207 src/test0207.cc)

target_link_libraries(test0207 ${ALLLIBS})

add_executable(test0209 src/test0209.cc)

target_link_libraries(test0209 ${ALLLIBS})

add_executable(test0218 src/test0218.cc)

target_link_libraries(test0218 ${ALLLIBS})
