cmake_minimum_required(VERSION 2.8.3)
project(sndt_rviz)

find_package(catkin REQUIRED
  COMPONENTS
    rviz
    sndt
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

link_directories(${catkin_LIBRARY_DIRS})

find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)

include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Gui_INCLUDE_DIRS})

find_package(Qt5Widgets REQUIRED)
if(${Qt5Widgets_VERSION} VERSION_LESS 5.7.0)
    message("QT less than 5.7")
else()
    message("Qt more than version 5.7, enabling c++11 support")
    set(CMAKE_CXX_STANDARD 11)
endif()

include_directories(include)

add_definitions(-DQT_NO_KEYWORDS)

qt5_wrap_cpp(MOC_FILES
  include/sndt_rviz/ndt_display.hpp
)

set(SOURCE_FILES
  src/ndt_display.cpp
  src/ndt_visual.cpp
  ${MOC_FILES}
)

add_library(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES} ${catkin_LIBRARIES})

catkin_package(
  DEPENDS  
    CATKIN_DEPENDS rviz sndt
    INCLUDE_DIRS include 
    LIBRARIES ${PROJECT_NAME}
)
