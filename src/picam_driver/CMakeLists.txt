cmake_minimum_required(VERSION 2.8.3)
project(picam_driver)
add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  image_transport
  camera_info_manager
  sensor_msgs
  nodelet
  dynamic_reconfigure)


find_package(OpenCV)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

add_executable(picam_driver src/driver.cpp)
target_link_libraries(picam_driver ${catkin_LIBRARIES})
