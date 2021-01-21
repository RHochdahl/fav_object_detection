cmake_minimum_required(VERSION 3.0.2)
project(fav_object_detection)

find_package(catkin REQUIRED COMPONENTS
  apriltag_ros
  geometry_msgs
)

catkin_package()
