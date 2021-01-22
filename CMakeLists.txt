cmake_minimum_required(VERSION 3.0.2)
project(fav_object_detection)

find_package(catkin REQUIRED COMPONENTS
  apriltag_ros
  message_generation
  geometry_msgs
  std_msgs
)

add_message_files(
  FILES
  ObjectDetectionArray.msg
  ObjectDetection.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)
