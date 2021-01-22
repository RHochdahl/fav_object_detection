#!/usr/bin/env python

import rospy
import numpy as np
from pyquaternion import Quaternion
import tf2_geometry_msgs
import tf
import threading
from hippocampus_common.tf_helper import TfHelper

from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance
from fav_object_detection.msg import ObjectDetectionArray, ObjectDetection


class ObjectDetectionNode():
    def __init__(self):
        self.lock = threading.Lock()

        rospy.init_node("object_detection")
        self.object_detection_pub = rospy.Publisher("objects",
                                                    ObjectDetectionArray,
                                                    queue_size=1)

        self.camera_name = rospy.get_param("~camera_name")
        vehicle_name = rospy.get_param("~vehicle_name")
        self.tf_helper = TfHelper(vehicle_name)

        self.tag_ids = range(63, 123)
        self.object_size = 0.12

        self.last_pose_time = 0

        self.position = np.zeros((3, 1)).reshape((-1, 1))
        self.rotation = Quaternion(x=0, y=0, z=0, w=1)
        self.frame = "map"

    def tag_callback(self, msg):
        if rospy.get_time() - self.last_pose_time > 0.5:
            return

        object_detections_msg = ObjectDetectionArray()
        object_detections_msg.header.stamp = rospy.Time.now()
        object_detections_msg.header.frame_id = self.frame
        for tag in msg.detections:
            if tag.id[0] in self.tag_ids:
                tag_rotation = Quaternion(x=tag.pose.pose.pose.orientation.x,
                                          y=tag.pose.pose.pose.orientation.y,
                                          z=tag.pose.pose.pose.orientation.z,
                                          w=tag.pose.pose.pose.orientation.w)

                object_position = tag_rotation.rotate(np.array([0.0, 0.0, -self.object_size/2.0]))

                object_pose = PoseWithCovariance()
                object_pose.pose.position.x = tag.pose.pose.pose.position.x + object_position[0]
                object_pose.pose.position.y = tag.pose.pose.pose.position.y + object_position[1]
                object_pose.pose.position.z = tag.pose.pose.pose.position.z + object_position[2]
                object_pose.pose.orientation = tag.pose.pose.pose.orientation

                transform = self.tf_helper.get_camera_frame_to_base_link_tf(self.camera_name)
                object_pose_in_base_link = tf2_geometry_msgs.do_transform_pose(object_pose, transform)

                object_position_in_base_link = np.array([
                    object_pose_in_base_link.pose.position.x,
                    object_pose_in_base_link.pose.position.y,
                    object_pose_in_base_link.pose.position.z
                ]).reshape((-1, 1))

                object_position_in_map = self.rotation.rotate(object_position_in_base_link).reshape((-1, 1)) + self.position

                quat = self.rotate_quaternion(self.rotation.elements, [object_pose_in_base_link.pose.orientation.w,
                                                                       object_pose_in_base_link.pose.orientation.x,
                                                                       object_pose_in_base_link.pose.orientation.y,
                                                                       object_pose_in_base_link.pose.orientation.z])

                e1, e2, e3 = tf.transformations.euler_from_quaternion([quat[1], quat[2], quat[3], quat[0]], axes='sxyz')
                e1 = (e1 + np.pi) % (np.pi/2)
                e2 = (e2 + np.pi) % (np.pi/2)
                e3 = (e3 + np.pi) % (np.pi/2)
                quat = tf.transformations.quaternion_from_euler(e1, e2, e3, axes='sxyz')
            
                object_pose_in_map = ObjectDetection()
                object_pose_in_map.pose.position.x = object_position_in_map[0]
                object_pose_in_map.pose.position.y = object_position_in_map[1]
                object_pose_in_map.pose.position.z = object_position_in_map[2]
                object_pose_in_map.pose.orientation.x = quat[0]
                object_pose_in_map.pose.orientation.y = quat[1]
                object_pose_in_map.pose.orientation.z = quat[2]
                object_pose_in_map.pose.orientation.w = quat[3]
                object_pose_in_map.size = self.object_size

                object_detections_msg.detections.append(object_pose_in_map)

        self.object_detection_pub.publish(object_detections_msg)

    def rotate_quaternion(self, q_1, q_2):
        (w_1, x_1, y_1, z_1) = q_1
        (w_2, x_2, y_2, z_2) = q_2
        quat_x = w_1 * z_2 + z_1 * w_2 + x_1 * y_2 - y_1 * x_2
        quat_y = w_1 * y_2 + y_1 * w_2 + z_1 * x_2 - x_1 * z_2
        quat_z = w_1 * x_2 + x_1 * w_2 + y_1 * z_2 - z_1 * y_2
        quat_w = w_1 * w_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
        return (quat_w, quat_x, quat_y, quat_z)

    def pose_callback(self, msg):
        self.position[0, 0] = msg.pose.pose.position.x
        self.position[1, 0] = msg.pose.pose.position.y
        self.position[2, 0] = msg.pose.pose.position.z
        self.rotation = Quaternion(x=msg.pose.pose.orientation.x,
                                   y=msg.pose.pose.orientation.y,
                                   z=msg.pose.pose.orientation.z,
                                   w=msg.pose.pose.orientation.w)
        self.frame = msg.header.frame_id
        self.last_pose_time = msg.header.stamp.to_sec()


def main():
    node = ObjectDetectionNode()
    rospy.Subscriber("ekf_pose",
                     PoseWithCovarianceStamped,
                     node.pose_callback,
                     queue_size=1)
    ns = rospy.get_param("~ns", "")
    rospy.Subscriber(ns + "/tag_detections",
                     AprilTagDetectionArray,
                     node.tag_callback,
                     queue_size=1)

    rospy.spin()


if __name__ == "__main__":
    main()
