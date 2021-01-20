#!/usr/bin/env python

import rospy
import numpy as np

from apriltag_ros.msg import AprilTagDetectionArray
from apriltag_ros.msg import AprilTagDetection


def callback(msg, tmp_list):
    [tag_ids, object_detection_pub] = tmp_list
    object_detections_msg = AprilTagDetectionArray()
    object_detections_msg.header.stamp = rospy.Time.now()
    object_detections_msg.header.frame_id = msg.header.frame_id
    for detection in msg.detections:
        if detection.id[0] in tag_ids:
            object_detections_msg.detections.append(detection)
    object_detection_pub.publish(object_detections_msg)


def main():
    rospy.init_node("object_detection")
    object_detection_pub = rospy.Publisher("objects",
                                           AprilTagDetectionArray,
                                           queue_size=1)

    camera_name = rospy.get_param("~camera_name")
    tag_ids = range(63, 123)

    rospy.Subscriber("tag_detections_" + camera_name,
                     AprilTagDetectionArray,
                     callback, [tag_ids, object_detection_pub],
                     queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
