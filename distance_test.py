#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data


class FrontScan(Node):

    def __init__(self):
        super().__init__('front_scan')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            qos_profile_sensor_data
        )

        # 二维数组
        self.scan_matrix = []

        # 最多保存100帧
        self.max_scans = 100

        self.get_logger().info("Front LiDAR Scan Node Started")

    def listener_callback(self, msg):

        # 取 -15° 到 +15° 的雷达
        front_scan = msg.ranges[0:15] + msg.ranges[345:360]

        # 保存到二维数组
        self.scan_matrix.append(front_scan)

        # 防止无限增长
        if len(self.scan_matrix) > self.max_scans:
            self.scan_matrix.pop(0)

        # 打印当前一帧
        self.get_logger().info(f"Front scan: {front_scan}")

        # 如果你想访问二维数组
        # print(self.scan_matrix)


def main(args=None):
    rclpy.init(args=args)
    node = FrontScan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()