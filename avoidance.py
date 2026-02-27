#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import time
import numpy as np
import random


class SmoothAvoidance(Node):
    def __init__(self):
        super().__init__('smooth_avoidance')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            qos_profile_sensor_data
        )

        # 状态机
        self.state = "GO"
        self.state_until = 0.0

        # 连续卡住计数
        self.stuck_counter = 0

        self.cmd = Twist()

        self.get_logger().info("Smooth Avoidance + Random Escape Running...")

    def smooth(self, old, new, alpha=0.6):
        return old * alpha + new * (1 - alpha)

    def listener_callback(self, msg):
        now = time.monotonic()

        # 状态未结束 → 执行同一个动作（抗抖动）
        if now < self.state_until:
            self.publisher.publish(self.cmd)
            return

        # 分扇区
        f_vals = msg.ranges[0:20] + msg.ranges[340:360]
        l_vals = msg.ranges[20:60]
        r_vals = msg.ranges[300:340]

        f_dist = min([x for x in f_vals if x > 0.05] or [10])
        l_dist = min([x for x in l_vals if x > 0.05] or [10])
        r_dist = min([x for x in r_vals if x > 0.05] or [10])

        # 平滑滤波（防抽搐）
        f_dist = self.smooth(getattr(self, 'f_last', f_dist), f_dist)
        l_dist = self.smooth(getattr(self, 'l_last', l_dist), l_dist)
        r_dist = self.smooth(getattr(self, 'r_last', r_dist), r_dist)

        self.f_last, self.l_last, self.r_last = f_dist, l_dist, r_dist

        # -----------------------------
        #     随机逃逸机制触发逻辑
        # -----------------------------
        if f_dist < 0.35:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        if self.stuck_counter > 6:
            self.get_logger().warn("STUCK detected! Triggering Random Escape...")

            # 随机倒车
            self.cmd.linear.x = -0.10
            self.cmd.angular.z = 0.0
            self.state = "BACK"
            self.state_until = now + 0.15
            self.publisher.publish(self.cmd)

            # 倒车后改成随机转向逃逸
            random_turn_dir = random.choice([-1.0, 1.0])
            random_turn_time = random.uniform(0.3, 0.6)

            self.cmd.linear.x = 0.0
            self.cmd.angular.z = random_turn_dir * 0.55

            self.state = "ESCAPE"
            self.state_until = now + random_turn_time
            self.get_logger().info(f"ESCAPING: turn_dir={random_turn_dir}, time={random_turn_time:.2f}")

            # 重置计数器
            self.stuck_counter = 0
            return

        # -------------------------------------
        #    原有逻辑（保留你的行为风格）
        # -------------------------------------

        # 1）前方危险 → 停 + 原地调整方向
        if f_dist < 0.50:
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.45 if l_dist > r_dist else -0.45
            self.state = "TURN"
            self.state_until = now + 0.35
            self.get_logger().info("Emergency Stop → Turning")

        # 2）左侧靠近 → 微右调
        elif l_dist < 0.35:
            self.cmd.linear.x = 0.1
            self.cmd.angular.z = -0.25
            self.state = "GO"
            self.state_until = now + 0.15
            self.get_logger().info("Left wall → slight right")

        # 3）右侧靠近 → 微左调
        elif r_dist < 0.35:
            self.cmd.linear.x = 0.1
            self.cmd.angular.z = 0.25
            self.state = "GO"
            self.state_until = now + 0.15
            self.get_logger().info("Right wall → slight left")

        else:
            # 4）安全区域 → 前进
            self.cmd.linear.x = 0.20
            self.cmd.angular.z = 0.0
            self.state = "GO"
            self.state_until = now + 0.15

        self.publisher.publish(self.cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SmoothAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()