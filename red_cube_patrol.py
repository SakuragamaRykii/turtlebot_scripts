#!/usr/bin/env python3
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan


class RedCubePatrol(Node):
    def __init__(self):
        super().__init__('red_cube_patrol')

        # =========================
        # 可按实际情况微调的参数
        # =========================
        self.LINEAR_SPEED = 0.08             # 所有前进统一速度
        self.BACKWARD_SPEED = -0.08          # 返回中点时倒车速度
        self.SEARCH_ANGULAR_SPEED = 0.35     # 原地搜索转速
        self.ALIGN_ANGULAR_SPEED = 0.22      # 对准方块时转速
        self.TURN_LEFT_SPEED = 0.30          # 巡逻左转转速

        self.WALL_STOP_DIST = 0.42           # 前方离墙小于该值时停止
        self.MIDPOINT_TOL = 0.06             # 返回中点的容差
        self.ALIGN_PIXEL_TOL = 25            # 图像中心允许误差
        self.APPROACH_PIXEL_TOL = 45         # 前进时允许的小角度修正误差
        self.RED_MIN_AREA = 1200             # 红色目标最小面积阈值
        self.LOST_TARGET_FRAMES = 3          # 连续丢失多少帧后认定“完全看不见”
        self.TURN_DONE_TOL = math.radians(4) # 转角完成容差

        # =========================
        # ROS2 通信
        # =========================
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.control_timer = self.create_timer(0.10, self.control_loop)   # 10 Hz
        self.status_timer = self.create_timer(1.00, self.status_loop)     # 1 Hz

        # =========================
        # 数据变量
        # =========================
        self.bridge = CvBridge()

        self.has_scan = False
        self.has_odom = False
        self.has_image = False

        self.front_dist = float('inf')

        self.world_x = 0.0
        self.world_y = 0.0
        self.world_yaw = 0.0

        self.init_world_x = None
        self.init_world_y = None
        self.init_world_yaw = None

        self.local_x = 0.0
        self.local_y = 0.0
        self.local_yaw = 0.0

        self.image_width = None
        self.red_visible = False
        self.red_cx = None
        self.red_area = 0.0
        self.lost_count = 0

        # =========================
        # 状态机变量
        # =========================
        self.state = 'WAIT_FOR_DATA'
        self.last_state = None
        self.state_enter_time = time.monotonic()

        self.search_prev_yaw = None
        self.search_accum_yaw = 0.0

        self.turn_prev_yaw = None
        self.turn_accum_yaw = 0.0

        self.segment_start = np.array([0.0, 0.0], dtype=float)
        self.segment_mid = np.array([0.0, 0.0], dtype=float)
        self.wall_stop_point = np.array([0.0, 0.0], dtype=float)

        print("[BOOT] Node started. Waiting for camera, odom and lidar data...")

    # =========================
    # 工具函数
    # =========================
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def stop_robot(self):
        try:
            msg = Twist()
            self.cmd_pub.publish(msg)
        except Exception:
            pass

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def current_pos(self):
        return np.array([self.local_x, self.local_y], dtype=float)

    def set_state(self, new_state, event_text=None):
        if self.state != new_state:
            self.last_state = self.state
            self.state = new_state
            self.state_enter_time = time.monotonic()
            if event_text:
                print(f"[EVENT] {event_text}")
            else:
                print(f"[EVENT] State changed to {new_state}")

            if new_state == 'SEARCH_ROTATE':
                self.search_prev_yaw = self.local_yaw
                self.search_accum_yaw = 0.0

            if new_state == 'TURN_LEFT_90':
                self.turn_prev_yaw = self.local_yaw
                self.turn_accum_yaw = 0.0

    def ready(self):
        return self.has_scan and self.has_odom and self.has_image

    # =========================
    # 回调函数
    # =========================
    def scan_callback(self, msg):
        # 参考你给的 distance_test.py 和 avoidance.py：
        # 取前方扇区，并过滤掉无效值，得到最小前向距离
        front_scan = list(msg.ranges[0:15]) + list(msg.ranges[345:360])
        valid = [x for x in front_scan if math.isfinite(x) and x > 0.05]
        raw_front_dist = min(valid) if valid else float('inf')

        # 简单平滑，避免数值抖动
        if not self.has_scan:
            self.front_dist = raw_front_dist
        else:
            self.front_dist = 0.7 * self.front_dist + 0.3 * raw_front_dist

        self.has_scan = True

    def odom_callback(self, msg):
        self.world_x = msg.pose.pose.position.x
        self.world_y = msg.pose.pose.position.y
        self.world_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        if self.init_world_x is None:
            # 以启动时的朝向和位置作为局部平面坐标系原点
            self.init_world_x = self.world_x
            self.init_world_y = self.world_y
            self.init_world_yaw = self.world_yaw
            print("[EVENT] Local 2D frame initialized at robot start pose.")

        dx = self.world_x - self.init_world_x
        dy = self.world_y - self.init_world_y
        c = math.cos(-self.init_world_yaw)
        s = math.sin(-self.init_world_yaw)

        # 将世界坐标转换到“起点为原点”的局部二维坐标系
        self.local_x = c * dx - s * dy
        self.local_y = s * dx + c * dy
        self.local_yaw = self.normalize_angle(self.world_yaw - self.init_world_yaw)

        self.has_odom = True

    def image_callback(self, msg):
        self.has_image = True

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.red_visible = False
            print(f"[WARN] Camera conversion failed: {e}")
            return

        self.image_width = frame.shape[1]

        # BGR -> HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 红色有两个 hue 区间
        lower_red_1 = np.array([0, 100, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 100, 70])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area = 0.0
        best_cx = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.RED_MIN_AREA:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])

            if area > best_area:
                best_area = area
                best_cx = cx

        if best_cx is not None:
            self.red_visible = True
            self.red_cx = best_cx
            self.red_area = best_area
        else:
            self.red_visible = False
            self.red_cx = None
            self.red_area = 0.0

    # =========================
    # 状态行为函数
    # =========================
    def handle_search_rotate(self):
        # 原地 360° 搜索红色方块
        if self.red_visible:
            self.stop_robot()
            self.set_state('ALIGN_TO_TARGET', 'Red target detected during 360 scan.')
            return

        delta = self.normalize_angle(self.local_yaw - self.search_prev_yaw)
        self.search_accum_yaw += abs(delta)
        self.search_prev_yaw = self.local_yaw

        if self.search_accum_yaw >= (2.0 * math.pi - self.TURN_DONE_TOL):
            self.stop_robot()
            self.segment_start = self.current_pos().copy()
            self.set_state('PATROL_FORWARD', 'No red target found in full rotation. Start patrol forward.')
            return

        self.publish_cmd(0.0, self.SEARCH_ANGULAR_SPEED)

    def handle_align_to_target(self):
        if not self.red_visible or self.image_width is None:
            self.stop_robot()
            self.set_state('SEARCH_ROTATE', 'Target lost while aligning. Restart 360 scan.')
            return

        center_x = self.image_width / 2.0
        error = self.red_cx - center_x

        if abs(error) <= self.ALIGN_PIXEL_TOL:
            self.stop_robot()
            self.lost_count = 0
            self.set_state('APPROACH_TARGET', 'Target aligned. Start constant-speed approach.')
            return

        # 目标在画面左边 -> 左转；目标在右边 -> 右转
        if error < 0:
            self.publish_cmd(0.0, self.ALIGN_ANGULAR_SPEED)
        else:
            self.publish_cmd(0.0, -self.ALIGN_ANGULAR_SPEED)

    def handle_approach_target(self):
        # 靠近方块时，前进速度始终固定
        if not self.red_visible:
            self.lost_count += 1
            if self.lost_count >= self.LOST_TARGET_FRAMES:
                self.stop_robot()
                self.set_state('TARGET_LOST_STOP', 'Target fully lost. Robot stopped for pickup placeholder.')
            return

        self.lost_count = 0

        center_x = self.image_width / 2.0
        error = self.red_cx - center_x

        # 允许小幅度角度修正，但前进速度始终不变
        angular = 0.0
        if abs(error) > self.APPROACH_PIXEL_TOL:
            angular = self.ALIGN_ANGULAR_SPEED if error < 0 else -self.ALIGN_ANGULAR_SPEED

        self.publish_cmd(self.LINEAR_SPEED, angular)

    def handle_target_lost_stop(self):
        # 这里留给你后续插入机械臂抓取代码
        # 现在按你的要求：一旦完全看不见方块，立刻停下并留空
        self.stop_robot()

    def handle_patrol_forward(self):
        # 巡逻过程中，一旦看见红色方块，优先追踪
        if self.red_visible:
            self.stop_robot()
            self.set_state('ALIGN_TO_TARGET', 'Red target detected during patrol.')
            return

        # 前方接近墙面 -> 立即停下，记录终点，计算中点
        if self.front_dist <= self.WALL_STOP_DIST:
            self.stop_robot()
            self.wall_stop_point = self.current_pos().copy()
            self.segment_mid = 0.5 * (self.segment_start + self.wall_stop_point)
            self.set_state('RETURN_TO_MIDPOINT', 'Wall detected. Stop and return to segment midpoint.')
            return

        self.publish_cmd(self.LINEAR_SPEED, 0.0)

    def handle_return_to_midpoint(self):
        # 返回当前这条直线段的中点
        dist_to_mid = np.linalg.norm(self.current_pos() - self.segment_mid)

        if dist_to_mid <= self.MIDPOINT_TOL:
            self.stop_robot()
            self.set_state('TURN_LEFT_90', 'Reached midpoint. Start 90-degree left turn.')
            return

        self.publish_cmd(self.BACKWARD_SPEED, 0.0)

    def handle_turn_left_90(self):
        # 转弯过程中如果看到目标，也优先处理目标
        if self.red_visible:
            self.stop_robot()
            self.set_state('ALIGN_TO_TARGET', 'Red target detected during left turn.')
            return

        delta = self.normalize_angle(self.local_yaw - self.turn_prev_yaw)
        self.turn_accum_yaw += delta
        self.turn_prev_yaw = self.local_yaw

        if self.turn_accum_yaw >= (math.pi / 2.0 - self.TURN_DONE_TOL):
            self.stop_robot()
            self.segment_start = self.current_pos().copy()
            self.set_state('PATROL_FORWARD', '90-degree left turn completed. Continue patrol.')
            return

        self.publish_cmd(0.0, self.TURN_LEFT_SPEED)

    # =========================
    # 主循环
    # =========================
    def control_loop(self):
        if not self.ready():
            self.stop_robot()
            return

        if self.state == 'WAIT_FOR_DATA':
            self.stop_robot()
            self.segment_start = self.current_pos().copy()
            self.set_state('SEARCH_ROTATE', 'Camera connected. Odom and lidar ready. Start 360 scan.')
            return

        if self.state == 'SEARCH_ROTATE':
            self.handle_search_rotate()
        elif self.state == 'ALIGN_TO_TARGET':
            self.handle_align_to_target()
        elif self.state == 'APPROACH_TARGET':
            self.handle_approach_target()
        elif self.state == 'TARGET_LOST_STOP':
            self.handle_target_lost_stop()
        elif self.state == 'PATROL_FORWARD':
            self.handle_patrol_forward()
        elif self.state == 'RETURN_TO_MIDPOINT':
            self.handle_return_to_midpoint()
        elif self.state == 'TURN_LEFT_90':
            self.handle_turn_left_90()
        else:
            self.stop_robot()
            print(f"[WARN] Unknown state: {self.state}")

    def status_loop(self):
        print(
            f"[STATUS] ready={self.ready()} "
            f"has_scan={self.has_scan} "
            f"has_odom={self.has_odom} "
            f"has_image={self.has_image} "
            f"state={self.state} "
            f"pos=[{self.local_x:.2f}, {self.local_y:.2f}] "
            f"yaw_deg={math.degrees(self.local_yaw):.1f} "
            f"front_dist={self.front_dist:.2f} "
            f"red_visible={self.red_visible}"
        )

        print(
            f"[STATUS] "
            f"state={self.state} "
            f"pos=[{self.local_x:.2f}, {self.local_y:.2f}] "
            f"yaw_deg={math.degrees(self.local_yaw):.1f} "
            f"front_dist={self.front_dist:.2f} "
            f"red_visible={self.red_visible} "
            f"red_area={self.red_area:.0f}"
        )

    def destroy_node(self):
        self.stop_robot()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RedCubePatrol()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("[EVENT] Keyboard interrupt received. Stopping robot.")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()