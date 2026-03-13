#!/usr/bin/env python3
import math
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan


class RedCubePatrol(Node):
    def __init__(self):
        super().__init__('red_cube_patrol_v3')

        # =========================
        # 可调参数（真机优先稳定）
        # =========================
        self.LINEAR_SPEED = 0.05
        self.BACKWARD_SPEED = -0.04

        self.SEARCH_MAX_ANG = 0.22
        self.SEARCH_MIN_ANG = 0.08
        self.ALIGN_MAX_ANG = 0.18
        self.ALIGN_MIN_ANG = 0.04
        self.TURN_MAX_ANG = 0.18
        self.TURN_MIN_ANG = 0.05

        self.WALL_STOP_DIST = 0.45
        self.APPROACH_STOP_DIST = 0.11
        self.EMERGENCY_WALL_DIST = 0.08

        self.ALIGN_PIXEL_TOL = 18
        self.REACQUIRE_PIXEL_TOL = 95
        self.RED_MIN_AREA = 700
        self.LOST_TARGET_FRAMES = 3
        self.CENTER_HOLD_FRAMES = 2

        self.SEARCH_DONE_TOL = math.radians(3)
        self.TURN_YAW_TOL = math.radians(3)
        self.MIDPOINT_TOL = 0.03

        self.APPROACH_MIN_TIME = 0.40
        self.APPROACH_MIN_TRAVEL = 0.04
        self.TARGET_HOLD_SECONDS = 1.20
        self.RESTART_PAUSE_SECONDS = 0.60

        self.CONTROL_DT = 0.05
        self.STATUS_DT = 1.00

        # =========================
        # ROS2 通信
        # =========================
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.control_timer = self.create_timer(self.CONTROL_DT, self.control_loop)
        self.status_timer = self.create_timer(self.STATUS_DT, self.status_loop)

        # =========================
        # 数据变量
        # =========================
        self.has_scan = False
        self.has_odom = False
        self.has_image = False

        self.front_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')

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
        self.red_seen_frames = 0
        self.red_lost_frames = 0
        self.center_hold_count = 0

        # =========================
        # 状态机变量
        # =========================
        self.state = 'WAIT_FOR_DATA'
        self.last_state = None
        self.state_enter_time = time.monotonic()

        self.search_prev_yaw = None
        self.search_accum_yaw = 0.0

        self.turn_target_yaw = 0.0

        self.segment_start = np.array([0.0, 0.0], dtype=float)
        self.segment_heading_yaw = 0.0
        self.segment_length = 0.0
        self.segment_mid_progress = 0.0

        self.approach_start_pos = np.array([0.0, 0.0], dtype=float)
        self.target_hold_reason = 'none'
        self.resume_time = 0.0

        print('[BOOT] Node started. Waiting for camera, odom and lidar data...')

    # =========================
    # 工具函数
    # =========================
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def clamp_abs(self, value, min_abs, max_abs):
        if abs(value) < 1e-9:
            return 0.0
        sign = 1.0 if value > 0.0 else -1.0
        mag = min(max(abs(value), min_abs), max_abs)
        return sign * mag

    def stop_robot(self):
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def current_pos(self):
        return np.array([self.local_x, self.local_y], dtype=float)

    def ready(self):
        return self.has_scan and self.has_odom and self.has_image

    def state_age(self):
        return time.monotonic() - self.state_enter_time

    def target_confirmed(self):
        return self.red_visible and self.red_area >= self.RED_MIN_AREA and self.red_seen_frames >= 2

    def current_progress_along_segment(self):
        direction = np.array([
            math.cos(self.segment_heading_yaw),
            math.sin(self.segment_heading_yaw),
        ])
        return float(np.dot(self.current_pos() - self.segment_start, direction))

    def begin_new_segment_here(self):
        self.segment_start = self.current_pos().copy()
        self.segment_heading_yaw = self.local_yaw
        self.segment_length = 0.0
        self.segment_mid_progress = 0.0

    def set_state(self, new_state, event_text=None):
        if self.state == new_state:
            return

        self.last_state = self.state
        self.state = new_state
        self.state_enter_time = time.monotonic()

        if event_text:
            print(f'[EVENT] {event_text}')
        else:
            print(f'[EVENT] State changed to {new_state}')

        if new_state == 'SEARCH_ROTATE':
            self.search_prev_yaw = self.local_yaw
            self.search_accum_yaw = 0.0
            self.center_hold_count = 0

        elif new_state == 'ALIGN_TO_TARGET':
            self.center_hold_count = 0
            self.red_lost_frames = 0

        elif new_state == 'APPROACH_TARGET':
            self.red_lost_frames = 0
            self.center_hold_count = 0
            self.approach_start_pos = self.current_pos().copy()

        elif new_state == 'PATROL_FORWARD':
            self.begin_new_segment_here()

        elif new_state == 'TURN_LEFT_90':
            self.turn_target_yaw = self.normalize_angle(self.local_yaw + math.pi / 2.0)

    # =========================
    # 回调函数
    # =========================
    def scan_callback(self, msg):
        front_vals = list(msg.ranges[0:20]) + list(msg.ranges[340:360])
        left_vals = list(msg.ranges[20:60])
        right_vals = list(msg.ranges[300:340])

        def valid_min(vals):
            good = [x for x in vals if math.isfinite(x) and x > 0.05]
            return min(good) if good else float('inf')

        raw_front = valid_min(front_vals)
        raw_left = valid_min(left_vals)
        raw_right = valid_min(right_vals)

        if not self.has_scan:
            self.front_dist = raw_front
            self.left_dist = raw_left
            self.right_dist = raw_right
        else:
            alpha = 0.60
            self.front_dist = alpha * self.front_dist + (1.0 - alpha) * raw_front
            self.left_dist = alpha * self.left_dist + (1.0 - alpha) * raw_left
            self.right_dist = alpha * self.right_dist + (1.0 - alpha) * raw_right

        self.has_scan = True

    def odom_callback(self, msg):
        self.world_x = msg.pose.pose.position.x
        self.world_y = msg.pose.pose.position.y
        self.world_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        if self.init_world_x is None:
            self.init_world_x = self.world_x
            self.init_world_y = self.world_y
            self.init_world_yaw = self.world_yaw
            print('[EVENT] Local 2D frame initialized at robot start pose.')

        dx = self.world_x - self.init_world_x
        dy = self.world_y - self.init_world_y
        c = math.cos(-self.init_world_yaw)
        s = math.sin(-self.init_world_yaw)

        self.local_x = c * dx - s * dy
        self.local_y = s * dx + c * dy
        self.local_yaw = self.normalize_angle(self.world_yaw - self.init_world_yaw)
        self.has_odom = True

    def image_callback(self, msg):
        self.has_image = True

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('decoded frame is None')
        except Exception as e:
            self.red_visible = False
            print(f'[WARN] Camera conversion failed: {e}')
            return

        self.image_width = frame.shape[1]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 95, 60])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 95, 60])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = cv2.bitwise_or(mask1, mask2)

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

            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
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
            self.red_seen_frames += 1
            self.red_lost_frames = 0
        else:
            self.red_visible = False
            self.red_cx = None
            self.red_area = 0.0
            self.red_seen_frames = 0
            self.red_lost_frames += 1
            self.center_hold_count = 0

    # =========================
    # 状态行为函数
    # =========================
    def handle_search_rotate(self):
        if self.target_confirmed():
            self.stop_robot()
            self.set_state('ALIGN_TO_TARGET', 'Red target detected during 360 scan.')
            return

        delta = self.normalize_angle(self.local_yaw - self.search_prev_yaw)
        if delta > 0.0:
            self.search_accum_yaw += delta
        self.search_prev_yaw = self.local_yaw

        remaining = 2.0 * math.pi - self.search_accum_yaw
        if remaining <= self.SEARCH_DONE_TOL:
            self.stop_robot()
            self.set_state('PATROL_FORWARD', 'No red target found in full rotation. Start patrol forward.')
            return

        angular = min(self.SEARCH_MAX_ANG, 0.75 * remaining)
        angular = max(self.SEARCH_MIN_ANG, angular)
        self.publish_cmd(0.0, angular)

    def handle_align_to_target(self):
        if not self.red_visible or self.image_width is None:
            if self.red_lost_frames >= self.LOST_TARGET_FRAMES:
                self.stop_robot()
                self.set_state('SEARCH_ROTATE', 'Target lost while aligning. Restart 360 scan.')
            return

        center_x = self.image_width / 2.0
        error = self.red_cx - center_x
        error_norm = error / center_x

        if abs(error) <= self.ALIGN_PIXEL_TOL:
            self.center_hold_count += 1
            self.stop_robot()
            if self.center_hold_count >= self.CENTER_HOLD_FRAMES:
                self.center_hold_count = 0
                self.set_state('APPROACH_TARGET', 'Target aligned. Start approach.')
            return

        self.center_hold_count = 0
        angular = self.clamp_abs(-0.24 * error_norm, self.ALIGN_MIN_ANG, self.ALIGN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_approach_target(self):
        # 保底紧急停止
        if self.front_dist <= self.EMERGENCY_WALL_DIST:
            self.stop_robot()
            self.target_hold_reason = 'emergency_front_stop'
            self.set_state('TARGET_HOLD', 'Emergency front stop triggered.')
            return

        # 目标完全丢失则停下
        if not self.red_visible or self.image_width is None:
            if self.red_lost_frames >= self.LOST_TARGET_FRAMES:
                self.stop_robot()
                self.target_hold_reason = 'target_lost'
                self.set_state('TARGET_HOLD', 'Target lost. Hold position.')
            return

        center_x = self.image_width / 2.0
        error = self.red_cx - center_x
        error_norm = error / center_x

        # 偏差太大先重新对准
        if abs(error) >= self.REACQUIRE_PIXEL_TOL:
            self.stop_robot()
            self.set_state('ALIGN_TO_TARGET', 'Target offset too large. Re-align first.')
            return

        # 只有已经真正前进了一小段之后，才启用近距离停止，避免“刚对准就不动”
        travel = float(np.linalg.norm(self.current_pos() - self.approach_start_pos))
        if self.state_age() >= self.APPROACH_MIN_TIME and travel >= self.APPROACH_MIN_TRAVEL:
            if self.front_dist <= self.APPROACH_STOP_DIST:
                self.stop_robot()
                self.target_hold_reason = 'target_close_enough'
                self.set_state('TARGET_HOLD', 'Target close enough. Hold for pickup placeholder.')
                return

        angular = 0.0
        if abs(error) > self.ALIGN_PIXEL_TOL:
            angular = self.clamp_abs(-0.18 * error_norm, 0.02, 0.10)

        self.publish_cmd(self.LINEAR_SPEED, angular)

    def handle_target_hold(self):
        # 这里本来给机械臂抓取使用
        # 现在为了方便重复测试，不永久卡死：停留一小段时间后自动重新搜索
        self.stop_robot()
        if self.state_age() >= self.TARGET_HOLD_SECONDS:
            self.resume_time = time.monotonic() + self.RESTART_PAUSE_SECONDS
            self.set_state('SEARCH_ROTATE', f'Hold finished ({self.target_hold_reason}). Resume search.')

    def handle_patrol_forward(self):
        if self.target_confirmed():
            self.stop_robot()
            self.set_state('ALIGN_TO_TARGET', 'Red target detected during patrol.')
            return

        progress = self.current_progress_along_segment()

        if self.front_dist <= self.WALL_STOP_DIST:
            self.stop_robot()
            self.segment_length = max(0.0, progress)
            self.segment_mid_progress = 0.5 * self.segment_length
            self.set_state('RETURN_TO_MIDPOINT', 'Wall detected. Stop and return to segment midpoint.')
            return

        self.publish_cmd(self.LINEAR_SPEED, 0.0)

    def handle_return_to_midpoint(self):
        if self.target_confirmed():
            self.stop_robot()
            self.set_state('ALIGN_TO_TARGET', 'Red target detected while returning to midpoint.')
            return

        progress = self.current_progress_along_segment()
        if progress <= self.segment_mid_progress + self.MIDPOINT_TOL:
            self.stop_robot()
            self.set_state('TURN_LEFT_90', 'Reached midpoint. Start 90-degree left turn.')
            return

        self.publish_cmd(self.BACKWARD_SPEED, 0.0)

    def handle_turn_left_90(self):
        if self.target_confirmed():
            self.stop_robot()
            self.set_state('ALIGN_TO_TARGET', 'Red target detected during left turn.')
            return

        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= self.TURN_YAW_TOL:
            self.stop_robot()
            self.set_state('PATROL_FORWARD', '90-degree left turn completed. Continue patrol.')
            return

        angular = self.clamp_abs(0.85 * error, self.TURN_MIN_ANG, self.TURN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    # =========================
    # 主循环
    # =========================
    def control_loop(self):
        if not self.ready():
            self.stop_robot()
            return

        if self.state == 'WAIT_FOR_DATA':
            self.stop_robot()
            self.set_state('SEARCH_ROTATE', 'Camera connected. Odom and lidar ready. Start 360 scan.')
            return

        # 刚从抓取占位状态恢复时，留一点点空档，避免立刻又被同一目标抢占
        if self.resume_time > 0.0 and time.monotonic() < self.resume_time:
            self.stop_robot()
            return
        if self.resume_time > 0.0 and time.monotonic() >= self.resume_time:
            self.resume_time = 0.0

        if self.state == 'SEARCH_ROTATE':
            self.handle_search_rotate()
        elif self.state == 'ALIGN_TO_TARGET':
            self.handle_align_to_target()
        elif self.state == 'APPROACH_TARGET':
            self.handle_approach_target()
        elif self.state == 'TARGET_HOLD':
            self.handle_target_hold()
        elif self.state == 'PATROL_FORWARD':
            self.handle_patrol_forward()
        elif self.state == 'RETURN_TO_MIDPOINT':
            self.handle_return_to_midpoint()
        elif self.state == 'TURN_LEFT_90':
            self.handle_turn_left_90()
        else:
            self.stop_robot()
            print(f'[WARN] Unknown state: {self.state}')

    def status_loop(self):
        print(
            f'[STATUS] '
            f'ready={self.ready()} '
            f'state={self.state} '
            f'state_age={self.state_age():.2f} '
            f'pos=[{self.local_x:.2f}, {self.local_y:.2f}] '
            f'yaw_deg={math.degrees(self.local_yaw):.1f} '
            f'front={self.front_dist:.2f} '
            f'left={self.left_dist:.2f} '
            f'right={self.right_dist:.2f} '
            f'red_visible={self.red_visible} '
            f'red_area={self.red_area:.0f} '
            f'red_seen={self.red_seen_frames}'
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
        print('[EVENT] Keyboard interrupt received. Stopping robot.')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
