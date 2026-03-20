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
        super().__init__('red_cube_patrol_v5')

        # =========================
        # 参数（真机测试版）
        # =========================
        self.PATROL_SPEED = 0.1
        self.BACKWARD_SPEED = -0.04
        self.TRACK_SPEED = 0.1
        self.TRACK_SLOW_SPEED = 0.05
        
        self.SEARCH_SPEED = 0.4  # Smooth 360 scanning speed (Rad/s). Adjust if needed.

        self.SEARCH_MAX_ANG = 0.16
        self.SEARCH_MIN_ANG = 0.04
        self.TURN_MAX_ANG = 0.14
        self.TURN_MIN_ANG = 0.04
        self.TRACK_MAX_ANG = 0.20
        self.TRACK_MIN_ANG = 0.03
        self.REACQUIRE_ANG = 0.10

        self.WALL_STOP_DIST = 0.32
        self.EMERGENCY_WALL_DIST = 0.12
        self.TARGET_STOP_DIST = 0.16

        self.SEARCH_DONE_TOL = math.radians(4)
        self.TURN_YAW_TOL = math.radians(2.5)
        self.MIDPOINT_TOL = 0.025

        self.RED_MIN_AREA = 550
        self.ALIGN_PIXEL_TOL = 18
        self.MOVE_PIXEL_TOL = 80
        self.ROTATE_ONLY_PIXEL_TOL = 120
        self.LOST_TARGET_TIMEOUT = 0.70

        self.CONTROL_DT = 0.05
        self.STATUS_DT = 1.0

        self.AUTO_EXIT_ON_TARGET_STOP = True

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
        # 传感器数据
        # =========================
        self.has_scan = False
        self.has_odom = False
        self.has_image = False

        self.front_dist = float('inf')
        self.left_front_dist = float('inf')
        self.right_front_dist = float('inf')

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
        self.red_bbox_w = 0
        self.red_bbox_h = 0

        self.last_target_seen_time = 0.0
        self.last_target_dir = 1.0
        self.filtered_cx = None
        self.filtered_area = 0.0

        # =========================
        # 状态机变量
        # =========================
        self.state = 'WAIT_FOR_DATA'
        self.state_enter_time = time.monotonic()

        self.search_prev_yaw = None
        self.search_accum_yaw = 0.0
        self.turn_target_yaw = 0.0
        
        # New variables for Smart Sweeping
        self.first_cube_yaw = None
        self.CENTER_FOV_RATIO = 0.50

        self.segment_start = np.array([0.0, 0.0], dtype=float)
        self.segment_heading_yaw = 0.0
        self.segment_length = 0.0
        self.segment_mid_progress = 0.0

        self.shutdown_requested = False
        self.shutdown_reason = ''
        self.shutdown_count = 0
        self.wait_status_last = None

        print('[BOOT] Node started. Waiting for data...')

    # =========================
    # 工具函数
    # =========================
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def clamp_abs(self, value, min_abs, max_abs):
        if abs(value) < 1e-9:
            return 0.0
        sign = 1.0 if value > 0.0 else -1.0
        mag = self.clamp(abs(value), min_abs, max_abs)
        return sign * mag

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        try:
            msg = Twist()
            msg.linear.x = float(linear_x)
            msg.angular.z = float(angular_z)
            self.cmd_pub.publish(msg)
        except Exception:
            pass

    def stop_robot_once(self):
        self.publish_cmd(0.0, 0.0)

    def stop_robot_reliable(self, repeat=12, delay=0.03):
        for _ in range(repeat):
            self.stop_robot_once()
            time.sleep(delay)

    def ready(self):
        return self.has_scan and self.has_odom and self.has_image

    def state_age(self):
        return time.monotonic() - self.state_enter_time

    def current_pos(self):
        return np.array([self.local_x, self.local_y], dtype=float)

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

    def target_seen(self):
        return self.red_visible and self.red_area >= self.RED_MIN_AREA and self.image_width is not None

    def target_recently_seen(self):
        return (time.monotonic() - self.last_target_seen_time) <= self.LOST_TARGET_TIMEOUT

    def target_error_pixels(self):
        if not self.target_seen() or self.image_width is None:
            return None
        return self.red_cx - self.image_width / 2.0

    def request_shutdown(self, reason):
        if not self.shutdown_requested:
            self.shutdown_requested = True
            self.shutdown_reason = reason
            self.shutdown_count = 0
            print(f'[EVENT] {reason}')

    def set_state(self, new_state, event_text=None):
        if self.state == new_state:
            return

        self.state = new_state
        self.state_enter_time = time.monotonic()

        if event_text:
            print(f'[EVENT] {event_text}')

        if new_state == 'SEARCH_ROTATE':
            self.search_prev_yaw = self.local_yaw
            self.search_accum_yaw = 0.0
            self.first_cube_yaw = None
        elif new_state == 'PATROL_FORWARD':
            self.begin_new_segment_here()
        elif new_state == 'TURN_LEFT_90':
            self.turn_target_yaw = self.normalize_angle(self.local_yaw + math.pi / 2.0)

    # =========================
    # 回调函数
    # =========================
    def scan_callback(self, msg):
        front_vals = list(msg.ranges[0:8]) + list(msg.ranges[352:360])
        left_front_vals = list(msg.ranges[15:45])
        right_front_vals = list(msg.ranges[315:345])

        def valid_min(vals):
            good = [x for x in vals if math.isfinite(x) and x > 0.05]
            return min(good) if good else float('inf')

        raw_front = valid_min(front_vals)
        raw_left_front = valid_min(left_front_vals)
        raw_right_front = valid_min(right_front_vals)

        if not self.has_scan:
            self.front_dist = raw_front
            self.left_front_dist = raw_left_front
            self.right_front_dist = raw_right_front
        else:
            alpha = 0.55
            self.front_dist = alpha * self.front_dist + (1.0 - alpha) * raw_front
            self.left_front_dist = alpha * self.left_front_dist + (1.0 - alpha) * raw_left_front
            self.right_front_dist = alpha * self.right_front_dist + (1.0 - alpha) * raw_right_front

        self.has_scan = True

    def odom_callback(self, msg):
        self.world_x = msg.pose.pose.position.x
        self.world_y = msg.pose.pose.position.y
        self.world_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        if self.init_world_x is None:
            self.init_world_x = self.world_x
            self.init_world_y = self.world_y
            self.init_world_yaw = self.world_yaw
            print('[EVENT] Local 2D frame initialized.')

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
        except Exception:
            self.red_visible = False
            return

        self.image_width = frame.shape[1]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 80, 50])
        upper_red_1 = np.array([12, 255, 255])
        lower_red_2 = np.array([165, 80, 50])
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
        best_w = 0
        best_h = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.RED_MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 8:
                continue

            cx = x + w / 2.0

            if area > best_area:
                best_area = area
                best_cx = cx
                best_w = w
                best_h = h

        if best_cx is not None:
            if self.filtered_cx is None:
                self.filtered_cx = best_cx
            else:
                self.filtered_cx = 0.70 * self.filtered_cx + 0.30 * best_cx

            self.filtered_area = 0.65 * self.filtered_area + 0.35 * best_area
            self.red_visible = True
            self.red_cx = float(self.filtered_cx)
            self.red_area = float(max(best_area, self.filtered_area))
            self.red_bbox_w = best_w
            self.red_bbox_h = best_h
            self.last_target_seen_time = time.monotonic()

            err = self.red_cx - self.image_width / 2.0
            if abs(err) > 2:
                self.last_target_dir = -1.0 if err > 0 else 1.0
        else:
            self.red_visible = False
            self.red_cx = None
            self.red_area = 0.0
            self.red_bbox_w = 0
            self.red_bbox_h = 0

    # =========================
    # 状态行为函数
    # =========================
    def handle_search_rotate(self):
        # 1. Check for target and log the FIRST one we see
        if self.target_seen():
            error = abs(self.target_error_pixels())
            allowed_fov_pixels = self.image_width * (self.CENTER_FOV_RATIO / 2.0)
            
            if error <= allowed_fov_pixels:
                # Log the first centered cube and stop logging others
                if self.first_cube_yaw is None:
                    self.first_cube_yaw = self.local_yaw
                    print(f"[EVENT] First cube spotted at yaw: {math.degrees(self.first_cube_yaw):.1f}")

        # 2. Safely accumulate rotation (Fixes the early-exit / vibration bug)
        delta = self.normalize_angle(self.local_yaw - self.search_prev_yaw)
        self.search_accum_yaw += abs(delta)  # <-- The abs() fixes the math bug
        self.search_prev_yaw = self.local_yaw

        remaining = 2.0 * math.pi - self.search_accum_yaw
        if remaining <= self.SEARCH_DONE_TOL:
            self.stop_robot_once()
            
            # 3. Scan finished. Did we log a cube?
            if self.first_cube_yaw is not None:
                self.turn_target_yaw = self.first_cube_yaw
                self.set_state('TURN_TO_LOGGED_TARGET', 'Scan complete. Returning to FIRST logged cube.')
            else:
                self.set_state('PATROL_FORWARD', 'Startup scan finished. No cube found. Enter patrol.')
            return

        # 4. Command the constant, adjustable speed
        self.publish_cmd(0.0, self.SEARCH_SPEED)

    def handle_turn_to_logged_target(self):
        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        
        # If we see the cube again while turning back, and it's roughly centered, lock on!
        if self.target_seen():
            fov_error = abs(self.target_error_pixels())
            if fov_error <= (self.image_width * 0.30): 
                self.stop_robot_once()
                self.set_state('TRACK_TARGET', 'Logged target in sight. Starting track.')
                return

        if abs(error) <= self.TURN_YAW_TOL:
            self.stop_robot_once()
            # We reached the logged angle. Transition to track.
            self.set_state('TRACK_TARGET', 'Turned to logged target yaw.')
            return

        # Turn efficiently towards the logged yaw
        angular = self.clamp_abs(0.75 * error, self.TURN_MIN_ANG, self.TURN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_track_target(self):
        if self.front_dist <= self.EMERGENCY_WALL_DIST:
            self.stop_robot_once()
            self.request_shutdown('Emergency stop: front obstacle too close. Program will exit.')
            return

        if self.target_seen():
            error = self.target_error_pixels()
            center_x = self.image_width / 2.0
            error_norm = error / center_x

            if self.front_dist <= self.TARGET_STOP_DIST:
                self.stop_robot_once()
                if self.AUTO_EXIT_ON_TARGET_STOP:
                    self.request_shutdown('Target reached. Program will exit.')
                else:
                    self.set_state('SEARCH_ROTATE', 'Target reached. Resume search.')
                return

            if abs(error) > self.ROTATE_ONLY_PIXEL_TOL:
                angular = self.clamp_abs(-0.34 * error_norm, self.TRACK_MIN_ANG, self.TRACK_MAX_ANG)
                self.publish_cmd(0.0, angular)
                return

            if abs(error) > self.MOVE_PIXEL_TOL:
                linear = self.TRACK_SLOW_SPEED
            else:
                linear = self.TRACK_SPEED

            if abs(error) <= self.ALIGN_PIXEL_TOL:
                angular = 0.0
            else:
                angular = self.clamp_abs(-0.28 * error_norm, self.TRACK_MIN_ANG, self.TRACK_MAX_ANG)

            self.publish_cmd(linear, angular)
            return

        if self.target_recently_seen():
            self.publish_cmd(self.TRACK_SLOW_SPEED, self.last_target_dir * self.REACQUIRE_ANG)
            return

        self.stop_robot_once()
        
        if self.filtered_area > 2000:
            if self.AUTO_EXIT_ON_TARGET_STOP:
                self.request_shutdown('Target reached (slipped under camera). Program will exit.')
            else:
                self.set_state('SEARCH_ROTATE', 'Target reached (slipped under camera). Resume search.')
        else:
            self.set_state('SEARCH_ROTATE', 'Target lost at a distance. Restart search.')

    def handle_patrol_forward(self):
        if self.target_seen():
            self.stop_robot_once()
            self.set_state('TRACK_TARGET', 'Target found during patrol.')
            return

        progress = self.current_progress_along_segment()
        if self.front_dist <= self.WALL_STOP_DIST:
            self.stop_robot_once()
            self.segment_length = max(0.0, progress)
            self.segment_mid_progress = 0.5 * self.segment_length
            self.set_state('RETURN_TO_MIDPOINT', 'Wall detected. Return to midpoint.')
            return

        self.publish_cmd(self.PATROL_SPEED, 0.0)

    def handle_return_to_midpoint(self):
        if self.target_seen():
            self.stop_robot_once()
            self.set_state('TRACK_TARGET', 'Target found while backing.')
            return

        progress = self.current_progress_along_segment()
        if progress <= self.segment_mid_progress + self.MIDPOINT_TOL:
            self.stop_robot_once()
            self.set_state('TURN_LEFT_90', 'Midpoint reached. Turn left 90.')
            return

        self.publish_cmd(self.BACKWARD_SPEED, 0.0)

    def handle_turn_left_90(self):
        if self.target_seen():
            self.stop_robot_once()
            self.set_state('TRACK_TARGET', 'Target found during turn.')
            return

        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= self.TURN_YAW_TOL:
            self.stop_robot_once()
            self.set_state('PATROL_FORWARD', 'Turn finished. Continue patrol.')
            return

        angular = self.clamp_abs(0.75 * error, self.TURN_MIN_ANG, self.TURN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    # =========================
    # 主循环
    # =========================
    def control_loop(self):
        if self.shutdown_requested:
            self.stop_robot_once()
            self.shutdown_count += 1
            if self.shutdown_count >= 8:
                try:
                    self.stop_robot_reliable(repeat=12, delay=0.03)
                except Exception:
                    pass
                rclpy.shutdown()
            return

        if not self.ready():
            self.stop_robot_once()
            return

        if self.state == 'WAIT_FOR_DATA':
            self.stop_robot_once()
            self.set_state('SEARCH_ROTATE', 'All sensors ready. Start 360 scan.')
            return

        if self.state == 'SEARCH_ROTATE':
            self.handle_search_rotate()
        elif self.state == 'TURN_TO_LOGGED_TARGET':
            self.handle_turn_to_logged_target()
        elif self.state == 'TRACK_TARGET':
            self.handle_track_target()
        elif self.state == 'PATROL_FORWARD':
            self.handle_patrol_forward()
        elif self.state == 'RETURN_TO_MIDPOINT':
            self.handle_return_to_midpoint()
        elif self.state == 'TURN_LEFT_90':
            self.handle_turn_left_90()
        else:
            self.stop_robot_once()

    def status_loop(self):
        if not self.ready():
            wait_now = (self.has_scan, self.has_odom, self.has_image)
            if wait_now != self.wait_status_last:
                print(f'[STATUS] waiting scan={self.has_scan} odom={self.has_odom} image={self.has_image}')
                self.wait_status_last = wait_now
            return

        if self.shutdown_requested:
            print(f'[STATUS] stopping state={self.state}')
            return

        target_text = 'yes' if self.target_seen() else 'no'
        area_text = int(self.red_area) if self.target_seen() else 0
        print(
            f'[STATUS] state={self.state} '
            f'pos=[{self.local_x:.2f},{self.local_y:.2f}] '
            f'yaw={math.degrees(self.local_yaw):.0f} '
            f'front={self.front_dist:.2f} '
            f'target={target_text} '
            f'area={area_text}'
        )

    def destroy_node(self):
        try:
            self.stop_robot_reliable(repeat=15, delay=0.03)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RedCubePatrol()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('[EVENT] Keyboard interrupt received. Stopping robot and exiting.')
        try:
            node.stop_robot_reliable(repeat=15, delay=0.03)
        except Exception:
            pass
    finally:
        try:
            node.stop_robot_reliable(repeat=15, delay=0.03)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()