#!/usr/bin/env python3
import math
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan


@dataclass
class CubeObservation:
    color: str
    cx: float
    cy: float
    area: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    holes: int
    hole_pitch: float
    hole_diam: float
    score: float


class TrackingCubeV1(Node):
    def __init__(self):
        super().__init__('tracking_cube_v1')

        # =========================
        # 行为参数
        # =========================
        self.TARGET_COLOR = 'any'  # 'any' / 'red' / 'blue'

        self.PATROL_SPEED = 0.05
        self.BACKWARD_SPEED = -0.04
        self.APPROACH_FAST_SPEED = 0.05
        self.APPROACH_SLOW_SPEED = 0.025

        self.SEARCH_ANG = 0.22
        self.ALIGN_MIN_ANG = 0.04
        self.ALIGN_MAX_ANG = 0.20
        self.TURN_MIN_ANG = 0.05
        self.TURN_MAX_ANG = 0.18
        self.RECOVER_ANG = 0.12

        self.WALL_STOP_DIST = 0.32
        self.EMERGENCY_STOP_DIST = 0.12
        self.HARD_FRONT_STOP_DIST = 0.16

        self.SEARCH_DONE_TOL = math.radians(4.0)
        self.TURN_YAW_TOL = math.radians(3.0)
        self.MIDPOINT_TOL = 0.025

        self.ALIGN_PIXEL_TOL = 18
        self.APPROACH_ROTATE_ONLY_PX = 120
        self.REACQUIRE_PIXEL_TOL = 85
        self.CENTER_HOLD_FRAMES = 3
        self.CONFIRM_FRAMES = 2
        self.LOST_TARGET_TIMEOUT = 0.70

        self.MIN_CONTOUR_AREA = 260
        self.MIN_BBOX_W = 12
        self.MIN_BBOX_H = 12
        self.MAX_ASPECT_RATIO = 2.2
        self.MIN_ASPECT_RATIO = 0.40

        self.MIN_HOLES_FOR_RANGE = 3
        self.SLOW_HOLE_PITCH_PX = 14.0
        self.STOP_HOLE_PITCH_PX = 18.0
        self.STOP_BBOX_H_PX = 150

        self.CONTROL_DT = 0.05
        self.STATUS_DT = 1.0

        # =========================
        # ROS 通信
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
        # 传感器缓存
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
        self.image_height = None

        # =========================
        # 目标缓存
        # =========================
        self.target_visible = False
        self.target_obs = None
        self.filtered_cx = None
        self.filtered_cy = None
        self.filtered_pitch = 0.0
        self.filtered_diam = 0.0
        self.filtered_bbox_h = 0.0
        self.target_seen_frames = 0
        self.last_target_seen_time = 0.0
        self.last_target_dir = 1.0
        self.last_target_pitch = 0.0
        self.last_target_color = 'none'

        # =========================
        # 状态机
        # =========================
        self.state = 'WAIT_FOR_DATA'
        self.state_enter_time = time.monotonic()
        self.search_prev_yaw = None
        self.search_accum_yaw = 0.0
        self.turn_target_yaw = 0.0

        self.segment_start = np.array([0.0, 0.0], dtype=float)
        self.segment_heading_yaw = 0.0
        self.segment_length = 0.0
        self.segment_mid_progress = 0.0

        self.shutdown_requested = False
        self.shutdown_reason = ''
        self.shutdown_count = 0
        self.manual_stop_requested = False
        self.wait_status_last = None

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        print('[BOOT] TrackingCube_V1 started')
        print('[CMD] 输入 H 停车并退出')

    # =========================
    # 基础工具
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

    def begin_new_segment_here(self):
        self.segment_start = self.current_pos().copy()
        self.segment_heading_yaw = self.local_yaw
        self.segment_length = 0.0
        self.segment_mid_progress = 0.0

    def current_progress_along_segment(self):
        direction = np.array([
            math.cos(self.segment_heading_yaw),
            math.sin(self.segment_heading_yaw),
        ])
        return float(np.dot(self.current_pos() - self.segment_start, direction))

    def target_confirmed(self):
        return self.target_visible and self.target_seen_frames >= self.CONFIRM_FRAMES

    def target_recently_seen(self):
        return (time.monotonic() - self.last_target_seen_time) <= self.LOST_TARGET_TIMEOUT

    def target_error_pixels(self):
        if not self.target_visible or self.image_width is None or self.target_obs is None:
            return None
        return self.target_obs.cx - self.image_width / 2.0

    def target_close_enough(self):
        if not self.target_visible or self.target_obs is None:
            return False

        by_holes = (
            self.target_obs.holes >= self.MIN_HOLES_FOR_RANGE
            and self.target_obs.hole_pitch >= self.STOP_HOLE_PITCH_PX
        )

        by_bbox = (
            self.target_obs.bbox_h >= self.STOP_BBOX_H_PX
            and self.target_obs.holes >= 2
        )

        by_lidar = self.front_dist <= self.HARD_FRONT_STOP_DIST
        return by_holes or by_bbox or by_lidar

    def set_state(self, new_state, text=None):
        if self.state == new_state:
            return

        self.state = new_state
        self.state_enter_time = time.monotonic()

        if new_state == 'SEARCH_SWEEP':
            self.search_prev_yaw = self.local_yaw
            self.search_accum_yaw = 0.0
        elif new_state == 'PATROL_FORWARD':
            self.begin_new_segment_here()
        elif new_state == 'TURN_LEFT_90':
            self.turn_target_yaw = self.normalize_angle(self.local_yaw + math.pi / 2.0)

        if text:
            print(f'[STATE] {new_state} | {text}')
        else:
            print(f'[STATE] {new_state}')

    def request_shutdown(self, reason):
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.shutdown_reason = reason
        self.shutdown_count = 0
        print(f'[STOP] {reason}')

    # =========================
    # 控制台输入
    # =========================
    def console_loop(self):
        while True:
            try:
                cmd = input().strip().lower()
            except EOFError:
                return
            except Exception:
                return

            if cmd == 'h':
                self.manual_stop_requested = True
                self.request_shutdown('收到 H 指令，停车并退出')
                return
            if cmd in ('q', 'quit', 'exit'):
                self.request_shutdown('收到退出指令')
                return

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
            print('[INFO] odom initialized')

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
            self.target_visible = False
            self.target_obs = None
            self.target_seen_frames = 0
            return

        self.image_height, self.image_width = frame.shape[:2]
        obs = self.detect_best_cube(frame)

        if obs is None:
            self.target_visible = False
            self.target_obs = None
            self.target_seen_frames = 0
            return

        alpha = 0.72
        if self.filtered_cx is None or self.last_target_color != obs.color:
            self.filtered_cx = obs.cx
            self.filtered_cy = obs.cy
            self.filtered_pitch = obs.hole_pitch
            self.filtered_diam = obs.hole_diam
            self.filtered_bbox_h = float(obs.bbox_h)
        else:
            self.filtered_cx = alpha * self.filtered_cx + (1.0 - alpha) * obs.cx
            self.filtered_cy = alpha * self.filtered_cy + (1.0 - alpha) * obs.cy
            self.filtered_pitch = alpha * self.filtered_pitch + (1.0 - alpha) * obs.hole_pitch
            self.filtered_diam = alpha * self.filtered_diam + (1.0 - alpha) * obs.hole_diam
            self.filtered_bbox_h = alpha * self.filtered_bbox_h + (1.0 - alpha) * float(obs.bbox_h)

        obs.cx = float(self.filtered_cx)
        obs.cy = float(self.filtered_cy)
        obs.hole_pitch = float(max(obs.hole_pitch, self.filtered_pitch))
        obs.hole_diam = float(max(obs.hole_diam, self.filtered_diam))
        obs.bbox_h = int(max(obs.bbox_h, round(self.filtered_bbox_h)))

        self.target_visible = True
        self.target_obs = obs
        self.target_seen_frames += 1
        self.last_target_seen_time = time.monotonic()
        self.last_target_pitch = obs.hole_pitch
        self.last_target_color = obs.color

        err = obs.cx - self.image_width / 2.0
        if abs(err) > 2.0:
            self.last_target_dir = -1.0 if err > 0.0 else 1.0

    # =========================
    # 图像处理
    # =========================
    def detect_best_cube(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        color_masks = {}

        lower_red_1 = np.array([0, 70, 40])
        upper_red_1 = np.array([12, 255, 255])
        lower_red_2 = np.array([165, 70, 40])
        upper_red_2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2),
        )
        color_masks['red'] = red_mask

        lower_blue = np.array([95, 70, 35])
        upper_blue = np.array([135, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        color_masks['blue'] = blue_mask

        candidate_colors = ['red', 'blue'] if self.TARGET_COLOR == 'any' else [self.TARGET_COLOR]
        best = None

        for color in candidate_colors:
            mask = color_masks[color]
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                if w < self.MIN_BBOX_W or h < self.MIN_BBOX_H:
                    continue

                aspect = w / float(h)
                if aspect < self.MIN_ASPECT_RATIO or aspect > self.MAX_ASPECT_RATIO:
                    continue

                roi_gray = gray[y:y + h, x:x + w]
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                shifted = contour - np.array([[x, y]])
                cv2.drawContours(roi_mask, [shifted], -1, 255, thickness=-1)

                holes, hole_pitch, hole_diam = self.detect_holes(roi_gray, roi_mask)
                bbox_area = float(w * h)
                score = area + 80.0 * holes + 8.0 * hole_pitch + 2.0 * hole_diam

                obs = CubeObservation(
                    color=color,
                    cx=float(x + w / 2.0),
                    cy=float(y + h / 2.0),
                    area=float(area),
                    bbox_x=int(x),
                    bbox_y=int(y),
                    bbox_w=int(w),
                    bbox_h=int(h),
                    holes=int(holes),
                    hole_pitch=float(hole_pitch),
                    hole_diam=float(hole_diam),
                    score=float(score + 0.02 * bbox_area),
                )

                if best is None or obs.score > best.score:
                    best = obs

        return best

    def detect_holes(self, gray_roi, color_mask_roi):
        if gray_roi.size == 0 or color_mask_roi.size == 0:
            return 0, 0.0, 0.0

        inner_mask = cv2.erode(color_mask_roi, np.ones((5, 5), np.uint8), iterations=1)
        valid_pixels = gray_roi[inner_mask > 0]
        if valid_pixels.size < 30:
            return 0, 0.0, 0.0

        dark_threshold = int(np.clip(np.percentile(valid_pixels, 24), 20, 95))
        dark_mask = cv2.inRange(gray_roi, 0, dark_threshold)
        dark_mask = cv2.bitwise_and(dark_mask, inner_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_area = max(1.0, float(np.count_nonzero(inner_mask)))
        pts = []
        diams = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(6.0, 0.00025 * mask_area):
                continue
            if area > 0.030 * mask_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 1e-6:
                continue

            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.20:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = float(M['m10'] / M['m00'])
            cy = float(M['m01'] / M['m00'])
            pts.append((cx, cy))
            diams.append(math.sqrt(4.0 * area / math.pi))

        if not pts:
            return 0, 0.0, 0.0

        if len(pts) >= 2:
            arr = np.array(pts, dtype=np.float32)
            nn = []
            for i in range(len(arr)):
                d = np.linalg.norm(arr - arr[i], axis=1)
                d = d[d > 1.0]
                if d.size > 0:
                    nn.append(float(np.min(d)))
            hole_pitch = float(np.median(nn)) if nn else 0.0
        else:
            hole_pitch = 0.0

        hole_diam = float(np.median(diams)) if diams else 0.0
        return len(pts), hole_pitch, hole_diam

    # =========================
    # 状态行为
    # =========================
    def handle_search_sweep(self):
        if self.target_confirmed():
            self.stop_robot_once()
            self.set_state('ALIGN_TARGET', f'发现 {self.target_obs.color} 方块，开始对准')
            return

        delta = self.normalize_angle(self.local_yaw - self.search_prev_yaw)
        self.search_accum_yaw += abs(delta)
        self.search_prev_yaw = self.local_yaw

        if self.search_accum_yaw >= 2.0 * math.pi - self.SEARCH_DONE_TOL:
            self.stop_robot_once()
            self.set_state('PATROL_FORWARD', '一圈未锁定目标，进入巡航')
            return

        self.publish_cmd(0.0, self.SEARCH_ANG)

    def handle_align_target(self):
        if self.front_dist <= self.EMERGENCY_STOP_DIST:
            self.request_shutdown('前方过近，紧急停车')
            return

        if not self.target_visible or self.target_obs is None or self.image_width is None:
            if self.target_recently_seen():
                self.publish_cmd(0.0, self.last_target_dir * self.RECOVER_ANG)
                return
            self.stop_robot_once()
            self.set_state('SEARCH_SWEEP', '目标丢失，重新搜索')
            return

        center_x = self.image_width / 2.0
        error = self.target_obs.cx - center_x
        error_norm = error / center_x

        if abs(error) <= self.ALIGN_PIXEL_TOL:
            if self.target_seen_frames >= self.CENTER_HOLD_FRAMES:
                self.stop_robot_once()
                self.set_state('APPROACH_TARGET', f'{self.target_obs.color} 方块已对正，开始前进')
                return
            self.stop_robot_once()
            return

        angular = self.clamp_abs(-0.30 * error_norm, self.ALIGN_MIN_ANG, self.ALIGN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_approach_target(self):
        if self.front_dist <= self.EMERGENCY_STOP_DIST:
            self.request_shutdown('前方过近，紧急停车')
            return

        if not self.target_visible or self.target_obs is None or self.image_width is None:
            if self.target_recently_seen():
                self.publish_cmd(0.0, self.last_target_dir * self.RECOVER_ANG)
                return

            if self.last_target_pitch >= self.STOP_HOLE_PITCH_PX * 0.90:
                self.request_shutdown('目标已到达近距离，停车')
            else:
                self.stop_robot_once()
                self.set_state('SEARCH_SWEEP', '前进时丢失目标，重新搜索')
            return

        error = self.target_error_pixels()
        center_x = self.image_width / 2.0
        error_norm = error / center_x

        if self.target_close_enough():
            self.request_shutdown(
                f'到达 {self.target_obs.color} 方块前方，holes={self.target_obs.holes} pitch={self.target_obs.hole_pitch:.1f}'
            )
            return

        if abs(error) >= self.REACQUIRE_PIXEL_TOL:
            self.stop_robot_once()
            self.set_state('ALIGN_TARGET', '偏差过大，先重新对准')
            return

        if abs(error) > self.APPROACH_ROTATE_ONLY_PX:
            angular = self.clamp_abs(-0.32 * error_norm, self.ALIGN_MIN_ANG, self.ALIGN_MAX_ANG)
            self.publish_cmd(0.0, angular)
            return

        if self.target_obs.hole_pitch >= self.SLOW_HOLE_PITCH_PX or abs(error) > self.ALIGN_PIXEL_TOL * 1.7:
            linear = self.APPROACH_SLOW_SPEED
        else:
            linear = self.APPROACH_FAST_SPEED

        if abs(error) <= self.ALIGN_PIXEL_TOL:
            angular = 0.0
        else:
            angular = self.clamp_abs(-0.24 * error_norm, 0.02, 0.10)

        self.publish_cmd(linear, angular)

    def handle_patrol_forward(self):
        if self.target_confirmed():
            self.stop_robot_once()
            self.set_state('ALIGN_TARGET', f'巡航时发现 {self.target_obs.color} 方块，开始对准')
            return

        progress = self.current_progress_along_segment()
        if self.front_dist <= self.WALL_STOP_DIST:
            self.stop_robot_once()
            self.segment_length = max(0.0, progress)
            self.segment_mid_progress = 0.5 * self.segment_length
            self.set_state('RETURN_TO_MIDPOINT', '前方接近边界，返回中点')
            return

        self.publish_cmd(self.PATROL_SPEED, 0.0)

    def handle_return_to_midpoint(self):
        if self.target_confirmed():
            self.stop_robot_once()
            self.set_state('ALIGN_TARGET', f'后退时发现 {self.target_obs.color} 方块，开始对准')
            return

        progress = self.current_progress_along_segment()
        if progress <= self.segment_mid_progress + self.MIDPOINT_TOL:
            self.stop_robot_once()
            self.set_state('TURN_LEFT_90', '回到中点，左转 90 度')
            return

        self.publish_cmd(self.BACKWARD_SPEED, 0.0)

    def handle_turn_left_90(self):
        if self.target_confirmed():
            self.stop_robot_once()
            self.set_state('ALIGN_TARGET', f'转弯时发现 {self.target_obs.color} 方块，开始对准')
            return

        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= self.TURN_YAW_TOL:
            self.stop_robot_once()
            self.set_state('PATROL_FORWARD', '转弯完成，继续巡航')
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
                    self.stop_robot_reliable(repeat=15, delay=0.03)
                except Exception:
                    pass
                rclpy.shutdown()
            return

        if not self.ready():
            self.stop_robot_once()
            return

        if self.state == 'WAIT_FOR_DATA':
            self.stop_robot_once()
            self.set_state('SEARCH_SWEEP', '传感器准备完成，开始搜索')
            return

        if self.state == 'SEARCH_SWEEP':
            self.handle_search_sweep()
        elif self.state == 'ALIGN_TARGET':
            self.handle_align_target()
        elif self.state == 'APPROACH_TARGET':
            self.handle_approach_target()
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
                print(f'[WAIT] scan={self.has_scan} odom={self.has_odom} image={self.has_image}')
                self.wait_status_last = wait_now
            return

        if self.shutdown_requested:
            print(f'[STATUS] stopping | reason={self.shutdown_reason}')
            return

        if self.target_visible and self.target_obs is not None:
            print(
                f'[STATUS] {self.state} '
                f'color={self.target_obs.color} '
                f'err={self.target_error_pixels():.0f} '
                f'holes={self.target_obs.holes} '
                f'pitch={self.target_obs.hole_pitch:.1f} '
                f'front={self.front_dist:.2f}'
            )
        else:
            print(f'[STATUS] {self.state} color=none front={self.front_dist:.2f}')

    def destroy_node(self):
        try:
            self.stop_robot_reliable(repeat=15, delay=0.03)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrackingCubeV1()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('[STOP] KeyboardInterrupt，停车退出')
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
