#!/usr/bin/env python3
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

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
    fill_ratio: float
    extent: float
    solidity: float
    score: float
    color_conf: float


class ServoArm:
    def __init__(self):
        self.available = False
        self.last_action = 'dummy'
        self.lift_servo = None
        self.clamp_servo = None

        self.LIFT_PIN = 12
        self.CLAMP_PIN = 13

        self.LIFT_STOW_ANGLE = 20
        self.LIFT_CAPTURE_ANGLE = 96
        self.LIFT_RELEASE_ANGLE = 18

        self.CLAMP_STOW_ANGLE = 88
        self.CLAMP_CAPTURE_ANGLE = 90
        self.CLAMP_RELEASE_ANGLE = 88

        try:
            from gpiozero import AngularServo
            from gpiozero.pins.pigpio import PiGPIOFactory

            factory = PiGPIOFactory()
            self.lift_servo = AngularServo(
                self.LIFT_PIN,
                min_angle=0,
                max_angle=180,
                min_pulse_width=0.0005,
                max_pulse_width=0.0025,
                pin_factory=factory,
            )
            self.clamp_servo = AngularServo(
                self.CLAMP_PIN,
                min_angle=0,
                max_angle=180,
                min_pulse_width=0.0005,
                max_pulse_width=0.0025,
                pin_factory=factory,
            )
            self.available = True
            print('[ARM] gpiozero + pigpio 已连接')
            self.stow()
        except Exception as e:
            print(f'[ARM] 未连接真实舵机，当前为模拟模式: {e}')

    def _set(self, lift_angle: Optional[float] = None, clamp_angle: Optional[float] = None):
        if self.available:
            if lift_angle is not None:
                self.lift_servo.angle = float(lift_angle)
            if clamp_angle is not None:
                self.clamp_servo.angle = float(clamp_angle)
        self.last_action = f'lift={lift_angle} clamp={clamp_angle}'

    def stow(self):
        self._set(self.LIFT_STOW_ANGLE, self.CLAMP_STOW_ANGLE)
        print('[ARM] stow')

    def capture(self):
        self._set(self.LIFT_CAPTURE_ANGLE, self.CLAMP_CAPTURE_ANGLE)
        print('[ARM] capture/down')

    def release(self):
        self._set(self.LIFT_RELEASE_ANGLE, self.CLAMP_RELEASE_ANGLE)
        print('[ARM] release/up')


class TrackingCubeV5(Node):
    MARKER_LABELS = {
        0: 'west',
        7: 'north',
        23: 'east',
        42: 'south',
    }

    def __init__(self):
        super().__init__('tracking_cube_v5')

        self.arm = ServoArm()
        self.TARGET_COLOR = 'any'

        self.SEARCH_ANG = 1.85
        self.ALIGN_MIN_ANG = 0.18
        self.ALIGN_MAX_ANG = 1.55
        self.RECOVER_ANG = 0.85
        self.APPROACH_FAST_SPEED = 0.28
        self.APPROACH_SLOW_SPEED = 0.16
        self.DELIVERY_REVERSE_SPEED = -0.24
        self.RETURN_FORWARD_SPEED = 0.26
        self.TURN_TO_SIDE_MAX_ANG = 1.25

        self.EMERGENCY_STOP_DIST = 0.11
        self.CAPTURE_FRONT_DIST = 0.14
        self.CLOSE_FRONT_DIST = 0.18

        self.ALIGN_PIXEL_TOL = 20
        self.APPROACH_ROTATE_ONLY_PX = 84
        self.REACQUIRE_PIXEL_TOL = 72
        self.LOST_TARGET_TIMEOUT = 0.48
        self.CENTER_HOLD_FRAMES = 2
        self.SEARCH_PRELOCK_FRAMES = 2
        self.CONFIRM_FRAMES = 3

        self.MIN_CONTOUR_AREA = 320
        self.MIN_TRACK_AREA = 360
        self.SEARCH_LOCK_MIN_AREA = 560
        self.MIN_BBOX_W = 16
        self.MIN_BBOX_H = 16
        self.MAX_ASPECT_RATIO = 1.95
        self.MIN_ASPECT_RATIO = 0.52
        self.MIN_FILL_RATIO = 0.30
        self.MIN_EXTENT = 0.22
        self.MIN_SOLIDITY = 0.70
        self.MIN_CENTER_Y_RATIO = 0.16
        self.MAX_RAW_JUMP_PX = 120.0
        self.SEARCH_ENTRY_CENTER_RATIO = 0.34
        self.CENTER_SCORE_GAIN = 280.0
        self.SAME_COLOR_BONUS = 1100.0

        self.MIN_HOLES_FOR_RANGE = 3
        self.CAPTURE_HOLE_PITCH_PX = 19.0
        self.CAPTURE_BBOX_H_PX = 178
        self.CAPTURE_BBOX_W_PX = 170

        self.ARM_SETTLE_SEC = 0.95
        self.RELEASE_SETTLE_SEC = 0.55
        self.DRAG_DISTANCE_M = 0.56
        self.RETURN_DISTANCE_M = 0.50

        self.BLUE_DELIVERY_FACE_YAW = -math.pi / 2.0
        self.RED_DELIVERY_FACE_YAW = math.pi / 2.0
        self.BLUE_DELIVERY_FACE_LABEL = 'east'
        self.RED_DELIVERY_FACE_LABEL = 'west'

        self.CONTROL_DT = 0.05
        self.STATUS_DT = 0.8

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

        self.target_visible = False
        self.target_obs: Optional[CubeObservation] = None
        self.filtered_cx = None
        self.filtered_cy = None
        self.filtered_pitch = 0.0
        self.filtered_bbox_h = 0.0
        self.filtered_bbox_w = 0.0
        self.target_seen_frames = 0
        self.last_target_seen_time = 0.0
        self.last_target_dir = 1.0
        self.last_target_pitch = 0.0
        self.last_target_color = 'none'
        self.prev_raw_obs = None
        self.lock_color = None
        self.current_job_color = None
        self.capture_decision_close = False

        self.marker_label = None
        self.marker_cx = None
        self.marker_area = 0.0
        self.marker_seen_time = 0.0
        self.aruco_detector = None
        self.aruco_dict = None
        self.setup_marker_detector()

        self.state = 'WAIT_FOR_DATA'
        self.state_enter_time = time.monotonic()
        self.search_prev_yaw = None
        self.search_accum_yaw = 0.0
        self.search_dir = 1.0
        self.turn_target_yaw = 0.0
        self.travel_start = np.array([0.0, 0.0], dtype=float)
        self.initial_heading_yaw = 0.0

        self.shutdown_requested = False
        self.shutdown_reason = ''
        self.shutdown_count = 0
        self.manual_stop_requested = False
        self.wait_status_last = None

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        print('[BOOT] TrackingCube_V5 started')
        print('[BOOT] 开局只原地搜索，不再巡航')
        print('[BOOT] 找到方块 -> 对准 -> 接近 -> 落臂 -> 转向目标侧 -> 倒车拖动 -> 抬臂 -> 回中间')
        print('[CMD] 输入 H 停车并退出')

    def setup_marker_detector(self):
        try:
            if hasattr(cv2, 'aruco'):
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
                params = cv2.aruco.DetectorParameters()
                if hasattr(cv2.aruco, 'ArucoDetector'):
                    self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
                print('[TAG] AprilTag detector ready')
        except Exception as e:
            self.aruco_detector = None
            self.aruco_dict = None
            print(f'[TAG] AprilTag detector unavailable: {e}')

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

    def current_pos(self):
        return np.array([self.local_x, self.local_y], dtype=float)

    def distance_from(self, p):
        return float(np.linalg.norm(self.current_pos() - p))

    def begin_travel_here(self):
        self.travel_start = self.current_pos().copy()

    def target_recently_seen(self):
        return (time.monotonic() - self.last_target_seen_time) <= self.LOST_TARGET_TIMEOUT

    def marker_recently_seen(self):
        return (time.monotonic() - self.marker_seen_time) <= 0.9 and self.marker_label is not None

    def target_error_pixels(self):
        if not self.target_visible or self.image_width is None or self.target_obs is None:
            return None
        return self.target_obs.cx - self.image_width / 2.0

    def desired_delivery_face_yaw(self, color):
        return self.BLUE_DELIVERY_FACE_YAW if color == 'blue' else self.RED_DELIVERY_FACE_YAW

    def desired_delivery_face_label(self, color):
        return self.BLUE_DELIVERY_FACE_LABEL if color == 'blue' else self.RED_DELIVERY_FACE_LABEL

    def lockworthy_target(self):
        if not self.target_visible or self.target_obs is None or self.image_height is None:
            return False
        obs = self.target_obs
        if obs.area < self.SEARCH_LOCK_MIN_AREA:
            return False
        if obs.cy < self.image_height * self.MIN_CENTER_Y_RATIO:
            return False
        if obs.fill_ratio < self.MIN_FILL_RATIO:
            return False
        if obs.extent < self.MIN_EXTENT:
            return False
        if obs.solidity < self.MIN_SOLIDITY:
            return False
        return True

    def target_in_search_sector(self):
        if not self.target_visible or self.target_obs is None or self.image_width is None:
            return False
        center_tol = self.image_width * self.SEARCH_ENTRY_CENTER_RATIO
        err = abs(self.target_obs.cx - self.image_width / 2.0)
        return err <= center_tol

    def search_ready_target(self):
        if not self.lockworthy_target():
            return False
        if not self.target_in_search_sector():
            return False
        return self.target_seen_frames >= self.SEARCH_PRELOCK_FRAMES

    def target_confirmed(self):
        return self.lockworthy_target() and self.target_seen_frames >= self.CONFIRM_FRAMES

    def close_enough_for_capture(self):
        if not self.target_visible or self.target_obs is None:
            return False
        by_holes = (
            self.target_obs.holes >= self.MIN_HOLES_FOR_RANGE
            and self.target_obs.hole_pitch >= self.CAPTURE_HOLE_PITCH_PX
        )
        by_bbox = (
            self.target_obs.bbox_h >= self.CAPTURE_BBOX_H_PX
            or self.target_obs.bbox_w >= self.CAPTURE_BBOX_W_PX
        )
        by_lidar = self.front_dist <= self.CAPTURE_FRONT_DIST
        return by_holes or by_bbox or by_lidar

    def set_state(self, new_state, text=None):
        if self.state == new_state:
            return
        self.state = new_state
        self.state_enter_time = time.monotonic()
        if new_state == 'SEARCH_SWEEP':
            self.search_prev_yaw = self.local_yaw
            self.search_accum_yaw = 0.0
            self.capture_decision_close = False
        elif new_state in ('TURN_TO_DELIVERY', 'DRAG_TO_SIDE', 'RETURN_TO_CENTER'):
            self.begin_travel_here()
        elif new_state == 'RESET_HEADING':
            pass
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

    def scan_callback(self, msg):
        front_vals = list(msg.ranges[0:8]) + list(msg.ranges[352:360])
        left_front_vals = list(msg.ranges[12:45])
        right_front_vals = list(msg.ranges[315:348])

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
            alpha = 0.50
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
            self.initial_heading_yaw = 0.0
            print('[INFO] odom initialized, 当前朝向记为 0 度')

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
            self.prev_raw_obs = None
            return

        self.image_height, self.image_width = frame.shape[:2]
        self.detect_direction_marker(frame)
        obs = self.detect_best_cube(frame)

        if obs is None:
            self.target_visible = False
            self.target_obs = None
            self.target_seen_frames = 0
            self.prev_raw_obs = None
            return

        stable = False
        if self.prev_raw_obs is not None and obs.color == self.prev_raw_obs.color:
            raw_jump = abs(obs.cx - self.prev_raw_obs.cx)
            area_ratio = obs.area / max(self.prev_raw_obs.area, 1.0)
            if raw_jump <= self.MAX_RAW_JUMP_PX and 0.34 <= area_ratio <= 3.0:
                stable = True

        if stable:
            self.target_seen_frames += 1
        else:
            self.target_seen_frames = 1
        self.prev_raw_obs = obs

        alpha = 0.45
        if self.filtered_cx is None or self.last_target_color != obs.color or not stable:
            self.filtered_cx = obs.cx
            self.filtered_cy = obs.cy
            self.filtered_pitch = obs.hole_pitch
            self.filtered_bbox_h = float(obs.bbox_h)
            self.filtered_bbox_w = float(obs.bbox_w)
        else:
            self.filtered_cx = alpha * self.filtered_cx + (1.0 - alpha) * obs.cx
            self.filtered_cy = alpha * self.filtered_cy + (1.0 - alpha) * obs.cy
            self.filtered_pitch = alpha * self.filtered_pitch + (1.0 - alpha) * obs.hole_pitch
            self.filtered_bbox_h = alpha * self.filtered_bbox_h + (1.0 - alpha) * float(obs.bbox_h)
            self.filtered_bbox_w = alpha * self.filtered_bbox_w + (1.0 - alpha) * float(obs.bbox_w)

        obs.cx = float(self.filtered_cx)
        obs.cy = float(self.filtered_cy)
        obs.hole_pitch = float(max(obs.hole_pitch, self.filtered_pitch))
        obs.bbox_h = int(max(obs.bbox_h, round(self.filtered_bbox_h)))
        obs.bbox_w = int(max(obs.bbox_w, round(self.filtered_bbox_w)))

        self.target_visible = obs.area >= self.MIN_TRACK_AREA
        self.target_obs = obs if self.target_visible else None
        if not self.target_visible:
            self.target_seen_frames = 0
            return

        self.last_target_seen_time = time.monotonic()
        self.last_target_pitch = obs.hole_pitch
        self.last_target_color = obs.color

        err = obs.cx - self.image_width / 2.0
        if abs(err) > 2.0:
            self.last_target_dir = -1.0 if err > 0.0 else 1.0

    def build_red_mask(self, hsv, bgr):
        lower_red_1 = np.array([0, 92, 55])
        upper_red_1 = np.array([7, 255, 255])
        lower_red_2 = np.array([174, 92, 55])
        upper_red_2 = np.array([179, 255, 255])
        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2),
        )
        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]
        rgb_mask = np.zeros_like(hsv_mask)
        red_dom = (r >= 78) & (r > g + 28) & (r > b + 34)
        rgb_mask[red_dom] = 255
        return cv2.bitwise_and(hsv_mask, rgb_mask)

    def build_blue_mask(self, hsv, bgr):
        lower_blue = np.array([108, 120, 45])
        upper_blue = np.array([124, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]
        rgb_mask = np.zeros_like(hsv_mask)
        blue_dom = (b >= 82) & (b > r + 50) & (b > g + 24)
        rgb_mask[blue_dom] = 255
        return cv2.bitwise_and(hsv_mask, rgb_mask)

    def color_signature(self, color, roi_bgr, roi_hsv, roi_mask):
        pixels = roi_bgr[roi_mask > 0]
        hsv_pixels = roi_hsv[roi_mask > 0]
        if pixels.size == 0 or hsv_pixels.size == 0:
            return False, 0.0

        mean_b, mean_g, mean_r = np.mean(pixels, axis=0)
        _, mean_s, mean_v = np.mean(hsv_pixels, axis=0)
        hue = hsv_pixels[:, 0]
        sat = hsv_pixels[:, 1]

        if color == 'blue':
            hue_ratio = float(np.mean((hue >= 109) & (hue <= 122) & (sat >= 130)))
            dom_rb = float(mean_b - mean_r)
            dom_gb = float(mean_b - mean_g)
            color_conf = 0.28 * float(mean_s) + 0.34 * dom_rb + 0.22 * dom_gb + 86.0 * hue_ratio
            ok = (
                hue_ratio >= 0.72
                and mean_s >= 155.0
                and mean_v >= 48.0
                and mean_b >= 86.0
                and dom_rb >= 58.0
                and dom_gb >= 25.0
            )
            return ok, float(color_conf)

        hue_ratio = float(np.mean(((hue <= 7) | (hue >= 174)) & (sat >= 100)))
        dom_br = float(mean_r - mean_b)
        dom_gr = float(mean_r - mean_g)
        color_conf = 0.24 * float(mean_s) + 0.28 * dom_br + 0.21 * dom_gr + 76.0 * hue_ratio
        ok = (
            hue_ratio >= 0.54
            and mean_s >= 118.0
            and mean_v >= 58.0
            and mean_r >= 88.0
            and dom_br >= 36.0
            and dom_gr >= 28.0
        )
        return ok, float(color_conf)

    def detect_best_cube(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        color_masks = {
            'red': self.build_red_mask(hsv, frame),
            'blue': self.build_blue_mask(hsv, frame),
        }

        if self.TARGET_COLOR != 'any':
            candidate_colors = [self.TARGET_COLOR]
        elif self.lock_color is not None and self.state not in ('SEARCH_SWEEP', 'WAIT_FOR_DATA'):
            candidate_colors = [self.lock_color]
        else:
            candidate_colors = ['red', 'blue']

        best = None

        for color in candidate_colors:
            mask = color_masks[color]
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                if w < self.MIN_BBOX_W or h < self.MIN_BBOX_H:
                    continue

                cy = y + h / 2.0
                if self.image_height is not None and cy < self.image_height * self.MIN_CENTER_Y_RATIO:
                    continue

                aspect = w / float(h)
                if aspect < self.MIN_ASPECT_RATIO or aspect > self.MAX_ASPECT_RATIO:
                    continue

                bbox_area = float(w * h)
                if bbox_area <= 1.0:
                    continue

                hull = cv2.convexHull(contour)
                hull_area = max(cv2.contourArea(hull), 1.0)
                solidity = float(area / hull_area)
                if solidity < self.MIN_SOLIDITY:
                    continue

                shifted = contour - np.array([[x, y]])
                roi_color_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(roi_color_mask, [shifted], -1, 255, thickness=-1)
                fill_ratio = float(np.count_nonzero(roi_color_mask) / bbox_area)
                if fill_ratio < self.MIN_FILL_RATIO:
                    continue

                extent = float(area / bbox_area)
                if extent < self.MIN_EXTENT:
                    continue

                roi_gray = gray[y:y + h, x:x + w]
                roi_hsv = hsv[y:y + h, x:x + w]
                roi_bgr = frame[y:y + h, x:x + w]

                color_ok, color_conf = self.color_signature(color, roi_bgr, roi_hsv, roi_color_mask)
                if not color_ok:
                    continue

                holes, hole_pitch, hole_diam = self.detect_holes(roi_gray, roi_color_mask)

                center_err = abs((x + w / 2.0) - frame.shape[1] / 2.0)
                center_bonus = self.CENTER_SCORE_GAIN * (1.0 - min(1.0, center_err / max(frame.shape[1] / 2.0, 1.0)))
                same_color_bonus = self.SAME_COLOR_BONUS if self.lock_color == color else 0.0
                score = (
                    1.85 * area
                    + 400.0 * fill_ratio
                    + 320.0 * extent
                    + 250.0 * solidity
                    + 7.5 * color_conf
                    + 18.0 * min(holes, 10)
                    + 1.8 * hole_pitch
                    + center_bonus
                    + same_color_bonus
                )

                obs = CubeObservation(
                    color=color,
                    cx=float(x + w / 2.0),
                    cy=float(cy),
                    area=float(area),
                    bbox_x=int(x),
                    bbox_y=int(y),
                    bbox_w=int(w),
                    bbox_h=int(h),
                    holes=int(holes),
                    hole_pitch=float(hole_pitch),
                    hole_diam=float(hole_diam),
                    fill_ratio=float(fill_ratio),
                    extent=float(extent),
                    solidity=float(solidity),
                    score=float(score),
                    color_conf=float(color_conf),
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

        dark_threshold = int(np.clip(np.percentile(valid_pixels, 20), 18, 95))
        dark_mask = cv2.inRange(gray_roi, 0, dark_threshold)
        dark_mask = cv2.bitwise_and(dark_mask, inner_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_area = max(1.0, float(np.count_nonzero(inner_mask)))
        pts = []
        diams = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(8.0, 0.00028 * mask_area):
                continue
            if area > 0.022 * mask_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.22:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = float(M['m10'] / M['m00'])
            cy = float(M['m01'] / M['m00'])
            pts.append((cx, cy))
            diams.append(math.sqrt(4.0 * area / math.pi))

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

    def detect_direction_marker(self, frame):
        self.marker_label = None
        self.marker_cx = None
        self.marker_area = 0.0
        if self.aruco_dict is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            if self.aruco_detector is not None:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
        except Exception:
            return

        if ids is None or len(ids) == 0:
            return

        best_idx = None
        best_area = 0.0
        for i, tag_id in enumerate(ids.flatten().tolist()):
            if tag_id not in self.MARKER_LABELS:
                continue
            pts = corners[i].reshape(-1, 2)
            area = float(cv2.contourArea(pts.astype(np.float32)))
            if area > best_area:
                best_area = area
                best_idx = i

        if best_idx is None:
            return

        tag_id = int(ids.flatten()[best_idx])
        pts = corners[best_idx].reshape(-1, 2)
        cx = float(np.mean(pts[:, 0]))
        self.marker_label = self.MARKER_LABELS[tag_id]
        self.marker_cx = cx
        self.marker_area = best_area
        self.marker_seen_time = time.monotonic()

    def handle_search_sweep(self):
        if self.search_ready_target() and self.target_obs is not None:
            self.lock_color = self.target_obs.color
            self.stop_robot_once()
            self.set_state('ALIGN_TARGET', f'发现 {self.lock_color} 方块，开始对准')
            return

        delta = self.normalize_angle(self.local_yaw - self.search_prev_yaw)
        self.search_accum_yaw += abs(delta)
        self.search_prev_yaw = self.local_yaw

        if self.search_accum_yaw >= 2.0 * math.pi - math.radians(4.0):
            self.search_accum_yaw = 0.0
            self.search_dir *= -1.0
            print('[SEARCH] 一圈未锁定，反向继续搜索')

        self.publish_cmd(0.0, self.search_dir * self.SEARCH_ANG)

    def handle_align_target(self):
        if self.front_dist <= self.EMERGENCY_STOP_DIST:
            self.request_shutdown('前方过近，紧急停车')
            return

        if not self.target_visible or self.target_obs is None or self.image_width is None:
            if self.target_recently_seen():
                self.publish_cmd(0.0, self.last_target_dir * self.RECOVER_ANG)
                return
            self.lock_color = None
            self.stop_robot_once()
            self.set_state('SEARCH_SWEEP', '目标丢失，重新搜索')
            return

        center_x = self.image_width / 2.0
        error = self.target_obs.cx - center_x
        error_norm = error / center_x

        if abs(error) <= self.ALIGN_PIXEL_TOL:
            if self.target_seen_frames >= self.CENTER_HOLD_FRAMES:
                self.stop_robot_once()
                self.set_state('APPROACH_TARGET', f'{self.target_obs.color} 方块已对正，开始接近')
                return
            self.stop_robot_once()
            return

        angular = self.clamp_abs(-1.20 * error_norm, self.ALIGN_MIN_ANG, self.ALIGN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_approach_target(self):
        if self.front_dist <= self.EMERGENCY_STOP_DIST:
            self.request_shutdown('前方过近，紧急停车')
            return

        if not self.target_visible or self.target_obs is None or self.image_width is None:
            if self.front_dist <= self.CLOSE_FRONT_DIST:
                self.capture_decision_close = True
                self.stop_robot_once()
                self.set_state('CAPTURE_CUBE', '近距离丢目标，按近距离捕获')
                return
            if self.target_recently_seen():
                self.publish_cmd(0.0, self.last_target_dir * self.RECOVER_ANG)
                return
            self.lock_color = None
            self.stop_robot_once()
            self.set_state('SEARCH_SWEEP', '接近时丢目标，重新搜索')
            return

        error = self.target_error_pixels()
        center_x = self.image_width / 2.0
        error_norm = error / center_x

        if self.close_enough_for_capture():
            self.capture_decision_close = True
            self.stop_robot_once()
            self.set_state(
                'CAPTURE_CUBE',
                f'进入捕获距离 color={self.target_obs.color} pitch={self.target_obs.hole_pitch:.1f} h={self.target_obs.bbox_h}',
            )
            return

        if abs(error) >= self.REACQUIRE_PIXEL_TOL:
            self.stop_robot_once()
            self.set_state('ALIGN_TARGET', '偏差过大，先重新对准')
            return

        if abs(error) > self.APPROACH_ROTATE_ONLY_PX:
            angular = self.clamp_abs(-1.05 * error_norm, self.ALIGN_MIN_ANG, self.ALIGN_MAX_ANG)
            self.publish_cmd(0.0, angular)
            return

        near_by_box = self.target_obs.bbox_h >= 130 or self.target_obs.bbox_w >= 120
        if near_by_box or self.target_obs.hole_pitch >= 15.0 or abs(error) > self.ALIGN_PIXEL_TOL * 1.6:
            linear = self.APPROACH_SLOW_SPEED
        else:
            linear = self.APPROACH_FAST_SPEED

        if abs(error) <= self.ALIGN_PIXEL_TOL:
            angular = 0.0
        else:
            angular = self.clamp_abs(-0.85 * error_norm, 0.08, 0.42)

        self.publish_cmd(linear, angular)

    def handle_capture_cube(self):
        elapsed = time.monotonic() - self.state_enter_time
        self.stop_robot_once()

        if self.current_job_color is None:
            self.current_job_color = self.lock_color or self.last_target_color
            self.arm.capture()
            print(f'[JOB] 捕获颜色 = {self.current_job_color}')

        if elapsed < self.ARM_SETTLE_SEC:
            return

        self.turn_target_yaw = self.desired_delivery_face_yaw(self.current_job_color)
        self.set_state('TURN_TO_DELIVERY', f'开始转向 {self.current_job_color} 对应投放侧')

    def handle_turn_to_delivery(self):
        desired_yaw = self.turn_target_yaw
        desired_label = self.desired_delivery_face_label(self.current_job_color)

        tag_locked = False
        if self.marker_recently_seen() and self.marker_label == desired_label and self.image_width is not None:
            marker_err = self.marker_cx - self.image_width / 2.0
            if abs(marker_err) <= 28:
                tag_locked = True
            else:
                marker_norm = marker_err / (self.image_width / 2.0)
                angular = self.clamp_abs(-1.25 * marker_norm, 0.20, self.TURN_TO_SIDE_MAX_ANG)
                self.publish_cmd(0.0, angular)
                return

        yaw_err = self.normalize_angle(desired_yaw - self.local_yaw)
        if tag_locked or abs(yaw_err) <= math.radians(4.0):
            self.stop_robot_once()
            self.begin_travel_here()
            self.set_state('DRAG_TO_SIDE', f'朝向到位，开始倒车拖动 {self.current_job_color} 方块')
            return

        angular = self.clamp_abs(1.05 * yaw_err, 0.18, self.TURN_TO_SIDE_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_drag_to_side(self):
        dist = self.distance_from(self.travel_start)
        yaw_err = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        angular = self.clamp(-0.95 * yaw_err, -0.30, 0.30)

        if dist >= self.DRAG_DISTANCE_M:
            self.stop_robot_once()
            self.set_state('RELEASE_CUBE', f'倒车距离到达 {dist:.2f}m，准备抬臂释放')
            return

        self.publish_cmd(self.DELIVERY_REVERSE_SPEED, angular)

    def handle_release_cube(self):
        elapsed = time.monotonic() - self.state_enter_time
        self.stop_robot_once()

        if elapsed < 0.05:
            self.arm.release()
            return

        if elapsed < self.RELEASE_SETTLE_SEC:
            return

        self.arm.stow()
        self.begin_travel_here()
        self.set_state('RETURN_TO_CENTER', '开始回到中间区域')

    def handle_return_to_center(self):
        dist = self.distance_from(self.travel_start)
        yaw_err = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        angular = self.clamp(-0.95 * yaw_err, -0.30, 0.30)

        if dist >= self.RETURN_DISTANCE_M:
            self.stop_robot_once()
            self.set_state('RESET_HEADING', '回中完成，准备恢复初始朝向')
            return

        self.publish_cmd(self.RETURN_FORWARD_SPEED, angular)

    def handle_reset_heading(self):
        yaw_err = self.normalize_angle(0.0 - self.local_yaw)

        if abs(yaw_err) <= math.radians(4.0):
            moved_color = self.current_job_color
            self.stop_robot_once()
            self.lock_color = None
            self.current_job_color = None
            self.capture_decision_close = False
            self.set_state('SEARCH_SWEEP', f'{moved_color} 方块流程完成，继续搜索下一个')
            return

        angular = self.clamp_abs(1.05 * yaw_err, 0.18, self.TURN_TO_SIDE_MAX_ANG)
        self.publish_cmd(0.0, angular)

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
            self.arm.stow()
            self.stop_robot_once()
            self.set_state('SEARCH_SWEEP', '传感器准备完成，开始原地搜索')
            return

        if self.state == 'SEARCH_SWEEP':
            self.handle_search_sweep()
        elif self.state == 'ALIGN_TARGET':
            self.handle_align_target()
        elif self.state == 'APPROACH_TARGET':
            self.handle_approach_target()
        elif self.state == 'CAPTURE_CUBE':
            self.handle_capture_cube()
        elif self.state == 'TURN_TO_DELIVERY':
            self.handle_turn_to_delivery()
        elif self.state == 'DRAG_TO_SIDE':
            self.handle_drag_to_side()
        elif self.state == 'RELEASE_CUBE':
            self.handle_release_cube()
        elif self.state == 'RETURN_TO_CENTER':
            self.handle_return_to_center()
        elif self.state == 'RESET_HEADING':
            self.handle_reset_heading()
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

        marker_text = self.marker_label if self.marker_recently_seen() else 'none'
        lock_text = self.lock_color or 'none'
        job_text = self.current_job_color or 'none'

        if self.target_visible and self.target_obs is not None:
            err = self.target_error_pixels()
            print(
                f'[STATUS] {self.state} '
                f'color={self.target_obs.color} '
                f'err={err:.0f} '
                f'area={self.target_obs.area:.0f} '
                f'box={self.target_obs.bbox_w}x{self.target_obs.bbox_h} '
                f'holes={self.target_obs.holes} '
                f'pitch={self.target_obs.hole_pitch:.1f} '
                f'conf={self.target_obs.color_conf:.0f} '
                f'lock={lock_text} '
                f'job={job_text} '
                f'tag={marker_text} '
                f'front={self.front_dist:.2f}'
            )
        else:
            print(
                f'[STATUS] {self.state} color=none lock={lock_text} job={job_text} '
                f'tag={marker_text} front={self.front_dist:.2f}'
            )

    def destroy_node(self):
        try:
            self.stop_robot_reliable(repeat=15, delay=0.03)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrackingCubeV5()

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
