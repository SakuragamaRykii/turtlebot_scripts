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

try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None


SPEED_SCALE = 1.0
SEARCH_ANG = 0.20 * SPEED_SCALE
SEARCH_TRACK_MAX_ANG = 0.14 * SPEED_SCALE
ALIGN_MIN_ANG = 0.035 * SPEED_SCALE
ALIGN_MAX_ANG = 0.18 * SPEED_SCALE
RECOVER_ANG = 0.08 * SPEED_SCALE
APPROACH_FAST_SPEED = 0.05 * SPEED_SCALE
APPROACH_SLOW_SPEED = 0.026 * SPEED_SCALE
GRAB_EXTRA_FORWARD_SPEED = 0.04
BACKUP_SPEED = -0.12

CUBE_ALIGN_PIXEL_TOL = 20
SEARCH_LOCK_PIXEL_TOL = 150
SEARCH_TRACK_PIXEL_TOL = 285
APPROACH_ROTATE_ONLY_PX = 85
REACQUIRE_PIXEL_TOL = 115
ZONE_ALIGN_PIXEL_TOL = 28

CENTER_HOLD_FRAMES = 3
CONFIRM_FRAMES = 3
LOST_TARGET_TIMEOUT = 1.20

CUBE_LIDAR_GRAB_CM = 6.0
CUBE_LIDAR_GRAB_M = CUBE_LIDAR_GRAB_CM / 100.0
ZONE_STOP_DISTANCE_CM = 12.0
ZONE_STOP_DISTANCE_M = ZONE_STOP_DISTANCE_CM / 100.0
EMERGENCY_STOP_CM = 3.0
EMERGENCY_STOP_M = EMERGENCY_STOP_CM / 100.0

GRAB_EXTRA_FORWARD_SEC = 1.0
BACKUP_AFTER_DROP_SEC = 1.0
CLOSE_CUBE_BBOX_H_PX = 205
LOST_CLOSE_BBOX_H_PX = 150
ZONE_STOP_BBOX_H_PX = 310

MIN_CONTOUR_AREA = 120.0
MIN_BBOX_W = 10
MIN_BBOX_H = 10
MIN_ASPECT = 0.55
MAX_ASPECT = 1.75
MIN_FILL_RATIO = 0.15
MIN_EXTENT = 0.13
MIN_SOLIDITY = 0.40
MIN_CENTER_Y_RATIO = 0.12
REAL_CUBE_SIZE_CM = 5.0
FOCAL_LENGTH_PX = 520.0

RED_ZONE_MARKER_ID = 0
BLUE_ZONE_MARKER_ID = 23

SERVO_GPIO_BCM = 12
SERVO_PWM_HZ = 50
SERVO_START_DUTY = 0.0
SERVO_OPEN_DUTY = 7.5
SERVO_CLOSE_DUTY = 6.0
SERVO_DOWN_DUTY = 9.0
SERVO_UP_DUTY = 5.5
SERVO_SETTLE_SEC = 0.55

CONTROL_DT = 0.05
STATUS_DT = 0.8


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
    fill_ratio: float
    extent: float
    solidity: float
    color_conf: float
    score: float
    distance_cm: float


@dataclass
class MarkerObservation:
    marker_id: int
    cx: float
    cy: float
    bbox_w: float
    bbox_h: float
    area: float


class ServoHelper:
    def __init__(self):
        self.enabled = GPIO is not None
        self.servo = None
        if not self.enabled:
            print('[SERVO] RPi.GPIO not available, servo disabled')
            return
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_GPIO_BCM, GPIO.OUT)
        self.servo = GPIO.PWM(SERVO_GPIO_BCM, SERVO_PWM_HZ)
        self.servo.start(SERVO_START_DUTY)
        time.sleep(0.1)
        self.servo.ChangeDutyCycle(0.0)
        print(f'[SERVO] BCM {SERVO_GPIO_BCM}, {SERVO_PWM_HZ} Hz ready')

    def set_duty(self, duty):
        if not self.enabled or self.servo is None:
            print(f'[SERVO] duty={duty:.2f} skipped')
            return
        self.servo.ChangeDutyCycle(float(duty))
        time.sleep(SERVO_SETTLE_SEC)
        self.servo.ChangeDutyCycle(0.0)

    def servo_open(self):
        self.set_duty(SERVO_OPEN_DUTY)

    def servo_close(self):
        self.set_duty(SERVO_CLOSE_DUTY)

    def servo_down(self):
        self.set_duty(SERVO_DOWN_DUTY)

    def servo_up(self):
        self.set_duty(SERVO_UP_DUTY)

    def cleanup(self):
        if not self.enabled:
            return
        try:
            if self.servo is not None:
                self.servo.ChangeDutyCycle(0.0)
                self.servo.stop()
            GPIO.cleanup(SERVO_GPIO_BCM)
        except Exception:
            pass


class CubeDetector:
    def __init__(self):
        self.image_width = 0
        self.image_height = 0

    def build_red_mask(self, hsv, bgr):
        lower_red_1 = np.array([0, 75, 40], dtype=np.uint8)
        upper_red_1 = np.array([16, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([164, 75, 40], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2),
        )

        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)
        rgb_mask = np.zeros_like(hsv_mask)
        rgb_mask[(r >= 70) & (r > g + 22) & (r > b + 26)] = 255

        return cv2.bitwise_and(hsv_mask, rgb_mask)

    def build_blue_mask(self, hsv, bgr):
        lower_blue = np.array([80, 25, 20], dtype=np.uint8)
        upper_blue = np.array([150, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)
        rgb_mask = np.zeros_like(hsv_mask)
        rgb_mask[(b >= 40) & (b > r + 6) & (b > g - 22)] = 255

        return cv2.bitwise_or(hsv_mask, rgb_mask)

    def preprocess_mask(self, mask):
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask

    def detect_holes(self, gray_roi, shape_mask_roi):
        if gray_roi.size == 0 or shape_mask_roi.size == 0:
            return 0, 0.0

        inner_mask = cv2.erode(shape_mask_roi, np.ones((5, 5), np.uint8), iterations=1)
        valid_pixels = gray_roi[inner_mask > 0]
        if valid_pixels.size < 20:
            return 0, 0.0

        dark_threshold = int(np.clip(np.percentile(valid_pixels, 30), 18, 130))
        dark_mask = cv2.inRange(gray_roi, 0, dark_threshold)
        dark_mask = cv2.bitwise_and(dark_mask, inner_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_area = max(float(np.count_nonzero(inner_mask)), 1.0)
        points = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(4.0, 0.00015 * mask_area):
                continue
            if area > 0.06 * mask_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.14:
                continue
            m = cv2.moments(cnt)
            if m['m00'] == 0:
                continue
            points.append((float(m['m10'] / m['m00']), float(m['m01'] / m['m00'])))

        if len(points) < 2:
            return len(points), 0.0

        arr = np.array(points, dtype=np.float32)
        nearest = []
        for i in range(len(arr)):
            d = np.linalg.norm(arr - arr[i], axis=1)
            d = d[d > 1.0]
            if d.size > 0:
                nearest.append(float(np.min(d)))
        return len(points), float(np.median(nearest)) if nearest else 0.0

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
            hue_ratio = float(np.mean((hue >= 80) & (hue <= 150) & (sat >= 25)))
            dom_br = float(mean_b - mean_r)
            dom_bg = float(mean_b - mean_g)
            conf = 0.25 * mean_s + 0.35 * dom_br + 0.15 * dom_bg + 90.0 * hue_ratio
            ok = hue_ratio >= 0.18 and mean_v >= 30 and mean_b >= 40 and dom_br >= 5
            return ok, float(conf)

        hue_ratio = float(np.mean(((hue <= 16) | (hue >= 164)) & (sat >= 65)))
        dom_rb = float(mean_r - mean_b)
        dom_rg = float(mean_r - mean_g)
        conf = 0.25 * mean_s + 0.35 * dom_rb + 0.25 * dom_rg + 90.0 * hue_ratio
        ok = (
            hue_ratio >= 0.25
            and mean_s >= 65
            and mean_v >= 40
            and mean_r >= 65
            and dom_rb >= 20
            and dom_rg >= 16
        )
        return ok, float(conf)

    def estimate_distance_cm(self, bbox_w, bbox_h):
        apparent_size_px = (bbox_w + bbox_h) / 2.0
        if apparent_size_px <= 0:
            return 0.0
        return float(REAL_CUBE_SIZE_CM * FOCAL_LENGTH_PX / apparent_size_px)

    def detect_best_for_color(self, color, color_mask, frame, hsv, gray):
        mask = self.preprocess_mask(color_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < MIN_BBOX_W or h < MIN_BBOX_H:
                continue
            if w > self.image_width * 0.48 or h > self.image_height * 0.65:
                continue

            cy = y + h / 2.0
            if cy < self.image_height * MIN_CENTER_Y_RATIO:
                continue

            aspect = w / float(h)
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue

            bbox_area = float(w * h)
            hull = cv2.convexHull(contour)
            solidity = float(area / max(cv2.contourArea(hull), 1.0))
            if solidity < MIN_SOLIDITY:
                continue

            shifted = contour - np.array([[x, y]])
            shape_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(shape_mask, [shifted], -1, 255, thickness=-1)
            fill_ratio = float(np.count_nonzero(shape_mask) / max(bbox_area, 1.0))
            extent = float(area / max(bbox_area, 1.0))
            if fill_ratio < MIN_FILL_RATIO or extent < MIN_EXTENT:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
            if len(approx) < 4 or len(approx) > 12:
                continue

            roi_bgr = frame[y:y + h, x:x + w]
            roi_hsv = hsv[y:y + h, x:x + w]
            roi_gray = gray[y:y + h, x:x + w]
            color_ok, color_conf = self.color_signature(color, roi_bgr, roi_hsv, shape_mask)
            if not color_ok:
                continue

            holes, hole_pitch = self.detect_holes(roi_gray, shape_mask)
            center_error = abs((x + w / 2.0) - self.image_width / 2.0)
            hole_bonus = 80.0 * min(holes, 12) if holes >= 1 else 0.0
            score = (
                1.7 * area
                + 420.0 * fill_ratio
                + 320.0 * extent
                + 240.0 * solidity
                + 2.2 * color_conf
                + hole_bonus
                + 3.0 * hole_pitch
                - 0.16 * center_error
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
                fill_ratio=float(fill_ratio),
                extent=float(extent),
                solidity=float(solidity),
                color_conf=float(color_conf),
                score=float(score),
                distance_cm=self.estimate_distance_cm(w, h),
            )

            if best is None or obs.score > best.score:
                best = obs

        return best

    def detect(self, frame, locked_color=None):
        self.image_height, self.image_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        red_best = None
        blue_best = None
        if locked_color in (None, 'red'):
            red_best = self.detect_best_for_color('red', self.build_red_mask(hsv, frame), frame, hsv, gray)
        if locked_color in (None, 'blue'):
            blue_best = self.detect_best_for_color('blue', self.build_blue_mask(hsv, frame), frame, hsv, gray)

        if locked_color == 'red':
            return red_best
        if locked_color == 'blue':
            return blue_best
        if blue_best is not None:
            return blue_best
        return red_best


class MarkerDetector:
    def __init__(self):
        self.has_aruco = hasattr(cv2, 'aruco')
        self.detectors = []
        if not self.has_aruco:
            print('[MARKER] cv2.aruco not available')
            return

        names = [
            'DICT_4X4_50',
            'DICT_4X4_100',
            'DICT_5X5_100',
            'DICT_6X6_250',
            'DICT_ARUCO_ORIGINAL',
            'DICT_APRILTAG_16h5',
            'DICT_APRILTAG_25h9',
            'DICT_APRILTAG_36h10',
            'DICT_APRILTAG_36h11',
        ]
        for name in names:
            if not hasattr(cv2.aruco, name):
                continue
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
            if hasattr(cv2.aruco, 'ArucoDetector'):
                params = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(dictionary, params)
                self.detectors.append((detector, None))
            else:
                params = cv2.aruco.DetectorParameters_create()
                self.detectors.append((dictionary, params))
        print(f'[MARKER] loaded {len(self.detectors)} dictionaries')

    def detect(self, frame, target_id):
        if not self.has_aruco:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best = None
        for detector_or_dict, params in self.detectors:
            if hasattr(cv2.aruco, 'ArucoDetector'):
                corners, ids, _ = detector_or_dict.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, detector_or_dict, parameters=params)

            if ids is None:
                continue

            for idx, marker_id in enumerate(ids.flatten()):
                if int(marker_id) != int(target_id):
                    continue
                pts = corners[idx].reshape((4, 2))
                x_min = float(np.min(pts[:, 0]))
                x_max = float(np.max(pts[:, 0]))
                y_min = float(np.min(pts[:, 1]))
                y_max = float(np.max(pts[:, 1]))
                w = x_max - x_min
                h = y_max - y_min
                obs = MarkerObservation(
                    marker_id=int(marker_id),
                    cx=float(np.mean(pts[:, 0])),
                    cy=float(np.mean(pts[:, 1])),
                    bbox_w=float(w),
                    bbox_h=float(h),
                    area=float(max(w * h, 0.0)),
                )
                if best is None or obs.area > best.area:
                    best = obs
        return best


class TrackingCubeV9(Node):
    def __init__(self):
        super().__init__('tracking_cube_v9')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.control_timer = self.create_timer(CONTROL_DT, self.control_loop)
        self.status_timer = self.create_timer(STATUS_DT, self.status_loop)

        self.cube_detector = CubeDetector()
        self.marker_detector = MarkerDetector()
        self.servo = ServoHelper()

        self.has_scan = False
        self.has_odom = False
        self.has_image = False
        self.front_dist = float('inf')
        self.left_front_dist = float('inf')
        self.right_front_dist = float('inf')
        self.image_width = None
        self.image_height = None
        self.image_logged = False

        self.world_yaw = 0.0
        self.init_yaw = None
        self.local_yaw = 0.0
        self.turn_target_yaw = 0.0

        self.target_visible = False
        self.target_obs: Optional[CubeObservation] = None
        self.prev_raw_obs: Optional[CubeObservation] = None
        self.target_seen_frames = 0
        self.centered_frames = 0
        self.last_target_seen_time = 0.0
        self.last_target_dir = 1.0
        self.last_cube_bbox_h = 0
        self.locked_cube_color = None
        self.candidate_color = None

        self.zone_marker_visible = False
        self.zone_marker: Optional[MarkerObservation] = None
        self.zone_seen_frames = 0

        self.carried_color = None
        self.destination_marker_id = None
        self.state = 'WAIT_FOR_DATA'
        self.state_enter_time = time.monotonic()
        self.wait_status_last = None
        self.shutdown_requested = False
        self.shutdown_reason = ''
        self.shutdown_count = 0

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        print('[BOOT] TrackingCube v9')
        print('[IMAGE] using /camera/image_raw/compressed')
        print('[VISION] friend detector thresholds + V3-style stable tracking')
        print(f'[PARAM] cube_grab={CUBE_LIDAR_GRAB_CM:.1f}cm zone_stop={ZONE_STOP_DISTANCE_CM:.1f}cm')
        print('[CMD] type H then Enter to stop and exit')

    def ready(self):
        return self.has_scan and self.has_odom and self.has_image

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def clamp_abs(self, value, min_abs, max_abs):
        if abs(value) < 1e-9:
            return 0.0
        return math.copysign(min(max(abs(value), min_abs), max_abs), value)

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_robot_once(self):
        self.publish_cmd(0.0, 0.0)

    def stop_robot_reliable(self, repeat=15, delay=0.025):
        for _ in range(repeat):
            self.stop_robot_once()
            time.sleep(delay)

    def set_state(self, new_state, text=''):
        if self.state == new_state:
            return
        self.state = new_state
        self.state_enter_time = time.monotonic()
        self.centered_frames = 0
        if text:
            print(f'[STATE] {new_state} | {text}')
        else:
            print(f'[STATE] {new_state}')

    def start_turn_relative(self, radians_delta, next_state):
        self.turn_target_yaw = self.normalize_angle(self.local_yaw + radians_delta)
        self.set_state(next_state, f'target_yaw={self.turn_target_yaw:.2f}')

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
            except Exception:
                return
            if cmd == 'h':
                self.request_shutdown('manual emergency stop')
                return

    def scan_callback(self, msg):
        ranges = list(msg.ranges)
        n = len(ranges)
        if n == 0:
            return

        front_vals = list(ranges[0:min(10, n)]) + list(ranges[max(0, n - 10):n])
        left_vals = list(ranges[15:min(45, n)])
        right_vals = list(ranges[max(0, n - 45):max(0, n - 15)])

        def valid_min(vals):
            good = [x for x in vals if math.isfinite(x) and x > 0.04]
            return min(good) if good else float('inf')

        raw_front = valid_min(front_vals)
        raw_left = valid_min(left_vals)
        raw_right = valid_min(right_vals)

        if not self.has_scan:
            self.front_dist = raw_front
            self.left_front_dist = raw_left
            self.right_front_dist = raw_right
        else:
            alpha = 0.55
            self.front_dist = alpha * self.front_dist + (1.0 - alpha) * raw_front
            self.left_front_dist = alpha * self.left_front_dist + (1.0 - alpha) * raw_left
            self.right_front_dist = alpha * self.right_front_dist + (1.0 - alpha) * raw_right
        self.has_scan = True

    def odom_callback(self, msg):
        self.world_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        if self.init_yaw is None:
            self.init_yaw = self.world_yaw
        self.local_yaw = self.normalize_angle(self.world_yaw - self.init_yaw)
        self.has_odom = True

    def image_callback(self, msg):
        self.has_image = True
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('decoded frame is None')
        except Exception as exc:
            print(f'[IMAGE] decode failed: {exc}')
            self.clear_cube()
            self.clear_zone_marker()
            return

        self.image_height, self.image_width = frame.shape[:2]
        if not self.image_logged:
            print(f'[IMAGE] compressed size={self.image_width}x{self.image_height}')
            self.image_logged = True

        locked = self.locked_cube_color if self.state in ('ALIGN_CUBE', 'APPROACH_CUBE') else None
        if self.state in ('WAIT_FOR_DATA', 'SEARCH_CUBE', 'ALIGN_CUBE', 'APPROACH_CUBE'):
            self.update_cube(self.cube_detector.detect(frame, locked_color=locked))
        else:
            self.clear_cube()

        if self.destination_marker_id is not None:
            self.update_zone_marker(self.marker_detector.detect(frame, self.destination_marker_id))
        else:
            self.clear_zone_marker()

    def update_cube(self, obs):
        if obs is None:
            self.clear_cube(keep_recent=True)
            return

        stable = False
        if self.prev_raw_obs is not None and obs.color == self.prev_raw_obs.color:
            raw_jump = abs(obs.cx - self.prev_raw_obs.cx)
            area_ratio = obs.area / max(self.prev_raw_obs.area, 1.0)
            stable = raw_jump <= 120.0 and 0.30 <= area_ratio <= 3.2

        self.target_seen_frames = self.target_seen_frames + 1 if stable else 1
        self.prev_raw_obs = obs
        self.target_visible = True
        self.target_obs = obs
        self.last_target_seen_time = time.monotonic()
        self.last_cube_bbox_h = obs.bbox_h

        err = obs.cx - self.image_width / 2.0
        if abs(err) > 2.0:
            self.last_target_dir = -1.0 if err > 0.0 else 1.0

    def clear_cube(self, keep_recent=False):
        self.target_visible = False
        self.target_obs = None
        self.target_seen_frames = 0
        self.centered_frames = 0
        self.prev_raw_obs = None
        if not keep_recent:
            self.last_target_seen_time = 0.0

    def update_zone_marker(self, obs):
        if obs is None:
            self.clear_zone_marker()
            return
        self.zone_marker_visible = True
        self.zone_marker = obs
        self.zone_seen_frames += 1

    def clear_zone_marker(self):
        self.zone_marker_visible = False
        self.zone_marker = None
        self.zone_seen_frames = 0

    def cube_error_pixels(self):
        if not self.target_visible or self.target_obs is None or self.image_width is None:
            return None
        return self.target_obs.cx - self.image_width / 2.0

    def zone_error_pixels(self):
        if not self.zone_marker_visible or self.zone_marker is None or self.image_width is None:
            return None
        return self.zone_marker.cx - self.image_width / 2.0

    def cube_recently_seen(self):
        return (time.monotonic() - self.last_target_seen_time) <= LOST_TARGET_TIMEOUT

    def cube_lockworthy(self):
        if not self.target_visible or self.target_obs is None:
            return False
        if self.target_seen_frames < CONFIRM_FRAMES:
            return False
        err = self.cube_error_pixels()
        if err is None:
            return False
        return abs(err) <= SEARCH_LOCK_PIXEL_TOL

    def cube_close_enough_to_grab(self):
        return self.front_dist <= CUBE_LIDAR_GRAB_M or self.last_cube_bbox_h >= CLOSE_CUBE_BBOX_H_PX

    def lost_cube_close_should_grab(self):
        return self.front_dist <= CUBE_LIDAR_GRAB_M + 0.04 or self.last_cube_bbox_h >= LOST_CLOSE_BBOX_H_PX

    def destination_for_color(self, color):
        return RED_ZONE_MARKER_ID if color == 'red' else BLUE_ZONE_MARKER_ID

    def reset_cube_plan(self):
        self.locked_cube_color = None
        self.candidate_color = None
        if self.state in ('SEARCH_CUBE', 'ALIGN_CUBE', 'APPROACH_CUBE'):
            self.carried_color = None
            self.destination_marker_id = None

    def prepare_delivery_plan(self, color):
        self.locked_cube_color = color
        self.candidate_color = color
        self.carried_color = color
        self.destination_marker_id = self.destination_for_color(color)

    def handle_search_cube(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if self.cube_lockworthy():
            self.locked_cube_color = self.target_obs.color
            self.candidate_color = self.target_obs.color
            self.stop_robot_once()
            self.set_state('ALIGN_CUBE', f'locked {self.locked_cube_color}')
            return

        if self.target_visible and self.target_obs is not None and self.image_width is not None:
            err = self.cube_error_pixels()
            if err is not None and abs(err) <= SEARCH_TRACK_PIXEL_TOL:
                err_norm = err / max(self.image_width / 2.0, 1.0)
                angular = self.clamp_abs(-0.22 * err_norm, ALIGN_MIN_ANG, SEARCH_TRACK_MAX_ANG)
                self.publish_cmd(0.0, angular)
                return

        self.reset_cube_plan()
        self.publish_cmd(0.0, SEARCH_ANG)

    def handle_align_cube(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if not self.target_visible or self.target_obs is None:
            if self.lost_cube_close_should_grab() and self.candidate_color is not None:
                self.prepare_delivery_plan(self.candidate_color)
                self.set_state('GRAB_FORWARD', 'cube lost close')
                return
            if self.cube_recently_seen():
                self.publish_cmd(0.0, self.last_target_dir * RECOVER_ANG)
                return
            self.reset_cube_plan()
            self.set_state('SEARCH_CUBE', 'cube lost before grab')
            return

        self.candidate_color = self.target_obs.color
        self.locked_cube_color = self.target_obs.color
        err = self.cube_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) <= CUBE_ALIGN_PIXEL_TOL:
            self.centered_frames += 1
            self.stop_robot_once()
            if self.centered_frames >= CENTER_HOLD_FRAMES:
                self.set_state('APPROACH_CUBE', f'centered {self.target_obs.color}')
            return

        self.centered_frames = 0
        err_norm = err / max(self.image_width / 2.0, 1.0)
        angular = self.clamp_abs(-0.28 * err_norm, ALIGN_MIN_ANG, ALIGN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_approach_cube(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if not self.target_visible or self.target_obs is None:
            if self.lost_cube_close_should_grab() and self.candidate_color is not None:
                self.prepare_delivery_plan(self.candidate_color)
                self.set_state('GRAB_FORWARD', 'cube disappeared close')
                return
            if self.cube_recently_seen():
                self.stop_robot_once()
                return
            self.reset_cube_plan()
            self.set_state('SEARCH_CUBE', 'cube lost far away')
            return

        self.candidate_color = self.target_obs.color
        self.locked_cube_color = self.target_obs.color

        if self.cube_close_enough_to_grab():
            self.prepare_delivery_plan(self.target_obs.color)
            self.set_state('GRAB_FORWARD', 'cube close enough')
            return

        err = self.cube_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) >= REACQUIRE_PIXEL_TOL:
            self.stop_robot_once()
            self.set_state('ALIGN_CUBE', 'cube off center')
            return

        err_norm = err / max(self.image_width / 2.0, 1.0)
        if abs(err) > APPROACH_ROTATE_ONLY_PX:
            angular = self.clamp_abs(-0.30 * err_norm, ALIGN_MIN_ANG, ALIGN_MAX_ANG)
            self.publish_cmd(0.0, angular)
            return

        linear = APPROACH_SLOW_SPEED if self.front_dist <= 0.18 or self.target_obs.bbox_h >= 160 else APPROACH_FAST_SPEED
        angular = 0.0 if abs(err) <= CUBE_ALIGN_PIXEL_TOL else self.clamp_abs(-0.18 * err_norm, 0.02, 0.09)
        self.publish_cmd(linear, angular)

    def handle_grab_forward(self):
        if time.monotonic() - self.state_enter_time < GRAB_EXTRA_FORWARD_SEC:
            self.publish_cmd(GRAB_EXTRA_FORWARD_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.set_state('SERVO_DOWN', 'lower servo')

    def handle_servo_down(self):
        if self.destination_marker_id is None:
            self.prepare_delivery_plan(self.candidate_color or self.locked_cube_color or 'blue')
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_down()
        self.start_turn_relative(math.pi, 'TURN_TO_ZONE')

    def handle_turn_to_zone(self):
        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= math.radians(4.0):
            self.stop_robot_once()
            self.set_state('SEARCH_ZONE', f'search marker {self.destination_marker_id}')
            return
        angular = self.clamp_abs(0.75 * error, 0.05, 0.18)
        self.publish_cmd(0.0, angular)

    def handle_search_zone(self):
        if self.zone_marker_visible and self.zone_marker is not None and self.zone_seen_frames >= 2:
            self.stop_robot_once()
            self.set_state('ALIGN_ZONE', f'found marker {self.destination_marker_id}')
            return
        self.publish_cmd(0.0, SEARCH_ANG)

    def handle_align_zone(self):
        if not self.zone_marker_visible or self.zone_marker is None:
            self.publish_cmd(0.0, SEARCH_ANG * 0.65)
            return

        err = self.zone_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) <= ZONE_ALIGN_PIXEL_TOL:
            self.centered_frames += 1
            self.stop_robot_once()
            if self.centered_frames >= CENTER_HOLD_FRAMES:
                self.set_state('APPROACH_ZONE', f'centered marker {self.destination_marker_id}')
            return

        self.centered_frames = 0
        err_norm = err / max(self.image_width / 2.0, 1.0)
        angular = self.clamp_abs(-0.30 * err_norm, ALIGN_MIN_ANG, ALIGN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_approach_zone(self):
        if not self.zone_marker_visible or self.zone_marker is None:
            self.stop_robot_once()
            self.set_state('SEARCH_ZONE', 'marker lost')
            return

        if self.front_dist <= ZONE_STOP_DISTANCE_M or self.zone_marker.bbox_h >= ZONE_STOP_BBOX_H_PX:
            self.stop_robot_once()
            self.set_state('SERVO_UP', 'arrived destination')
            return

        err = self.zone_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) > 130:
            self.stop_robot_once()
            self.set_state('ALIGN_ZONE', 'marker off center')
            return

        err_norm = err / max(self.image_width / 2.0, 1.0)
        linear = APPROACH_SLOW_SPEED if self.front_dist <= ZONE_STOP_DISTANCE_M + 0.12 else APPROACH_FAST_SPEED
        angular = 0.0 if abs(err) <= ZONE_ALIGN_PIXEL_TOL else self.clamp_abs(-0.22 * err_norm, 0.02, 0.09)
        self.publish_cmd(linear, angular)

    def handle_servo_up(self):
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_up()
        self.carried_color = None
        self.destination_marker_id = None
        self.locked_cube_color = None
        self.candidate_color = None
        self.set_state('BACKUP_AFTER_DROP', 'back away')

    def handle_backup_after_drop(self):
        if time.monotonic() - self.state_enter_time < BACKUP_AFTER_DROP_SEC:
            self.publish_cmd(BACKUP_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.start_turn_relative(math.pi, 'TURN_AFTER_DROP')

    def handle_turn_after_drop(self):
        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= math.radians(4.0):
            self.stop_robot_once()
            self.set_state('SEARCH_CUBE', 'continue search')
            return
        angular = self.clamp_abs(0.75 * error, 0.05, 0.18)
        self.publish_cmd(0.0, angular)

    def control_loop(self):
        if self.shutdown_requested:
            self.stop_robot_once()
            self.shutdown_count += 1
            if self.shutdown_count >= 10:
                self.stop_robot_reliable()
                self.servo.cleanup()
                rclpy.shutdown()
            return

        if not self.ready():
            self.stop_robot_once()
            return

        if self.state == 'WAIT_FOR_DATA':
            self.stop_robot_once()
            self.servo.servo_up()
            self.set_state('SEARCH_CUBE', 'data ready')
            return

        handlers = {
            'SEARCH_CUBE': self.handle_search_cube,
            'ALIGN_CUBE': self.handle_align_cube,
            'APPROACH_CUBE': self.handle_approach_cube,
            'GRAB_FORWARD': self.handle_grab_forward,
            'SERVO_DOWN': self.handle_servo_down,
            'TURN_TO_ZONE': self.handle_turn_to_zone,
            'SEARCH_ZONE': self.handle_search_zone,
            'ALIGN_ZONE': self.handle_align_zone,
            'APPROACH_ZONE': self.handle_approach_zone,
            'SERVO_UP': self.handle_servo_up,
            'BACKUP_AFTER_DROP': self.handle_backup_after_drop,
            'TURN_AFTER_DROP': self.handle_turn_after_drop,
            'STOPPED': self.stop_robot_once,
        }
        handlers.get(self.state, self.stop_robot_once)()

    def status_loop(self):
        if not self.ready():
            wait_now = (self.has_scan, self.has_odom, self.has_image)
            if wait_now != self.wait_status_last:
                print(f'[WAIT] scan={self.has_scan} odom={self.has_odom} image={self.has_image}')
                self.wait_status_last = wait_now
            return

        cube_text = 'cube=none'
        if self.target_visible and self.target_obs is not None:
            err = self.cube_error_pixels()
            cube_text = (
                f'cube={self.target_obs.color} err={err:.0f} '
                f'bbox=({self.target_obs.bbox_w}x{self.target_obs.bbox_h}) '
                f'holes={self.target_obs.holes} dist={self.target_obs.distance_cm:.1f}cm '
                f'seen={self.target_seen_frames}'
            )

        marker_text = 'marker=none'
        if self.zone_marker_visible and self.zone_marker is not None:
            err = self.zone_error_pixels()
            marker_text = f'marker={self.zone_marker.marker_id} err={err:.0f} h={self.zone_marker.bbox_h:.0f}'

        print(
            f'[STATUS] {self.state} {cube_text} {marker_text} '
            f'lock={self.locked_cube_color} carry={self.carried_color} dest={self.destination_marker_id} '
            f'front={self.front_dist:.3f}m image={self.image_width}x{self.image_height}'
        )

    def destroy_node(self):
        try:
            self.stop_robot_reliable()
        except Exception:
            pass
        try:
            self.servo.cleanup()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrackingCubeV9()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.request_shutdown('KeyboardInterrupt')
        node.stop_robot_reliable()
    finally:
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
