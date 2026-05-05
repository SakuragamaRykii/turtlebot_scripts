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


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DETECT_AT_REFERENCE_SIZE = True

MIN_RED_AREA = 2200
MIN_BLUE_AREA = 1800
MAX_AREA = 120000

SPEED_SCALE = 1.4
SEARCH_ANG = 0.22 * SPEED_SCALE
ALIGN_MIN_ANG = 0.04 * SPEED_SCALE
ALIGN_MAX_ANG = 0.20 * SPEED_SCALE
RECOVER_ANG = 0.12 * SPEED_SCALE
APPROACH_FAST_SPEED = 0.05 * SPEED_SCALE
APPROACH_SLOW_SPEED = 0.024 * SPEED_SCALE
BACKUP_SPEED = -0.12

CUBE_ALIGN_PIXEL_TOL = 18
ZONE_ALIGN_PIXEL_TOL = 28
CENTER_HOLD_FRAMES = 3
CONFIRM_FRAMES = 2
LOST_TARGET_TIMEOUT = 0.55

GRAB_EXTRA_FORWARD_SEC = 1.0
GRAB_EXTRA_FORWARD_SPEED = 0.045
BACKUP_AFTER_DROP_SEC = 1.0

CUBE_LIDAR_GRAB_CM = 6.0
CUBE_LIDAR_GRAB_M = CUBE_LIDAR_GRAB_CM / 100.0
ZONE_STOP_DISTANCE_CM = 12.0
ZONE_STOP_DISTANCE_M = ZONE_STOP_DISTANCE_CM / 100.0
EMERGENCY_STOP_CM = 3.0
EMERGENCY_STOP_M = EMERGENCY_STOP_CM / 100.0

CLOSE_CUBE_BBOX_H_PX = 230
LOST_CLOSE_BBOX_H_PX = 180
ZONE_STOP_BBOX_H_PX = 310

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
    score: float
    missing: int


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


def clean_mask(mask):
    mask = cv2.medianBlur(mask, 5)
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def build_red_mask(hsv, bgr):
    lower1 = np.array([0, 110, 70], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 110, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    hsv_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )

    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)

    bgr_mask = np.zeros_like(hsv_mask)
    bgr_mask[(r > 110) & (r > g + 35) & (r > b + 35)] = 255
    return clean_mask(cv2.bitwise_and(hsv_mask, bgr_mask))


def build_blue_mask(hsv, bgr):
    lower = np.array([85, 35, 40], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)

    bgr_mask = np.zeros_like(hsv_mask)
    bgr_mask[(b > 80) & (b > r + 15) & (b > g - 5)] = 255
    return clean_mask(cv2.bitwise_and(hsv_mask, bgr_mask))


def find_square(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.035 * perimeter, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < 45 or h < 45:
            continue

        aspect = w / float(h)
        if aspect < 0.72 or aspect > 1.38:
            continue

        rect_area = w * h
        fill_ratio = area / float(rect_area)
        if fill_ratio < 0.42:
            continue

        score = area * fill_ratio
        if best is None or score > best[4]:
            best = (x, y, w, h, score)

    return best


class StableTracker:
    def __init__(self, alpha=0.14, max_missing=8, max_jump=70):
        self.box = None
        self.missing = 0
        self.alpha = alpha
        self.max_missing = max_missing
        self.max_jump = max_jump

    def update(self, new_box):
        if new_box is None:
            self.missing += 1
            if self.missing > self.max_missing:
                self.box = None
            return self.box

        if self.box is None:
            self.box = new_box
            self.missing = 0
            return self.box

        ox, oy, ow, oh, _ = self.box
        nx, ny, nw, nh, ns = new_box

        old_cx = ox + ow / 2
        old_cy = oy + oh / 2
        new_cx = nx + nw / 2
        new_cy = ny + nh / 2
        distance = ((old_cx - new_cx) ** 2 + (old_cy - new_cy) ** 2) ** 0.5

        if distance > self.max_jump:
            self.missing += 1
            if self.missing > self.max_missing:
                self.box = None
            return self.box

        x = int(ox * (1 - self.alpha) + nx * self.alpha)
        y = int(oy * (1 - self.alpha) + ny * self.alpha)
        w = int(ow * (1 - self.alpha) + nw * self.alpha)
        h = int(oh * (1 - self.alpha) + nh * self.alpha)

        self.box = (x, y, w, h, ns)
        self.missing = 0
        return self.box


class CubeDetector:
    def __init__(self):
        self.red_tracker = StableTracker(alpha=0.10, max_missing=10, max_jump=55)
        self.blue_tracker = StableTracker(alpha=0.16, max_missing=10, max_jump=80)

    def detect(self, frame):
        original_h, original_w = frame.shape[:2]
        work = frame
        scale_x = 1.0
        scale_y = 1.0

        if DETECT_AT_REFERENCE_SIZE and (original_w != FRAME_WIDTH or original_h != FRAME_HEIGHT):
            work = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
            scale_x = original_w / float(FRAME_WIDTH)
            scale_y = original_h / float(FRAME_HEIGHT)

        work = cv2.GaussianBlur(work, (5, 5), 0)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

        red_box = self.red_tracker.update(find_square(build_red_mask(hsv, work), MIN_RED_AREA))
        blue_box = self.blue_tracker.update(find_square(build_blue_mask(hsv, work), MIN_BLUE_AREA))

        blue_obs = self.box_to_obs('blue', blue_box, self.blue_tracker.missing, scale_x, scale_y)
        red_obs = self.box_to_obs('red', red_box, self.red_tracker.missing, scale_x, scale_y)

        if blue_obs is not None:
            return blue_obs
        return red_obs

    def box_to_obs(self, color, box, missing, scale_x, scale_y):
        if box is None:
            return None
        x, y, w, h, score = box
        mx = int(round(x * scale_x))
        my = int(round(y * scale_y))
        mw = int(round(w * scale_x))
        mh = int(round(h * scale_y))
        return CubeObservation(
            color=color,
            cx=float(mx + mw / 2.0),
            cy=float(my + mh / 2.0),
            area=float(mw * mh),
            bbox_x=mx,
            bbox_y=my,
            bbox_w=mw,
            bbox_h=mh,
            score=float(score),
            missing=int(missing),
        )


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


class TrackingCubeV7(Node):
    def __init__(self):
        super().__init__('tracking_cube_v7')

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
        self.image_width = None
        self.image_height = None
        self.image_logged = False

        self.world_yaw = 0.0
        self.init_yaw = None
        self.local_yaw = 0.0
        self.turn_target_yaw = 0.0

        self.target_visible = False
        self.target_obs: Optional[CubeObservation] = None
        self.target_seen_frames = 0
        self.centered_frames = 0
        self.last_target_seen_time = 0.0
        self.last_target_dir = 1.0
        self.last_cube_bbox_h = 0

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

        print('[BOOT] TrackingCube v7')
        print('[IMAGE] using /camera/image_raw/compressed only, same as v5')
        print(f'[PARAM] cube_grab={CUBE_LIDAR_GRAB_CM:.1f}cm zone_stop={ZONE_STOP_DISTANCE_CM:.1f}cm')
        print(f'[SERVO] boot will call servo_up once, BCM{SERVO_GPIO_BCM}, 50Hz')
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

        if msg.angle_increment > 0.0 and math.isfinite(msg.angle_increment):
            count = max(2, int(round(math.radians(8.0) / msg.angle_increment)))
        else:
            count = 8

        indices = list(range(0, min(count + 1, n))) + list(range(max(0, n - count), n))
        vals = [ranges[i] for i in indices if math.isfinite(ranges[i]) and ranges[i] > 0.02]
        raw_front = min(vals) if vals else float('inf')

        if not self.has_scan:
            self.front_dist = raw_front
        else:
            alpha = 0.55
            self.front_dist = alpha * self.front_dist + (1.0 - alpha) * raw_front
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
            print(f'[IMAGE] compressed decode failed: {exc}')
            self.clear_cube()
            self.clear_zone_marker()
            return

        self.image_height, self.image_width = frame.shape[:2]
        if not self.image_logged:
            print(f'[IMAGE] compressed size={self.image_width}x{self.image_height}')
            self.image_logged = True

        if self.state in ('WAIT_FOR_DATA', 'SEARCH_CUBE', 'ALIGN_CUBE', 'APPROACH_CUBE'):
            self.update_cube(self.cube_detector.detect(frame))
        else:
            self.clear_cube()

        if self.destination_marker_id is not None:
            self.update_zone_marker(self.marker_detector.detect(frame, self.destination_marker_id))
        else:
            self.clear_zone_marker()

    def update_cube(self, obs):
        if obs is None:
            self.clear_cube()
            return

        self.target_visible = True
        self.target_obs = obs
        self.target_seen_frames += 1
        self.last_target_seen_time = time.monotonic()
        self.last_cube_bbox_h = obs.bbox_h

        err = obs.cx - self.image_width / 2.0
        if abs(err) > 2.0:
            self.last_target_dir = -1.0 if err > 0.0 else 1.0

    def clear_cube(self):
        self.target_visible = False
        self.target_obs = None
        self.target_seen_frames = 0

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

    def cube_confirmed(self):
        return (
            self.target_visible
            and self.target_obs is not None
            and self.target_obs.missing <= 2
            and self.target_seen_frames >= CONFIRM_FRAMES
        )

    def cube_close_enough_to_grab(self):
        return self.front_dist <= CUBE_LIDAR_GRAB_M or self.last_cube_bbox_h >= CLOSE_CUBE_BBOX_H_PX

    def lost_cube_close_should_grab(self):
        return self.front_dist <= CUBE_LIDAR_GRAB_M + 0.04 or self.last_cube_bbox_h >= LOST_CLOSE_BBOX_H_PX

    def destination_for_color(self, color):
        if color == 'red':
            return RED_ZONE_MARKER_ID
        return BLUE_ZONE_MARKER_ID

    def handle_search_cube(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if self.cube_confirmed():
            self.carried_color = self.target_obs.color
            self.destination_marker_id = self.destination_for_color(self.carried_color)
            self.stop_robot_once()
            self.set_state('ALIGN_CUBE', f'found {self.carried_color}, dest_marker={self.destination_marker_id}')
            return

        self.publish_cmd(0.0, SEARCH_ANG)

    def handle_align_cube(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if not self.target_visible or self.target_obs is None:
            if self.lost_cube_close_should_grab():
                self.set_state('GRAB_FORWARD', 'cube lost close')
                return
            if self.cube_recently_seen():
                self.publish_cmd(0.0, self.last_target_dir * RECOVER_ANG)
                return
            self.set_state('SEARCH_CUBE', 'cube lost')
            return

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
        angular = self.clamp_abs(-0.30 * err_norm, ALIGN_MIN_ANG, ALIGN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_approach_cube(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if not self.target_visible or self.target_obs is None:
            if self.lost_cube_close_should_grab() or self.cube_recently_seen():
                self.set_state('GRAB_FORWARD', 'cube disappeared, forward before servo down')
                return
            self.set_state('SEARCH_CUBE', 'cube lost far away')
            return

        if self.cube_close_enough_to_grab():
            self.set_state('GRAB_FORWARD', 'cube close enough')
            return

        err = self.cube_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) > 120:
            self.stop_robot_once()
            self.set_state('ALIGN_CUBE', 'cube off center')
            return

        err_norm = err / max(self.image_width / 2.0, 1.0)
        linear = APPROACH_SLOW_SPEED if self.front_dist <= 0.18 or self.target_obs.bbox_h >= 160 else APPROACH_FAST_SPEED
        angular = 0.0 if abs(err) <= CUBE_ALIGN_PIXEL_TOL else self.clamp_abs(-0.24 * err_norm, 0.02 * SPEED_SCALE, 0.10 * SPEED_SCALE)
        self.publish_cmd(linear, angular)

    def handle_grab_forward(self):
        if time.monotonic() - self.state_enter_time < GRAB_EXTRA_FORWARD_SEC:
            self.publish_cmd(GRAB_EXTRA_FORWARD_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.set_state('SERVO_DOWN', 'lower servo')

    def handle_servo_down(self):
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_down()
        self.start_turn_relative(math.pi, 'TURN_TO_ZONE')

    def handle_turn_to_zone(self):
        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= math.radians(4.0):
            self.stop_robot_once()
            self.set_state('SEARCH_ZONE', f'search marker {self.destination_marker_id}')
            return
        angular = self.clamp_abs(0.75 * error, 0.05 * SPEED_SCALE, 0.18 * SPEED_SCALE)
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
        angular = 0.0 if abs(err) <= ZONE_ALIGN_PIXEL_TOL else self.clamp_abs(-0.22 * err_norm, 0.02 * SPEED_SCALE, 0.10 * SPEED_SCALE)
        self.publish_cmd(linear, angular)

    def handle_servo_up(self):
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_up()
        self.carried_color = None
        self.destination_marker_id = None
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
            self.set_state('SEARCH_CUBE', 'continue')
            return
        angular = self.clamp_abs(0.75 * error, 0.05 * SPEED_SCALE, 0.18 * SPEED_SCALE)
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
            cube_text = f'cube={self.target_obs.color} err={err:.0f} bbox=({self.target_obs.bbox_w}x{self.target_obs.bbox_h})'

        marker_text = 'marker=none'
        if self.zone_marker_visible and self.zone_marker is not None:
            err = self.zone_error_pixels()
            marker_text = f'marker={self.zone_marker.marker_id} err={err:.0f} h={self.zone_marker.bbox_h:.0f}'

        print(
            f'[STATUS] {self.state} {cube_text} {marker_text} '
            f'carry={self.carried_color} dest={self.destination_marker_id} '
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
    node = TrackingCubeV7()
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
