#!/usr/bin/env python3
import math
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

try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None


# =========================
# Mission tuning
# =========================
TARGET_COLOR = "any"          # "any", "red", or "blue"
PREFER_BLUE = True

SEARCH_ANG = 0.176            # v12 1.1x speed, still slow enough to lock
SEARCH_CENTER_BAND_PX = 80
MIN_CENTER_FRAMES = 3
STARTUP_IGNORE_SEC = 0.8
SEARCH_LOCK_MAX_JUMP_PX = 85
SEARCH_LOCK_MIN_AREA = 170.0
SEARCH_LOCK_MIN_BBOX_H = 12

CENTER_STOP_SEC = 1.0
BACKTRACK_AFTER_CENTER_SEC = 1.0
BACKTRACK_AFTER_CENTER_ANG = -0.10
FINE_ALIGN_LOST_RECOVER_SEC = 0.6
FINE_ALIGN_SEARCH_ANG = 0.07
FINE_ALIGN_PIXEL_TOL = 28
FINE_ALIGN_HOLD_FRAMES = 2
FINE_ALIGN_MIN_ANG = 0.025
FINE_ALIGN_MAX_ANG = 0.09

APPROACH_SPEED = 0.050
EXTRA_FORWARD_SPEED = 0.044
EXTRA_FORWARD_AFTER_LOST_SEC = 1.0
APPROACH_LOST_GRACE_FRAMES = 6
MIN_APPROACH_TIME_BEFORE_LOST_GRAB_SEC = 0.8
CAPTURE_CLOSE_DISTANCE_CM = 20.0
CAPTURE_CLOSE_BBOX_H_PX = 125

TURN_AFTER_GRAB_TARGET_DEG = 155.0
TURN_RETURN_TARGET_DEG = 180.0
TURN_AFTER_GRAB_MAX_ANG = 0.168
TURN_RETURN_MAX_ANG = 0.220
TURN_180_TOL_DEG = 4.0

FORWARD_AFTER_TURN_SPEED = 0.077
FORWARD_AFTER_TURN_SEC = 3.0
BACKUP_SPEED = -0.26
BACKUP_AFTER_RELEASE_SEC = 1.0

EMERGENCY_STOP_CM = 3.0
EMERGENCY_STOP_M = EMERGENCY_STOP_CM / 100.0


# =========================
# Servo tuning
# =========================
SERVO_GPIO_BCM = 12
SERVO_PWM_HZ = 50
SERVO_START_DUTY = 0.0
SERVO_UP_DUTY = 9.0
SERVO_DOWN_DUTY = 4.2
SERVO_SETTLE_SEC = 0.55


# =========================
# Detection tuning
# =========================
MIN_CONTOUR_AREA = 120.0
MIN_BBOX_W = 10
MIN_BBOX_H = 10
MIN_ASPECT = 0.55
MAX_ASPECT = 1.75
MIN_FILL_RATIO = 0.15
MIN_EXTENT = 0.13
MIN_SOLIDITY = 0.40
MIN_CENTER_Y_RATIO = 0.12
RED_MIN_HOLES_REQUIRED = 1
BLUE_MIN_HOLES_REQUIRED = 0

REAL_CUBE_SIZE_CM = 5.0
FOCAL_LENGTH_PX = 520.0

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
    distance_cm: float = 0.0


class ServoHelper:
    def __init__(self):
        self.enabled = GPIO is not None
        self.servo = None
        if not self.enabled:
            print("[SERVO] RPi.GPIO not available, servo disabled")
            return
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_GPIO_BCM, GPIO.OUT)
        self.servo = GPIO.PWM(SERVO_GPIO_BCM, SERVO_PWM_HZ)
        self.servo.start(SERVO_START_DUTY)
        time.sleep(0.1)
        self.servo.ChangeDutyCycle(0.0)
        print(f"[SERVO] BCM {SERVO_GPIO_BCM}, {SERVO_PWM_HZ} Hz ready")

    def set_duty(self, duty):
        if not self.enabled or self.servo is None:
            print(f"[SERVO] duty={duty:.2f} skipped")
            return
        self.servo.ChangeDutyCycle(float(duty))
        time.sleep(SERVO_SETTLE_SEC)
        self.servo.ChangeDutyCycle(0.0)

    def servo_up(self):
        print(f"[SERVO] up duty={SERVO_UP_DUTY}")
        self.set_duty(SERVO_UP_DUTY)

    def servo_down(self):
        print(f"[SERVO] down duty={SERVO_DOWN_DUTY}")
        self.set_duty(SERVO_DOWN_DUTY)

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
        self.focal_length_px = FOCAL_LENGTH_PX

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

            moments = cv2.moments(cnt)
            if moments["m00"] == 0:
                continue
            points.append((float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])))

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

        if color == "blue":
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

    def estimate_distance_cm(self, obs):
        apparent_size_px = (obs.bbox_w + obs.bbox_h) / 2.0
        if apparent_size_px <= 0:
            return 0.0
        return float(REAL_CUBE_SIZE_CM * self.focal_length_px / apparent_size_px)

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
            if w > self.image_width * 0.45 or h > self.image_height * 0.60:
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
            min_holes = RED_MIN_HOLES_REQUIRED if color == "red" else BLUE_MIN_HOLES_REQUIRED
            if holes < min_holes:
                continue

            center_error = abs((x + w / 2.0) - self.image_width / 2.0)
            score = (
                1.7 * area
                + 420.0 * fill_ratio
                + 320.0 * extent
                + 240.0 * solidity
                + 2.2 * color_conf
                + 80.0 * min(holes, 12)
                + 3.0 * hole_pitch
                - 0.16 * center_error
            )

            obs = CubeObservation(
                color=color,
                cx=x + w / 2.0,
                cy=cy,
                area=area,
                bbox_x=x,
                bbox_y=y,
                bbox_w=w,
                bbox_h=h,
                holes=holes,
                hole_pitch=hole_pitch,
                fill_ratio=fill_ratio,
                extent=extent,
                solidity=solidity,
                color_conf=color_conf,
                score=score,
            )
            obs.distance_cm = self.estimate_distance_cm(obs)
            if best is None or obs.score > best.score:
                best = obs

        return best

    def detect(self, frame):
        self.image_height, self.image_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        red_best = None
        blue_best = None
        if TARGET_COLOR in ("any", "red"):
            red_best = self.detect_best_for_color("red", self.build_red_mask(hsv, frame), frame, hsv, gray)
        if TARGET_COLOR in ("any", "blue"):
            blue_best = self.detect_best_for_color("blue", self.build_blue_mask(hsv, frame), frame, hsv, gray)

        if TARGET_COLOR == "red":
            return red_best, red_best, blue_best
        if TARGET_COLOR == "blue":
            return blue_best, red_best, blue_best
        if red_best is not None and blue_best is not None:
            chosen = blue_best if PREFER_BLUE and blue_best.score >= red_best.score else red_best
        else:
            chosen = blue_best if blue_best is not None else red_best
        return chosen, red_best, blue_best


class SimpleCubeMissionV15(Node):
    def __init__(self):
        super().__init__("simple_cube_mission_v15")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.control_timer = self.create_timer(CONTROL_DT, self.control_loop)
        self.status_timer = self.create_timer(STATUS_DT, self.status_loop)

        self.detector = CubeDetector()
        self.servo = ServoHelper()

        self.has_scan = False
        self.has_odom = False
        self.has_image = False
        self.front_dist = float("inf")
        self.image_width = None
        self.image_height = None
        self.image_logged = False

        self.world_yaw = 0.0
        self.init_yaw = None
        self.local_yaw = 0.0
        self.turn_target_yaw = 0.0

        self.target_visible = False
        self.target_obs = None
        self.red_obs = None
        self.blue_obs = None
        self.center_seen_frames = 0
        self.grab_color = None
        self.detect_error_count = 0
        self.approach_lost_frames = 0
        self.last_seen_target = None
        self.completed_cycles = 0
        self.lock_color = None
        self.lock_cx = None
        self.lock_frames = 0

        self.state = "WAIT_FOR_DATA"
        self.state_enter_time = time.monotonic()
        self.shutdown_requested = False
        self.shutdown_count = 0
        self.wait_status_last = None

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        print("[BOOT] SimpleCubeMission v15")
        print("[MODE] compressed image only, no imports from old versions")
        print("[FLOW] search -> stop 1s -> backtrack turn 1s -> fine align -> approach")
        print("[FLOW] only close target loss triggers forward 1.0s and servo down")
        print(f"[TARGET] TARGET_COLOR={TARGET_COLOR} PREFER_BLUE={PREFER_BLUE}")
        print(f"[SERVO] UP={SERVO_UP_DUTY} DOWN={SERVO_DOWN_DUTY} BCM={SERVO_GPIO_BCM}")
        print("[CMD] type H then Enter to stop and exit")

    def ready(self):
        return self.has_scan and self.has_odom and self.has_image

    def state_age(self):
        return time.monotonic() - self.state_enter_time

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
        if not rclpy.ok():
            return
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_robot_once(self):
        self.publish_cmd(0.0, 0.0)

    def stop_robot_reliable(self, repeat=12, delay=0.025):
        if not rclpy.ok():
            return
        for _ in range(repeat):
            self.stop_robot_once()
            time.sleep(delay)

    def set_state(self, new_state, text=""):
        if self.state == new_state:
            return
        self.state = new_state
        self.state_enter_time = time.monotonic()
        if new_state == "SEARCH":
            self.center_seen_frames = 0
            self.grab_color = None
            self.approach_lost_frames = 0
            self.last_seen_target = None
            self.lock_color = None
            self.lock_cx = None
            self.lock_frames = 0
        if new_state in ("CENTER_STOP", "BACKTRACK_AFTER_CENTER", "FINE_ALIGN"):
            self.center_seen_frames = 0
        if new_state == "APPROACH":
            self.approach_lost_frames = 0
        if text:
            print(f"[STATE] {new_state} | {text}")
        else:
            print(f"[STATE] {new_state}")

    def start_turn_relative(self, radians_delta, state_name):
        self.turn_target_yaw = self.normalize_angle(self.local_yaw + radians_delta)
        self.set_state(state_name, f"target_yaw={self.turn_target_yaw:.2f}")

    def request_shutdown(self, reason):
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.shutdown_count = 0
        print(f"[STOP] {reason}")

    def console_loop(self):
        while True:
            try:
                cmd = input().strip().lower()
            except Exception:
                return
            if cmd == "h":
                self.request_shutdown("manual emergency stop")
                return

    def scan_callback(self, msg):
        ranges = list(msg.ranges)
        n = len(ranges)
        if n == 0:
            return
        vals = list(ranges[0:min(10, n)]) + list(ranges[max(0, n - 10):n])
        good = [x for x in vals if math.isfinite(x) and x > 0.04]
        raw_front = min(good) if good else float("inf")
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
                raise ValueError("decoded frame is None")
        except Exception as exc:
            print(f"[IMAGE] compressed decode failed: {exc}")
            self.target_obs = None
            self.target_visible = False
            return

        self.image_height, self.image_width = frame.shape[:2]
        if not self.image_logged:
            print(f"[IMAGE] compressed size={self.image_width}x{self.image_height}")
            self.image_logged = True

        if self.state in ("WAIT_FOR_DATA", "SEARCH", "CENTER_STOP", "BACKTRACK_AFTER_CENTER", "FINE_ALIGN", "APPROACH"):
            try:
                chosen, red_obs, blue_obs = self.detector.detect(frame)
                self.detect_error_count = 0
            except Exception as exc:
                self.detect_error_count += 1
                print(f"[VISION] detect failed count={self.detect_error_count}: {exc}")
                chosen, red_obs, blue_obs = None, None, None

            self.target_obs = chosen
            self.red_obs = red_obs
            self.blue_obs = blue_obs
            self.target_visible = chosen is not None
        else:
            self.target_obs = None
            self.red_obs = None
            self.blue_obs = None
            self.target_visible = False

    def target_error_pixels(self):
        if not self.target_visible or self.target_obs is None or self.image_width is None:
            return None
        return self.target_obs.cx - self.image_width / 2.0

    def target_centered(self):
        err = self.target_error_pixels()
        return err is not None and abs(err) <= SEARCH_CENTER_BAND_PX

    def target_good_enough_to_lock(self, obs):
        if obs is None:
            return False
        if obs.area < SEARCH_LOCK_MIN_AREA or obs.bbox_h < SEARCH_LOCK_MIN_BBOX_H:
            return False
        if obs.color == "red":
            if obs.holes < 1:
                return False
            if obs.area < 220.0 and obs.bbox_h < 18:
                return False
        else:
            if obs.area < 150.0 or obs.bbox_h < 12:
                return False
            if obs.holes <= 0 and obs.area < 260.0 and obs.bbox_h < 18:
                return False
        return True

    def update_search_lock(self):
        if not self.target_visible or self.target_obs is None:
            self.lock_color = None
            self.lock_cx = None
            self.lock_frames = 0
            return 0
        if not self.target_centered() or not self.target_good_enough_to_lock(self.target_obs):
            self.lock_color = None
            self.lock_cx = None
            self.lock_frames = 0
            return 0

        if (
            self.lock_color == self.target_obs.color
            and self.lock_cx is not None
            and abs(self.target_obs.cx - self.lock_cx) <= SEARCH_LOCK_MAX_JUMP_PX
        ):
            self.lock_frames += 1
            self.lock_cx = 0.65 * self.lock_cx + 0.35 * self.target_obs.cx
        else:
            self.lock_color = self.target_obs.color
            self.lock_cx = self.target_obs.cx
            self.lock_frames = 1
        return self.lock_frames

    def remember_target_for_capture(self):
        if not self.target_visible or self.target_obs is None:
            return
        self.last_seen_target = {
            "color": self.target_obs.color,
            "bbox_h": float(self.target_obs.bbox_h),
            "bottom_y": float(self.target_obs.bbox_y + self.target_obs.bbox_h),
            "distance_cm": float(self.target_obs.distance_cm),
            "time": time.monotonic(),
        }

    def last_target_was_close(self):
        if self.last_seen_target is None or self.image_height is None:
            return False
        distance_cm = self.last_seen_target["distance_cm"]
        bbox_h = self.last_seen_target["bbox_h"]
        close_by_distance = 0.1 < distance_cm <= CAPTURE_CLOSE_DISTANCE_CM
        close_by_size = bbox_h >= CAPTURE_CLOSE_BBOX_H_PX
        return close_by_distance or close_by_size

    def last_target_summary(self):
        if self.last_seen_target is None:
            return "last=none"
        return (
            f"last={self.last_seen_target['color']} "
            f"h={self.last_seen_target['bbox_h']:.0f} "
            f"bottom={self.last_seen_target['bottom_y']:.0f} "
            f"dist={self.last_seen_target['distance_cm']:.1f}cm"
        )

    def handle_search(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        if self.state_age() >= STARTUP_IGNORE_SEC:
            lock_frames = self.update_search_lock()
            if lock_frames >= MIN_CENTER_FRAMES:
                self.grab_color = self.target_obs.color
                self.remember_target_for_capture()
                self.stop_robot_once()
                self.set_state("CENTER_STOP", f"{self.grab_color} stable centered, stop before correction")
                return
            if lock_frames > 0:
                self.publish_cmd(0.0, SEARCH_ANG * 0.35)
                return
        else:
            self.lock_color = None
            self.lock_cx = None
            self.lock_frames = 0

        self.publish_cmd(0.0, SEARCH_ANG)

    def handle_center_stop(self):
        self.stop_robot_once()
        if self.state_age() >= CENTER_STOP_SEC:
            self.set_state("BACKTRACK_AFTER_CENTER", "undo spin overshoot for 1s")

    def handle_backtrack_after_center(self):
        if self.state_age() < BACKTRACK_AFTER_CENTER_SEC:
            self.publish_cmd(0.0, BACKTRACK_AFTER_CENTER_ANG)
            return
        self.stop_robot_once()
        self.set_state("FINE_ALIGN", "fine align after backtrack")

    def handle_fine_align(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        if not self.target_visible or self.target_obs is None:
            if self.state_age() < FINE_ALIGN_LOST_RECOVER_SEC:
                self.publish_cmd(0.0, FINE_ALIGN_SEARCH_ANG)
                return
            self.set_state("SEARCH", "target lost after fine-align recovery")
            return

        err = self.target_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) <= FINE_ALIGN_PIXEL_TOL:
            self.center_seen_frames += 1
            self.stop_robot_once()
            if self.center_seen_frames >= FINE_ALIGN_HOLD_FRAMES:
                self.grab_color = self.target_obs.color
                self.remember_target_for_capture()
                self.set_state("APPROACH", f"{self.grab_color} fine aligned, drive straight")
            return

        self.center_seen_frames = 0
        err_norm = err / max(self.image_width / 2.0, 1.0)
        angular = self.clamp_abs(-0.18 * err_norm, FINE_ALIGN_MIN_ANG, FINE_ALIGN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_approach(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        if not self.target_visible or self.target_obs is None:
            self.approach_lost_frames += 1
            if self.approach_lost_frames < APPROACH_LOST_GRACE_FRAMES:
                self.publish_cmd(APPROACH_SPEED * 0.7, 0.0)
                return
            if self.state_age() >= MIN_APPROACH_TIME_BEFORE_LOST_GRAB_SEC and self.last_target_was_close():
                self.set_state("EXTRA_FORWARD_AFTER_LOST", f"close target disappeared, {self.last_target_summary()}")
            else:
                self.stop_robot_once()
                self.set_state("SEARCH", f"far target lost, no grab, {self.last_target_summary()}")
            return

        self.approach_lost_frames = 0
        self.remember_target_for_capture()
        self.publish_cmd(APPROACH_SPEED, 0.0)

    def handle_extra_forward_after_lost(self):
        if self.state_age() < EXTRA_FORWARD_AFTER_LOST_SEC:
            self.publish_cmd(EXTRA_FORWARD_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.set_state("SERVO_DOWN", "lower arm")

    def handle_servo_down(self):
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_down()
        time.sleep(0.12)
        self.servo.servo_down()
        self.start_turn_relative(math.radians(TURN_AFTER_GRAB_TARGET_DEG), "TURN_180_AFTER_GRAB")

    def handle_turn_180_after_grab(self):
        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= math.radians(TURN_180_TOL_DEG):
            self.stop_robot_once()
            self.set_state("FORWARD_AFTER_TURN", "drive forward 3s")
            return
        angular = self.clamp_abs(0.65 * error, 0.04, TURN_AFTER_GRAB_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_forward_after_turn(self):
        if self.state_age() < FORWARD_AFTER_TURN_SEC:
            self.publish_cmd(FORWARD_AFTER_TURN_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.set_state("SERVO_UP_RELEASE", "raise arm")

    def handle_servo_up_release(self):
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_up()
        self.set_state("BACKUP_AFTER_RELEASE", "backup 1s")

    def handle_backup_after_release(self):
        if self.state_age() < BACKUP_AFTER_RELEASE_SEC:
            self.publish_cmd(BACKUP_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.start_turn_relative(math.radians(TURN_RETURN_TARGET_DEG), "TURN_180_RETURN")

    def handle_turn_180_return(self):
        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= math.radians(TURN_180_TOL_DEG):
            self.stop_robot_once()
            self.completed_cycles += 1
            self.servo.servo_up()
            self.set_state("SEARCH", f"continue searching, cycles={self.completed_cycles}")
            return
        angular = self.clamp_abs(0.75 * error, 0.05, TURN_RETURN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def control_loop(self):
        if self.shutdown_requested:
            self.stop_robot_once()
            self.shutdown_count += 1
            if self.shutdown_count >= 10:
                self.stop_robot_reliable()
                self.servo.cleanup()
                if rclpy.ok():
                    rclpy.shutdown()
            return

        if not self.ready():
            self.stop_robot_once()
            return

        if self.state == "WAIT_FOR_DATA":
            self.stop_robot_once()
            self.servo.servo_up()
            self.set_state("SEARCH", "data ready, arm up")
            return

        handlers = {
            "SEARCH": self.handle_search,
            "CENTER_STOP": self.handle_center_stop,
            "BACKTRACK_AFTER_CENTER": self.handle_backtrack_after_center,
            "FINE_ALIGN": self.handle_fine_align,
            "APPROACH": self.handle_approach,
            "EXTRA_FORWARD_AFTER_LOST": self.handle_extra_forward_after_lost,
            "SERVO_DOWN": self.handle_servo_down,
            "TURN_180_AFTER_GRAB": self.handle_turn_180_after_grab,
            "FORWARD_AFTER_TURN": self.handle_forward_after_turn,
            "SERVO_UP_RELEASE": self.handle_servo_up_release,
            "BACKUP_AFTER_RELEASE": self.handle_backup_after_release,
            "TURN_180_RETURN": self.handle_turn_180_return,
            "STOPPED": self.stop_robot_once,
        }
        handlers.get(self.state, self.stop_robot_once)()

    def status_loop(self):
        if not self.ready():
            wait_now = (self.has_scan, self.has_odom, self.has_image)
            if wait_now != self.wait_status_last:
                print(f"[WAIT] scan={self.has_scan} odom={self.has_odom} image={self.has_image}")
                self.wait_status_last = wait_now
            return

        target_text = "target=none"
        if self.target_visible and self.target_obs is not None:
            err = self.target_error_pixels()
            target_text = (
                f"target={self.target_obs.color} err={err:.0f} "
                f"bbox=({self.target_obs.bbox_w}x{self.target_obs.bbox_h}) "
                f"holes={self.target_obs.holes} dist={self.target_obs.distance_cm:.1f}cm "
                f"center_frames={self.center_seen_frames}"
            )

        print(
            f"[STATUS] {self.state} {target_text} grab={self.grab_color} "
            f"front={self.front_dist:.3f}m yaw={math.degrees(self.local_yaw):.0f} "
            f"lost={self.approach_lost_frames} cycles={self.completed_cycles} "
            f"errors={self.detect_error_count} image=compressed {self.image_width}x{self.image_height}"
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
    node = SimpleCubeMissionV15()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.request_shutdown("KeyboardInterrupt")
    finally:
        try:
            if rclpy.ok():
                node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
