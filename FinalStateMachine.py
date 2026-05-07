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
DEFAULT_TARGET_COLOR = "red"
PREFER_BLUE = False

# UPDATED: Centering parameters
SEARCH_ANG = 0.176
SEARCH_CENTER_BAND_PX = 95
MIN_CENTER_FRAMES = 2

# REMOVED: CENTER_STOP_SEC, BACKTRACK_AFTER_CENTER_SEC, BACKTRACK_AFTER_CENTER_ANG
# NEW: Active centering parameters
ACTIVE_CENTER_MIN_DURATION = 0.5     # Must be centered for this long
ACTIVE_CENTER_PIXEL_TOL = 25         # Tight tolerance for "centered"
ACTIVE_CENTER_MIN_ANG = 0.02         # Minimum rotation speed
ACTIVE_CENTER_MAX_ANG = 0.12         # Maximum rotation speed
ACTIVE_CENTER_GAIN = 0.22            # Proportional gain for centering
ACTIVE_CENTER_DECEL_ZONE = 0.3       # Normalized error threshold for deceleration
ACTIVE_CENTER_TIMEOUT = 8.0          # Maximum time in centering phase
ACTIVE_CENTER_LOST_RECOVER = 1.5     # Recovery search time if target lost
ACTIVE_CENTER_RECOVER_ANG = 0.06     # Recovery search speed

# NEW: Adaptive approach parameters
APPROACH_SPEED = 0.065
APPROACH_MAX_SPEED = 0.08
APPROACH_MIN_SPEED = 0.03
APPROACH_STEER_GAIN = 0.15           # How aggressively to correct heading
APPROACH_STEER_MAX = 0.12            # Maximum steering during approach
APPROACH_CENTER_TOLERANCE = 40       # Pixels - wider than fine align for stability
APPROACH_CENTER_HOLD = 3             # Frames to confirm re-centered
APPROACH_RECENTER_SPEED = 0.04       # Slower speed while re-centering
APPROACH_LOST_GRACE_FRAMES = 5
BLUE_APPROACH_LOST_GRACE_FRAMES = 14
EXTRA_FORWARD_SPEED = 0.055
EXTRA_FORWARD_AFTER_LOST_SEC = 1.5

LOST_CAPTURE_CLOSE_DISTANCE_CM = 24.0
LOST_CAPTURE_CLOSE_BBOX_H_PX = 105

MARKER_ALIGN_PIXEL_TOL = 24.0
MARKER_ALIGN_HOLD_FRAMES = 2
MARKER_TURN_SPEED = 0.220
MARKER_TURN_TIMEOUT_SEC = 35.0

DELIVERY_FORWARD_SPEED = 0.116
DELIVERY_FORWARD_SEC = 5.0
BACKUP_SPEED = -0.26
BACKUP_AFTER_RELEASE_SEC = 2.0

EMERGENCY_STOP_CM = 3.0
EMERGENCY_STOP_M = EMERGENCY_STOP_CM / 100.0

# Navigation tuning
NAVIGATE_TO_ZONE_SPEED = 0.08
NAVIGATE_TO_ZONE_ANG = 0.15
NAVIGATE_TARGET_DISTANCE_M = 0.3
ZONE_BOUNDARY_X = 0.0
POSITION_HISTORY_SEC = 2.0
MAX_CUBES = 4


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

RED_MIN_HOLES_REQUIRED = 3
BLUE_MIN_HOLES_REQUIRED = 1

RED_ZONE_MARKER_IDS = {0}
BLUE_ZONE_MARKER_IDS = {23}
RED_ZONE_EDGE_IDS = {7}
BLUE_ZONE_EDGE_IDS = {42}

SEARCH_START_DIRECTION = 1.0
SEARCH_AFTER_RELEASE_DIRECTION = -1.0

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


@dataclass
class MarkerObservation:
    marker_id: int
    cx: float
    cy: float
    error: float
    width: float


@dataclass
class RobotPosition:
    """Tracks robot position with timestamp for history"""
    x: float
    y: float
    yaw: float
    timestamp: float


class PositionTracker:
    """Tracks robot position using odometry integration"""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_odom_time = None
        self.last_linear_x = 0.0
        self.last_angular_z = 0.0
        self.position_history = []
        
    def update_from_velocity(self, linear_x: float, angular_z: float, current_time: float):
        if self.last_odom_time is None:
            self.last_odom_time = current_time
            return
            
        dt = current_time - self.last_odom_time
        if dt <= 0:
            return
            
        if abs(angular_z) < 1e-6:
            self.x += linear_x * math.cos(self.yaw) * dt
            self.y += linear_x * math.sin(self.yaw) * dt
        else:
            radius = linear_x / angular_z
            d_theta = angular_z * dt
            self.x += radius * (math.sin(self.yaw + d_theta) - math.sin(self.yaw))
            self.y -= radius * (math.cos(self.yaw + d_theta) - math.cos(self.yaw))
            self.yaw += d_theta
            
        self.yaw = self.normalize_angle(self.yaw)
        self.last_odom_time = current_time
        
        self.position_history.append(RobotPosition(
            x=self.x, y=self.y, yaw=self.yaw, timestamp=current_time
        ))
        
        cutoff = current_time - POSITION_HISTORY_SEC
        self.position_history = [p for p in self.position_history if p.timestamp > cutoff]
    
    def update_from_odometry(self, x: float, y: float, yaw: float, current_time: float):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.last_odom_time = current_time
        
        self.position_history.append(RobotPosition(
            x=self.x, y=self.y, yaw=self.yaw, timestamp=current_time
        ))
        
        cutoff = current_time - POSITION_HISTORY_SEC
        self.position_history = [p for p in self.position_history if p.timestamp > cutoff]
    
    def get_zone(self) -> str:
        if self.x > ZONE_BOUNDARY_X:
            return "RED_ZONE"
        elif self.x < -ZONE_BOUNDARY_X:
            return "BLUE_ZONE"
        return "BOUNDARY"
    
    def normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def get_distance_to(self, target_x: float, target_y: float) -> float:
        return math.sqrt((self.x - target_x)**2 + (self.y - target_y)**2)
    
    def get_angle_to(self, target_x: float, target_y: float) -> float:
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        return self.normalize_angle(target_angle - self.yaw)


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

    def detect(self, frame, target_color=DEFAULT_TARGET_COLOR):
        self.image_height, self.image_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        red_best = None
        blue_best = None
        if target_color in ("any", "red"):
            red_best = self.detect_best_for_color("red", self.build_red_mask(hsv, frame), frame, hsv, gray)
        if target_color in ("any", "blue"):
            blue_best = self.detect_best_for_color("blue", self.build_blue_mask(hsv, frame), frame, hsv, gray)

        if target_color == "red":
            return red_best, red_best, blue_best
        if target_color == "blue":
            return blue_best, red_best, blue_best
        if red_best is not None and blue_best is not None:
            chosen = blue_best if PREFER_BLUE and blue_best.score >= red_best.score else red_best
        else:
            chosen = blue_best if blue_best is not None else red_best
        return chosen, red_best, blue_best


class MarkerDetector:
    def __init__(self):
        self.available = hasattr(cv2, "aruco")
        self.use_new_api = False
        self.aruco_detector = None
        self.aruco_dict = None
        self.aruco_params = None
        if not self.available:
            print("[MARKER] cv2.aruco not available, marker detection disabled")
            return
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False
        print("[MARKER] ArUco DICT_4X4_50 ready")

    def detect(self, frame):
        if not self.available:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.use_new_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0:
            return None

        best_i = 0
        best_width = 0.0
        for i in range(len(ids)):
            pts = corners[i][0]
            width_px = float(np.linalg.norm(pts[0] - pts[1]))
            if width_px > best_width:
                best_width = width_px
                best_i = i

        pts = corners[best_i][0]
        marker_id = int(ids[best_i][0])
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        image_width = frame.shape[1]
        return MarkerObservation(
            marker_id=marker_id,
            cx=cx,
            cy=cy,
            error=cx - image_width / 2.0,
            width=best_width,
        )


class SimpleCubeMissionV30(Node):
    def __init__(self):
        super().__init__("simple_cube_mission_v30")

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
        self.marker_detector = MarkerDetector()
        self.servo = ServoHelper()
        
        # Position tracking
        self.position_tracker = PositionTracker()
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0

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
        self.marker_obs = None
        self.marker_visible = False
        self.current_zone = "UNKNOWN"
        self.current_target_color = DEFAULT_TARGET_COLOR
        self.delivery_marker_ids = set()
        self.marker_align_frames = 0
        self.center_seen_frames = 0
        
        # NEW: Active centering timing
        self.centered_start_time = None    # When we first achieved center
        self.centered_ready = False        # True once held for required duration
        
        # NEW: Approach adjust tracking
        self.approach_lost_frames = 0
        self.approach_recenter_frames = 0  # Frames cube has been re-centered
        self.approach_is_recentering = False  # Whether we're in re-center mode
        
        self.grab_color = None
        self.last_seen_target = None
        self.completed_cycles = 0
        self.detect_error_count = 0
        self.search_direction = SEARCH_START_DIRECTION
        
        # Mission tracking
        self.moved_cubes = 0
        self.navigate_target_x = None
        self.navigate_target_y = None
        self.navigate_target_yaw = None
        self.navigate_start_time = 0.0

        self.state = "WAIT_FOR_DATA"
        self.state_enter_time = time.monotonic()
        self.shutdown_requested = False
        self.shutdown_count = 0
        self.wait_status_last = None

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        print("[BOOT] SimpleCubeMission v30 - Adaptive Centering")
        print("[MODE] full standalone, compressed camera only")
        print("[FEATURE] Active centering with minimum hold duration")
        print("[FEATURE] Adaptive approach with re-centering capability")
        print(f"[MARKER] RED_ZONE={sorted(RED_ZONE_MARKER_IDS)} BLUE_ZONE={sorted(BLUE_ZONE_MARKER_IDS)}")
        print(f"[TARGET] default={DEFAULT_TARGET_COLOR}, blue-zone=>red cube, red-zone=>blue cube")
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
        
        self.last_cmd_linear = linear_x
        self.last_cmd_angular = angular_z
        
        self.position_tracker.update_from_velocity(
            linear_x, angular_z, time.monotonic()
        )

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
        old_state = self.state
        self.state = new_state
        self.state_enter_time = time.monotonic()
        
        # Reset state-specific variables
        if new_state == "SEARCH":
            self.center_seen_frames = 0
            self.approach_lost_frames = 0
            self.grab_color = None
            self.last_seen_target = None
            self.delivery_marker_ids = set()
            self.marker_align_frames = 0
            self.navigate_target_x = None
            self.navigate_target_y = None
            self.navigate_target_yaw = None
        
        if new_state == "ACTIVE_CENTER":
            self.center_seen_frames = 0
            self.centered_start_time = None
            self.centered_ready = False
        
        if new_state == "APPROACH":
            self.approach_lost_frames = 0
            self.approach_recenter_frames = 0
            self.approach_is_recentering = False
        
        if new_state == "TURN_TO_DELIVERY_MARKER":
            self.marker_align_frames = 0
        
        if new_state == "NAVIGATE_TO_ZONE":
            self.navigate_start_time = time.monotonic()
        
        if text:
            print(f"[STATE] {old_state} -> {new_state} | {text}")
        else:
            print(f"[STATE] {old_state} -> {new_state}")

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

        # Run detection when needed
        detection_states = ("WAIT_FOR_DATA", "SEARCH", "ACTIVE_CENTER", "APPROACH", "NAVIGATE_TO_ZONE")
        if self.state in detection_states:
            try:
                chosen, red_obs, blue_obs = self.detector.detect(frame, self.active_target_color())
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

        try:
            marker = self.marker_detector.detect(frame)
        except Exception as exc:
            print(f"[MARKER] detect failed: {exc}")
            marker = None
        self.marker_obs = marker
        self.marker_visible = marker is not None
        self.update_zone_from_marker()

    def target_error_pixels(self):
        if not self.target_visible or self.target_obs is None or self.image_width is None:
            return None
        return self.target_obs.cx - self.image_width / 2.0

    def target_centered(self, tolerance=None):
        """Check if target is centered within given pixel tolerance"""
        if tolerance is None:
            tolerance = SEARCH_CENTER_BAND_PX
        err = self.target_error_pixels()
        return err is not None and abs(err) <= tolerance
    
    def target_centered_tight(self):
        """Check if target is tightly centered for active centering"""
        return self.target_centered(tolerance=ACTIVE_CENTER_PIXEL_TOL)

    def update_zone_from_marker(self):
        if not self.marker_visible or self.marker_obs is None:
            return
        mid = self.marker_obs.marker_id
        if mid in RED_ZONE_MARKER_IDS or (mid in RED_ZONE_EDGE_IDS and self.marker_obs.error > 100.0):
            self.current_zone = "RED_ZONE"
        elif mid in BLUE_ZONE_MARKER_IDS or (mid in BLUE_ZONE_EDGE_IDS and self.marker_obs.error > 100.0):
            self.current_zone = "BLUE_ZONE"
    
    def update_zone_from_position(self):
        pos_zone = self.position_tracker.get_zone()
        if pos_zone != "BOUNDARY":
            if self.current_zone == "UNKNOWN":
                self.current_zone = pos_zone
                print(f"[ZONE] Position-based: {pos_zone} (x={self.position_tracker.x:.2f})")

    def active_target_color(self):
        self.update_zone_from_position()
        
        if self.current_zone == "BLUE_ZONE":
            self.current_target_color = "red"
        elif self.current_zone == "RED_ZONE":
            self.current_target_color = "blue"
        else:
            self.current_target_color = DEFAULT_TARGET_COLOR
        return self.current_target_color
    
    def is_cube_in_wrong_zone(self, cube_color: str) -> bool:
        pos_zone = self.position_tracker.get_zone()
        
        if pos_zone == "BOUNDARY" or pos_zone == "UNKNOWN":
            return True
        
        if cube_color == "red" and pos_zone == "BLUE_ZONE":
            return True
        if cube_color == "blue" and pos_zone == "RED_ZONE":
            return True
        
        return False

    def delivery_ids_for_color(self, color):
        if color == "red":
            return set(RED_ZONE_MARKER_IDS)
        if color == "blue":
            return set(BLUE_ZONE_MARKER_IDS)
        return set(RED_ZONE_MARKER_IDS)

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

    def last_target_summary(self):
        if self.last_seen_target is None:
            return "last=none"
        return (
            f"last={self.last_seen_target['color']} "
            f"h={self.last_seen_target['bbox_h']:.0f} "
            f"bottom={self.last_seen_target['bottom_y']:.0f} "
            f"dist={self.last_seen_target['distance_cm']:.1f}cm"
        )

    def last_target_was_close(self):
        if self.last_seen_target is None:
            return False
        distance_cm = self.last_seen_target["distance_cm"]
        bbox_h = self.last_seen_target["bbox_h"]
        close_by_distance = 0.1 < distance_cm <= LOST_CAPTURE_CLOSE_DISTANCE_CM
        close_by_size = bbox_h >= LOST_CAPTURE_CLOSE_BBOX_H_PX
        return close_by_distance or close_by_size

    # ========================================================================
    # NEW: ACTIVE CENTERING - replaces CENTER_STOP, BACKTRACK, FINE_ALIGN
    # ========================================================================
    def handle_active_center(self):
        """
        Active centering with feedback control.
        Rotates to center the cube and holds it centered for minimum duration.
        No hardcoded backtrack - uses continuous feedback.
        """
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        # Timeout protection
        if self.state_age() > ACTIVE_CENTER_TIMEOUT:
            print(f"[CENTER] Timeout after {ACTIVE_CENTER_TIMEOUT:.1f}s, retrying search")
            self.set_state("SEARCH", "centering timeout")
            return

        # Target lost
        if not self.target_visible or self.target_obs is None:
            self.centered_start_time = None
            self.centered_ready = False
            self.center_seen_frames = 0
            
            if self.state_age() < ACTIVE_CENTER_LOST_RECOVER:
                # Gentle search in last known direction
                search_dir = 1.0 if self.search_direction > 0 else -1.0
                self.publish_cmd(0.0, search_dir * ACTIVE_CENTER_RECOVER_ANG)
                return
            
            print("[CENTER] Target lost, returning to search")
            self.set_state("SEARCH", "target lost during centering")
            return

        # Get pixel error
        err = self.target_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        err_abs = abs(err)
        
        # Check if tightly centered
        if err_abs <= ACTIVE_CENTER_PIXEL_TOL:
            self.center_seen_frames += 1
            
            # Track when we first achieved center
            if self.centered_start_time is None:
                self.centered_start_time = time.monotonic()
            
            # Check if we've held center long enough
            centered_duration = time.monotonic() - self.centered_start_time
            if centered_duration >= ACTIVE_CENTER_MIN_DURATION and self.center_seen_frames >= 2:
                if not self.centered_ready:
                    self.centered_ready = True
                    print(f"[CENTER] Centered for {centered_duration:.2f}s, ready to approach")
                
                self.stop_robot_once()
                self.grab_color = self.target_obs.color
                self.remember_target_for_capture()
                self.set_state("APPROACH", f"{self.grab_color} centered, driving forward")
                return
            
            # Still holding position
            self.stop_robot_once()
            return
        
        # NOT centered - reset hold tracking
        self.center_seen_frames = 0
        self.centered_start_time = None
        self.centered_ready = False
        
        # Calculate angular correction
        err_norm = err / max(self.image_width / 2.0, 1.0)
        err_norm_abs = abs(err_norm)
        
        # Adaptive gain: slower when close, faster when far
        if err_norm_abs < ACTIVE_CENTER_DECEL_ZONE:
            # Close to center - use gentler correction
            gain = ACTIVE_CENTER_GAIN * (err_norm_abs / ACTIVE_CENTER_DECEL_ZONE)
            gain = max(gain, 0.3)  # Minimum gain to overcome friction
        else:
            # Far from center - use full gain
            gain = ACTIVE_CENTER_GAIN
        
        # Negative sign: counter-steer toward center
        angular = -gain * err_norm
        angular = self.clamp_abs(angular, ACTIVE_CENTER_MIN_ANG, ACTIVE_CENTER_MAX_ANG)
        
        # Print occasional debug
        if int(self.state_age() * 10) % 5 == 0:
            print(f"[CENTER] err={err:.0f}px norm={err_norm:.2f} ang={angular:.3f} gain={gain:.2f}")
        
        self.publish_cmd(0.0, angular)

    # ========================================================================
    # NEW: ADAPTIVE APPROACH with re-centering
    # ========================================================================
    def handle_approach(self):
        """
        Adaptive approach that maintains centering while driving forward.
        Can slow down, steer to re-center, then resume full speed.
        """
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        # Target lost during approach
        if not self.target_visible or self.target_obs is None:
            self.approach_lost_frames += 1
            self.approach_is_recentering = False
            self.approach_recenter_frames = 0
            
            grace_frames = BLUE_APPROACH_LOST_GRACE_FRAMES if self.grab_color == "blue" else APPROACH_LOST_GRACE_FRAMES
            
            if self.approach_lost_frames < grace_frames:
                # Brief loss - keep moving slowly
                self.publish_cmd(APPROACH_SPEED * 0.5, 0.0)
                return
            
            # Extended loss - check if cube was close enough
            if self.last_target_was_close():
                self.set_state("EXTRA_FORWARD_AFTER_LOST", f"close target disappeared, {self.last_target_summary()}")
            else:
                self.stop_robot_once()
                self.set_state("SEARCH", f"far target lost, {self.last_target_summary()}")
            return

        # Target is visible
        self.approach_lost_frames = 0
        self.remember_target_for_capture()
        
        # Get current error
        err = self.target_error_pixels()
        if err is None:
            self.publish_cmd(APPROACH_SPEED * 0.5, 0.0)
            return
        
        err_abs = abs(err)
        
        # Determine if we need to re-center
        cube_off_center = err_abs > APPROACH_CENTER_TOLERANCE
        
        if cube_off_center:
            # Cube has drifted off center - enter re-centering mode
            if not self.approach_is_recentering:
                print(f"[APPROACH] Cube off-center ({err:.0f}px), re-centering")
                self.approach_is_recentering = True
                self.approach_recenter_frames = 0
            
            # Slow down while re-centering
            approach_speed = APPROACH_RECENTER_SPEED
            
            # Calculate steering correction
            err_norm = err / max(self.image_width / 2.0, 1.0)
            steer = -APPROACH_STEER_GAIN * err_norm
            steer = max(-APPROACH_STEER_MAX, min(APPROACH_STEER_MAX, steer))
            
            # Check if we're now close enough to center
            if err_abs <= APPROACH_CENTER_TOLERANCE:
                self.approach_recenter_frames += 1
                if self.approach_recenter_frames >= APPROACH_CENTER_HOLD:
                    print(f"[APPROACH] Re-centered, resuming full approach")
                    self.approach_is_recentering = False
                    self.approach_recenter_frames = 0
                    approach_speed = APPROACH_SPEED
                    steer = 0.0
            else:
                self.approach_recenter_frames = 0
        else:
            # Cube is centered - proceed normally
            if self.approach_is_recentering:
                self.approach_recenter_frames += 1
                if self.approach_recenter_frames >= APPROACH_CENTER_HOLD:
                    print(f"[APPROACH] Re-centered and stable, resuming")
                    self.approach_is_recentering = False
                    self.approach_recenter_frames = 0
                approach_speed = APPROACH_RECENTER_SPEED
                steer = 0.0
            else:
                approach_speed = APPROACH_SPEED
                
                # Minor course correction even when centered
                err_norm = err / max(self.image_width / 2.0, 1.0)
                steer = -APPROACH_STEER_GAIN * 0.5 * err_norm  # Half gain for minor corrections
                steer = max(-APPROACH_STEER_MAX * 0.3, min(APPROACH_STEER_MAX * 0.3, steer))
        
        # Adjust speed based on distance
        if self.target_obs.distance_cm < 15.0:
            approach_speed *= 0.7  # Slow down when very close
        elif self.target_obs.distance_cm > 60.0:
            approach_speed = min(approach_speed * 1.2, APPROACH_MAX_SPEED)  # Speed up when far
        
        # Print debug info
        status = "RECENTER" if self.approach_is_recentering else "NORMAL"
        if int(self.state_age() * 10) % 5 == 0:
            print(f"[APPROACH] {status} err={err:.0f}px speed={approach_speed:.3f} steer={steer:.3f} dist={self.target_obs.distance_cm:.1f}cm")
        
        self.publish_cmd(approach_speed, steer)

    # ========================================================================
    # SEARCH - initial cube finding
    # ========================================================================
    def handle_search(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        if self.moved_cubes >= MAX_CUBES:
            self.stop_robot_once()
            if self.state_age() > 3.0:
                print(f"[MISSION] Complete! Moved {self.moved_cubes} cubes")
                self.request_shutdown("mission complete")
            return

        self.active_target_color()
        
        # Ignore cubes already in correct zone
        if self.target_visible and self.target_obs is not None:
            if not self.is_cube_in_wrong_zone(self.target_obs.color):
                print(f"[SEARCH] {self.target_obs.color} cube in correct zone, ignoring")
                self.target_visible = False
                self.target_obs = None
        
        if self.target_visible and self.target_obs is not None:
            if self.target_centered():
                self.center_seen_frames += 1
                self.stop_robot_once()
                if self.center_seen_frames >= MIN_CENTER_FRAMES:
                    self.grab_color = self.target_obs.color
                    self.remember_target_for_capture()
                    # Go straight to active centering (replaces old CENTER_STOP/BACKTRACK/FINE_ALIGN)
                    self.set_state("ACTIVE_CENTER", f"{self.grab_color} found, beginning active centering")
                return
            self.center_seen_frames = 0

        # Navigate to other zone if needed
        if self.completed_cycles > 0 and self.state_age() > 5.0:
            target_x = -0.5 if self.position_tracker.x > 0 else 0.5
            self.navigate_target_x = target_x
            self.navigate_target_y = self.position_tracker.y
            self.set_state("NAVIGATE_TO_ZONE", f"moving to x={target_x:.1f}")
            return

        self.publish_cmd(0.0, self.search_direction * SEARCH_ANG)

    def handle_navigate_to_zone(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return
        
        if self.navigate_target_x is None:
            self.set_state("SEARCH", "no navigation target")
            return
        
        dist = self.position_tracker.get_distance_to(self.navigate_target_x, self.navigate_target_y)
        if dist < NAVIGATE_TARGET_DISTANCE_M:
            print(f"[NAV] Reached target position (dist={dist:.2f}m)")
            self.set_state("SEARCH", "arrived at navigation target")
            return
        
        if self.state_age() > 15.0:
            print("[NAV] Navigation timeout")
            self.set_state("SEARCH", "navigation timeout")
            return
        
        # Check for cubes while navigating
        if self.target_visible and self.target_obs and self.is_cube_in_wrong_zone(self.target_obs.color):
            print("[NAV] Found cube while navigating, switching to search")
            self.set_state("SEARCH", "found cube during navigation")
            return
        
        target_angle = self.position_tracker.get_angle_to(self.navigate_target_x, self.navigate_target_y)
        
        angular = self.clamp_abs(0.5 * target_angle, 0.05, NAVIGATE_TO_ZONE_ANG)
        linear = NAVIGATE_TO_ZONE_SPEED * (1.0 - abs(target_angle) / math.pi)
        linear = max(0.02, linear)
        
        self.publish_cmd(linear, angular)

    # ========================================================================
    # REMAINING HANDLERS (mostly unchanged from original)
    # ========================================================================
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
        self.delivery_marker_ids = self.delivery_ids_for_color(self.grab_color)
        self.set_state("TURN_TO_DELIVERY_MARKER", f"find marker ids={sorted(self.delivery_marker_ids)}")

    def handle_turn_to_delivery_marker(self):
        if self.state_age() > MARKER_TURN_TIMEOUT_SEC:
            self.stop_robot_once()
            if self.grab_color == "red":
                self.navigate_target_x = 0.5
                self.navigate_target_y = self.position_tracker.y
                self.set_state("NAVIGATE_TO_ZONE", f"marker not found, navigating to red zone")
            else:
                self.navigate_target_x = -0.5
                self.navigate_target_y = self.position_tracker.y
                self.set_state("NAVIGATE_TO_ZONE", f"marker not found, navigating to blue zone")
            return

        if not self.marker_visible or self.marker_obs is None or self.marker_obs.marker_id not in self.delivery_marker_ids:
            self.marker_align_frames = 0
            self.publish_cmd(0.0, MARKER_TURN_SPEED)
            return

        err = self.marker_obs.error
        if abs(err) <= MARKER_ALIGN_PIXEL_TOL:
            self.marker_align_frames += 1
            self.stop_robot_once()
            if self.marker_align_frames >= MARKER_ALIGN_HOLD_FRAMES:
                self.set_state("DRIVE_TO_DELIVERY_ZONE", f"marker {self.marker_obs.marker_id} aligned")
            return

        self.marker_align_frames = 0
        err_norm = err / max(self.image_width / 2.0, 1.0)
        angular = self.clamp_abs(-0.28 * err_norm, 0.045, MARKER_TURN_SPEED)
        self.publish_cmd(0.0, angular)

    def handle_drive_to_delivery_zone(self):
        if self.state_age() < DELIVERY_FORWARD_SEC:
            steer = 0.0
            if self.marker_visible and self.marker_obs is not None and self.marker_obs.marker_id in self.delivery_marker_ids:
                steer = -0.0015 * self.marker_obs.error
                steer = max(-0.10, min(0.10, steer))
            self.publish_cmd(DELIVERY_FORWARD_SPEED, steer)
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
        self.completed_cycles += 1
        self.moved_cubes += 1
        self.servo.servo_up()
        self.search_direction = SEARCH_AFTER_RELEASE_DIRECTION
        
        print(f"[MISSION] Cubes moved: {self.moved_cubes}/{MAX_CUBES}")
        
        if self.moved_cubes < MAX_CUBES:
            pos_zone = self.position_tracker.get_zone()
            if pos_zone == "RED_ZONE":
                self.navigate_target_x = -0.5
                self.navigate_target_y = self.position_tracker.y
                self.set_state("NAVIGATE_TO_ZONE", f"heading to blue zone for remaining cubes")
            elif pos_zone == "BLUE_ZONE":
                self.navigate_target_x = 0.5
                self.navigate_target_y = self.position_tracker.y
                self.set_state("NAVIGATE_TO_ZONE", f"heading to red zone for remaining cubes")
            else:
                self.set_state("SEARCH", f"continue searching, cycles={self.completed_cycles}")
        else:
            self.set_state("SEARCH", f"all {MAX_CUBES} cubes moved, mission complete")

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
            "ACTIVE_CENTER": self.handle_active_center,  # NEW - replaces 3 states
            "APPROACH": self.handle_approach,
            "EXTRA_FORWARD_AFTER_LOST": self.handle_extra_forward_after_lost,
            "SERVO_DOWN": self.handle_servo_down,
            "TURN_TO_DELIVERY_MARKER": self.handle_turn_to_delivery_marker,
            "DRIVE_TO_DELIVERY_ZONE": self.handle_drive_to_delivery_zone,
            "SERVO_UP_RELEASE": self.handle_servo_up_release,
            "BACKUP_AFTER_RELEASE": self.handle_backup_after_release,
            "NAVIGATE_TO_ZONE": self.handle_navigate_to_zone,
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
            )
        
        # Show centering status if active
        center_info = ""
        if self.state == "ACTIVE_CENTER" and self.centered_start_time is not None:
            dur = time.monotonic() - self.centered_start_time
            center_info = f"centered_for={dur:.2f}s "

        print(
            f"[STATUS] {self.state} {center_info}{target_text} grab={self.grab_color} "
            f"zone={self.current_zone} want={self.current_target_color} "
            f"marker={self.marker_obs.marker_id if self.marker_obs else 'none'} "
            f"front={self.front_dist:.3f}m "
            f"cubes={self.moved_cubes}/{MAX_CUBES} "
            f"pos=({self.position_tracker.x:.2f},{self.position_tracker.y:.2f}) "
            f"errors={self.detect_error_count}"
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
    node = SimpleCubeMissionV30()
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