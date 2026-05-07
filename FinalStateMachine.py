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
DEFAULT_TARGET_COLOR = "red"   # used before a zone marker has been seen
PREFER_BLUE = False

# v42: exam-room mode. Faster lock, no one-second reverse correction.
SEARCH_ANG = 0.176
SEARCH_CENTER_BAND_PX = 95
MIN_CENTER_FRAMES = 1

CENTER_STOP_SEC = 0.12
BACKTRACK_AFTER_CENTER_SEC = 0.0
BACKTRACK_AFTER_CENTER_ANG = 0.0
FINE_ALIGN_LOST_RECOVER_SEC = 0.7
FINE_ALIGN_SEARCH_ANG = 0.07
FINE_ALIGN_PIXEL_TOL = 34
FINE_ALIGN_HOLD_FRAMES = 1
FINE_ALIGN_MIN_ANG = 0.025
FINE_ALIGN_MAX_ANG = 0.09

APPROACH_SPEED = 0.078
APPROACH_STEER_GAIN = 0.0022
APPROACH_STEER_MAX = 0.13
EXTRA_FORWARD_SPEED = 0.060
EXTRA_FORWARD_AFTER_LOST_SEC = 1.2
BACKUP_AFTER_GRAB_SPEED = -0.12
BACKUP_AFTER_GRAB_SEC = 0.7
APPROACH_LOST_GRACE_FRAMES = 5
BLUE_APPROACH_LOST_GRACE_FRAMES = 14

LOST_CAPTURE_CLOSE_DISTANCE_CM = 24.0
LOST_CAPTURE_CLOSE_BBOX_H_PX = 105

MARKER_ALIGN_PIXEL_TOL = 24.0
MARKER_ALIGN_HOLD_FRAMES = 2
MARKER_TURN_SPEED = 0.220
MARKER_TURN_TIMEOUT_SEC = 35.0
MARKER_TIMEOUT_STOP = True
ZONE_MARKER_STABLE_FRAMES = 3

# NEW: Position-based delivery parameters
POSITION_BASED_DELIVERY = True       # Use position instead of marker search for delivery
DELIVERY_BACKWARD_SPEED = -0.10      # Speed when backing into delivery zone
DELIVERY_BACKWARD_STEER_MAX = 0.12   # Max steering correction while backing
DELIVERY_BACKWARD_STEER_GAIN = 0.003 # Steering gain while backing
DELIVERY_BACKWARD_TIMEOUT_SEC = 20.0 # Max time for backward delivery
ZONE_DELIVERY_X_THRESHOLD = 0.15     # X position considered "in zone" for delivery (meters)
ZONE_BOUNDARY_X = 0.0                # x > 0 = red zone, x < 0 = blue zone

DELIVERY_FORWARD_SPEED = 0.116
DELIVERY_FORWARD_MIN_SEC = 5.0
DELIVERY_FORWARD_MAX_SEC = 15.0
DELIVERY_MARKER_LOST_RELEASE_FRAMES = 6
DELIVERY_MARKER_TOO_CLOSE_WIDTH = 155.0
DELIVERY_STEER_GAIN = 0.0022
DELIVERY_STEER_MAX = 0.14
BACKUP_SPEED = -0.26
BACKUP_AFTER_RELEASE_SEC = 1.0
TURN_AFTER_RELEASE_DEG = -90.0
TURN_AFTER_RELEASE_TOL_DEG = 5.0
TURN_AFTER_RELEASE_SPEED = 0.20
RETURN_CENTER_AFTER_RELEASE = True
RETURN_CENTER_TARGET_MARKER_WIDTH = 41.0
RETURN_CENTER_WIDTH_TOL = 2.0
RETURN_CENTER_SPEED = 0.12
RETURN_CENTER_STEER_GAIN = 0.0030
RETURN_CENTER_STEER_MAX = 0.18
RETURN_CENTER_TIMEOUT_SEC = 8.0

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
MAX_ASPECT = 2.05
MIN_FILL_RATIO = 0.15
MIN_EXTENT = 0.13
MIN_SOLIDITY = 0.40
MIN_CENTER_Y_RATIO = 0.12

RED_MIN_HOLES_REQUIRED = 3
BLUE_MIN_HOLES_REQUIRED = 0

RED_ZONE_MARKER_IDS = {0}
BLUE_ZONE_MARKER_IDS = {23}
NORTH_LINE_MARKER_IDS = {7}
SOUTH_LINE_MARKER_IDS = {42}

SEARCH_START_DIRECTION = 1.0      # positive angular.z, startup counterclockwise
SEARCH_AFTER_RELEASE_DIRECTION = -1.0
ZONE_UPDATE_STATES = {
    "WAIT_FOR_DATA",
    "SEARCH",
}

REAL_CUBE_SIZE_CM = 5.0
FOCAL_LENGTH_PX = 520.0

CONTROL_DT = 0.05
STATUS_DT = 0.8
CAMERA_GAMMA = 1.5


# =========================
# Position tracking tuning (NEW)
# =========================
POSITION_HISTORY_SEC = 2.0
MAX_CUBES = 4


# NEW: Pre-delivery rotation to avoid cube collisions
PREDELIVERY_ROTATE_ANG = 0.176       # Rotation speed for pre-delivery turn
PREDELIVERY_ROTATE_SEC = 0.5         # Duration of slight rotation (anti-clockwise)
PREDELIVERY_ROTATE_DIRECTION = 1.0   # 1.0 = anti-clockwise, -1.0 = clockwise

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
    """Tracks robot position using velocity integration from cmd_vel"""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_update_time = None
        self.position_history = []
        
    def update_from_velocity(self, linear_x: float, angular_z: float, current_time: float):
        """Update position based on velocity commands (dead reckoning)"""
        if self.last_update_time is None:
            self.last_update_time = current_time
            return
            
        dt = current_time - self.last_update_time
        if dt <= 0 or dt > 1.0:  # Prevent large jumps
            self.last_update_time = current_time
            return
            
        # Simple differential drive model
        if abs(angular_z) < 1e-6:
            # Straight movement
            self.x += linear_x * math.cos(self.yaw) * dt
            self.y += linear_x * math.sin(self.yaw) * dt
        else:
            # Arc movement
            radius = linear_x / angular_z
            d_theta = angular_z * dt
            self.x += radius * (math.sin(self.yaw + d_theta) - math.sin(self.yaw))
            self.y -= radius * (math.cos(self.yaw + d_theta) - math.cos(self.yaw))
            self.yaw += d_theta
            
        self.yaw = self.normalize_angle(self.yaw)
        self.last_update_time = current_time
        
        # Store position history
        self.position_history.append(RobotPosition(
            x=self.x, y=self.y, yaw=self.yaw, timestamp=current_time
        ))
        
        # Clean old history
        cutoff = current_time - POSITION_HISTORY_SEC
        self.position_history = [p for p in self.position_history if p.timestamp > cutoff]
    
    def normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def get_zone(self) -> str:
        """Determine which zone the robot is in based on x position"""
        if self.x > ZONE_BOUNDARY_X:
            return "RED_ZONE"
        elif self.x < -ZONE_BOUNDARY_X:
            return "BLUE_ZONE"
        return "BOUNDARY"
    
    def get_distance_to(self, target_x: float, target_y: float) -> float:
        """Get Euclidean distance to a target point"""
        return math.sqrt((self.x - target_x)**2 + (self.y - target_y)**2)
    
    def get_angle_to(self, target_x: float, target_y: float) -> float:
        """Get angle from current heading to a target point"""
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        return self.normalize_angle(target_angle - self.yaw)
    
    def is_in_zone(self, target_zone: str) -> bool:
        """Check if robot position is within the specified zone"""
        if target_zone == "RED_ZONE":
            return self.x >= ZONE_DELIVERY_X_THRESHOLD
        elif target_zone == "BLUE_ZONE":
            return self.x <= -ZONE_DELIVERY_X_THRESHOLD
        return False
    
    def get_zone_for_cube(self, cube_color: str) -> str:
        """Return the correct zone for a cube color"""
        if cube_color == "red":
            return "RED_ZONE"
        return "BLUE_ZONE"
    
    def get_zone_boundary_x(self, zone: str) -> float:
        """Return the x position threshold for a zone"""
        if zone == "RED_ZONE":
            return ZONE_DELIVERY_X_THRESHOLD
        return -ZONE_DELIVERY_X_THRESHOLD
    
    def reset_position(self):
        """Reset position to origin"""
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.position_history.clear()
        print("[POS] Position reset to (0,0)")


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
        lower_red_1 = np.array([0, 115, 50], dtype=np.uint8)
        upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([170, 115, 50], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2),
        )

        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)
        rgb_mask = np.zeros_like(hsv_mask)
        rgb_mask[(r >= 65) & (r > g + 18) & (r > b + 22)] = 255
        return cv2.bitwise_and(hsv_mask, rgb_mask)

    def build_blue_mask(self, hsv, bgr):
        lower_blue = np.array([102, 125, 45], dtype=np.uint8)
        upper_blue = np.array([132, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)
        rgb_mask = np.zeros_like(hsv_mask)
        rgb_mask[(b >= 45) & (b > r + 12) & (b > g - 6)] = 255
        return cv2.bitwise_and(hsv_mask, rgb_mask)

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
            hue_ratio = float(np.mean((hue >= 102) & (hue <= 132) & (sat >= 115)))
            dom_br = float(mean_b - mean_r)
            dom_bg = float(mean_b - mean_g)
            conf = 0.25 * mean_s + 0.35 * dom_br + 0.15 * dom_bg + 90.0 * hue_ratio
            ok = hue_ratio >= 0.16 and mean_s >= 105 and mean_v >= 40 and mean_b >= 45 and dom_br >= 10
            return ok, float(conf)

        hue_ratio = float(np.mean(((hue <= 10) | (hue >= 170)) & (sat >= 105)))
        dom_rb = float(mean_r - mean_b)
        dom_rg = float(mean_r - mean_g)
        conf = 0.25 * mean_s + 0.35 * dom_rb + 0.25 * dom_rg + 90.0 * hue_ratio
        ok = (
            hue_ratio >= 0.18
            and mean_s >= 105
            and mean_v >= 40
            and mean_r >= 65
            and dom_rb >= 18
            and dom_rg >= 14
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


class SimpleCubeMissionV42(Node):
    def __init__(self):
        super().__init__("simple_cube_mission_v42")

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
        
        # NEW: Position tracking system
        self.position_tracker = PositionTracker()
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        
        # NEW: Mission tracking
        self.moved_cubes = 0
        self.grab_position_x = None  # Store position where cube was grabbed

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
        self.locked_search_color = None
        self.zone_candidate = "UNKNOWN"
        self.zone_candidate_frames = 0
        self.last_marker_id = None
        self.last_zone_reason = "none"
        self.delivery_marker_ids = set()
        self.delivery_marker_lost_frames = 0
        self.delivery_marker_seen_once = False
        self.marker_align_frames = 0
        self.center_seen_frames = 0
        self.approach_lost_frames = 0
        self.grab_color = None
        self.last_seen_target = None
        self.completed_cycles = 0
        self.detect_error_count = 0
        self.search_direction = SEARCH_START_DIRECTION

        self.state = "WAIT_FOR_DATA"
        self.state_enter_time = time.monotonic()
        self.shutdown_requested = False
        self.shutdown_count = 0
        self.wait_status_last = None

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        print("[BOOT] SimpleCubeMission v42 with Position Tracking")
        print("[MODE] full standalone, compressed camera only")
        print("[POSITION] Tracking active - initial position (0, 0)")
        print(f"[POSITION] x > {ZONE_BOUNDARY_X} = RED_ZONE, x < -{ZONE_BOUNDARY_X} = BLUE_ZONE")
        print(f"[DELIVERY] Position-based delivery: {POSITION_BASED_DELIVERY}")
        if POSITION_BASED_DELIVERY:
            print(f"[DELIVERY] Backward speed: {DELIVERY_BACKWARD_SPEED} m/s")
            print(f"[DELIVERY] Zone threshold: ±{ZONE_DELIVERY_X_THRESHOLD}m")
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
        
        # NEW: Update position tracker with velocity commands
        self.last_cmd_linear = linear_x
        self.last_cmd_angular = angular_z
        self.position_tracker.update_from_velocity(linear_x, angular_z, time.monotonic())

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
        if new_state == "SEARCH":
            self.center_seen_frames = 0
            self.approach_lost_frames = 0
            self.grab_color = None
            self.last_seen_target = None
            self.delivery_marker_ids = set()
            self.marker_align_frames = 0
            self.locked_search_color = None
            self.target_visible = False
            self.target_obs = None
            self.red_obs = None
            self.blue_obs = None
            self.grab_position_x = None
        if new_state in ("CENTER_STOP", "BACKTRACK_AFTER_CENTER", "FINE_ALIGN"):
            self.center_seen_frames = 0
        if new_state == "APPROACH":
            self.approach_lost_frames = 0
        if new_state == "TURN_TO_DELIVERY_MARKER":
            self.marker_align_frames = 0
        if new_state == "BACKUP_AFTER_GRAB":
            self.target_visible = False
            self.target_obs = None
            self.red_obs = None
            self.blue_obs = None
            # NEW: Record position where cube was grabbed
            self.grab_position_x = self.position_tracker.x
            print(f"[POS] Cube grabbed at x={self.grab_position_x:.2f}m")
        if new_state == "DRIVE_TO_DELIVERY_ZONE":
            self.delivery_marker_lost_frames = 0
            self.delivery_marker_seen_once = False
            if (
                self.marker_visible
                and self.marker_obs is not None
                and self.marker_obs.marker_id in self.delivery_marker_ids
            ):
                self.delivery_marker_seen_once = True
        if new_state == "BACKWARD_DELIVERY":
            print(f"[DELIVERY] Starting backward delivery of {self.grab_color} cube")
            print(f"[DELIVERY] Current pos: ({self.position_tracker.x:.2f}, {self.position_tracker.y:.2f})")
            print(f"[DELIVERY] Target zone: {self.position_tracker.get_zone_for_cube(self.grab_color or 'red')}")
        if new_state == "TURN_AFTER_RELEASE_90":
            self.target_visible = False
            self.target_obs = None
            self.red_obs = None
            self.blue_obs = None
            self.locked_search_color = None
        if new_state == "RETURN_CENTER_AFTER_RELEASE":
            self.target_visible = False
            self.target_obs = None
            self.red_obs = None
            self.blue_obs = None
            self.locked_search_color = None
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
            elif cmd == "p":
                # NEW: Print current position
                print(f"[POS] Current: x={self.position_tracker.x:.2f}m, y={self.position_tracker.y:.2f}m, yaw={math.degrees(self.position_tracker.yaw):.0f}°")
                print(f"[POS] Zone: {self.position_tracker.get_zone()}")
            elif cmd == "r":
                # NEW: Reset position
                self.position_tracker.reset_position()

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

        if CAMERA_GAMMA and abs(CAMERA_GAMMA - 1.0) > 1e-6:
            inv_gamma = 1.0 / CAMERA_GAMMA
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
            frame = cv2.LUT(frame, table)

        self.image_height, self.image_width = frame.shape[:2]
        if not self.image_logged:
            print(f"[IMAGE] compressed size={self.image_width}x{self.image_height}")
            self.image_logged = True

        try:
            marker = self.marker_detector.detect(frame)
        except Exception as exc:
            print(f"[MARKER] detect failed: {exc}")
            marker = None
        self.marker_obs = marker
        self.marker_visible = marker is not None
        self.update_zone_from_marker()

        if self.state in ("WAIT_FOR_DATA", "SEARCH", "CENTER_STOP", "BACKTRACK_AFTER_CENTER", "FINE_ALIGN", "APPROACH"):
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

    def target_error_pixels(self):
        if not self.target_visible or self.target_obs is None or self.image_width is None:
            return None
        return self.target_obs.cx - self.image_width / 2.0

    def target_centered(self):
        err = self.target_error_pixels()
        return err is not None and abs(err) <= SEARCH_CENTER_BAND_PX

    def update_zone_from_marker(self):
        if not self.marker_visible or self.marker_obs is None:
            return
        if self.state not in ZONE_UPDATE_STATES:
            return
        mid = self.marker_obs.marker_id
        self.last_marker_id = mid

        if mid in RED_ZONE_MARKER_IDS:
            self.current_zone = "RED_ZONE"
            self.zone_candidate = "RED_ZONE"
            self.zone_candidate_frames = ZONE_MARKER_STABLE_FRAMES
            self.last_zone_reason = "direct_red_marker"
            return

        if mid in BLUE_ZONE_MARKER_IDS:
            self.current_zone = "BLUE_ZONE"
            self.zone_candidate = "BLUE_ZONE"
            self.zone_candidate_frames = ZONE_MARKER_STABLE_FRAMES
            self.last_zone_reason = "direct_blue_marker"
            return

        if mid in NORTH_LINE_MARKER_IDS and self.marker_obs.error > 100.0:
            self.current_zone = "RED_ZONE"
            self.zone_candidate = "RED_ZONE"
            self.zone_candidate_frames = ZONE_MARKER_STABLE_FRAMES
            self.last_zone_reason = "north_line_buffer"
            return

        if mid in SOUTH_LINE_MARKER_IDS and self.marker_obs.error > 100.0:
            self.current_zone = "BLUE_ZONE"
            self.zone_candidate = "BLUE_ZONE"
            self.zone_candidate_frames = ZONE_MARKER_STABLE_FRAMES
            self.last_zone_reason = "south_line_buffer"

    # NEW: Update zone from position tracking
    def update_zone_from_position(self):
        pos_zone = self.position_tracker.get_zone()
        if pos_zone != "BOUNDARY":
            if self.current_zone == "UNKNOWN":
                self.current_zone = pos_zone
                self.last_zone_reason = f"position_tracker_{pos_zone.lower()}"
                print(f"[ZONE] Position-based: {pos_zone} (x={self.position_tracker.x:.2f})")

    def infer_zone_from_line_marker(self, marker_id):
        direction = self.search_direction
        if marker_id in NORTH_LINE_MARKER_IDS:
            if direction > 0:
                return "RED_ZONE"
            if direction < 0:
                return "BLUE_ZONE"
        if marker_id in SOUTH_LINE_MARKER_IDS:
            if direction > 0:
                return "BLUE_ZONE"
            if direction < 0:
                return "RED_ZONE"
        return "UNKNOWN"

    def active_target_color(self):
        # NEW: Also update zone from position
        self.update_zone_from_position()
        
        if self.locked_search_color is not None:
            self.current_target_color = self.locked_search_color
            return self.current_target_color

        if self.current_zone == "BLUE_ZONE":
            self.current_target_color = "red"
        elif self.current_zone == "RED_ZONE":
            self.current_target_color = "blue"
        else:
            self.current_target_color = DEFAULT_TARGET_COLOR
        return self.current_target_color
    
    # NEW: Check if cube is in wrong zone
    def is_cube_in_wrong_zone(self, cube_color: str) -> bool:
        """Check if a cube of given color is in the wrong zone based on position"""
        pos_zone = self.position_tracker.get_zone()
        
        if pos_zone == "BOUNDARY" or pos_zone == "UNKNOWN":
            return True  # Can't determine - process any cube
        
        target_zone = self.position_tracker.get_zone_for_cube(cube_color)
        return pos_zone != target_zone

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

    def handle_search(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        self.active_target_color()
        
        # NEW: Check if visible cube is in wrong zone (should be moved)
        if self.target_visible and self.target_obs is not None:
            if not self.is_cube_in_wrong_zone(self.target_obs.color):
                print(f"[SEARCH] {self.target_obs.color} cube already in correct zone, ignoring")
                self.target_visible = False
                self.target_obs = None
                # Continue searching
        
        if self.target_visible and self.target_obs is not None:
            if self.locked_search_color is None:
                self.locked_search_color = self.target_obs.color
                self.current_target_color = self.locked_search_color
            if self.target_centered():
                self.center_seen_frames += 1
                self.stop_robot_once()
                if self.center_seen_frames >= MIN_CENTER_FRAMES:
                    self.grab_color = self.target_obs.color
                    self.remember_target_for_capture()
                    self.set_state("CENTER_STOP", f"{self.grab_color} centered, short stop")
                return
            self.center_seen_frames = 0
            err = self.target_error_pixels()
            if err is None:
                self.stop_robot_once()
                return
            err_norm = err / max(self.image_width / 2.0, 1.0)
            angular = self.clamp_abs(-0.24 * err_norm, 0.045, SEARCH_ANG)
            self.publish_cmd(0.0, angular)
            return

        self.publish_cmd(0.0, self.search_direction * SEARCH_ANG)

    def handle_center_stop(self):
        self.stop_robot_once()
        if self.state_age() >= CENTER_STOP_SEC:
            self.set_state("FINE_ALIGN", "quick fine align")

    def handle_backtrack_after_center(self):
        if self.state_age() < BACKTRACK_AFTER_CENTER_SEC:
            self.publish_cmd(0.0, BACKTRACK_AFTER_CENTER_ANG)
            return
        self.stop_robot_once()
        self.set_state("FINE_ALIGN", "fine align")

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
            grace_frames = BLUE_APPROACH_LOST_GRACE_FRAMES if self.grab_color == "blue" else APPROACH_LOST_GRACE_FRAMES
            if self.approach_lost_frames < grace_frames:
                self.publish_cmd(APPROACH_SPEED * 0.65, 0.0)
                return
            if self.last_target_was_close():
                self.set_state("EXTRA_FORWARD_AFTER_LOST", f"close target disappeared, {self.last_target_summary()}")
            else:
                self.stop_robot_once()
                self.set_state("SEARCH", f"far target lost, no grab, {self.last_target_summary()}")
            return

        self.approach_lost_frames = 0
        self.remember_target_for_capture()
        err = self.target_error_pixels()
        steer = 0.0 if err is None else -APPROACH_STEER_GAIN * err
        steer = max(-APPROACH_STEER_MAX, min(APPROACH_STEER_MAX, steer))
        self.publish_cmd(APPROACH_SPEED, steer)

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
        
        # Record position where cube was grabbed
        self.grab_position_x = self.position_tracker.x
        print(f"[POS] Cube grabbed at x={self.grab_position_x:.2f}m")
        
        # Always go through BACKUP_AFTER_GRAB (which now includes rotation)
        self.set_state("BACKUP_AFTER_GRAB", f"backup {BACKUP_AFTER_GRAB_SEC}s + rotate {PREDELIVERY_ROTATE_SEC}s")

    # ========================================================================
    # NEW: Position-based backward delivery
    # ========================================================================
    def handle_backward_delivery(self):
        """Back up into the correct zone based on position tracking"""
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance during backward delivery")
            return
        
        # Timeout protection
        if self.state_age() > DELIVERY_BACKWARD_TIMEOUT_SEC:
            print(f"[DELIVERY] Backward delivery timeout ({DELIVERY_BACKWARD_TIMEOUT_SEC}s)")
            self.stop_robot_once()
            self.set_state("SERVO_UP_RELEASE", "delivery timeout, releasing cube")
            return
        
        # Determine target zone
        target_zone = self.position_tracker.get_zone_for_cube(self.grab_color or "red")
        target_x = self.position_tracker.get_zone_boundary_x(target_zone)
        
        # Check if we've reached the target zone
        if self.position_tracker.is_in_zone(target_zone):
            print(f"[DELIVERY] Reached {target_zone}! x={self.position_tracker.x:.2f}m")
            self.stop_robot_once()
            self.set_state("SERVO_UP_RELEASE", f"cube delivered to {target_zone}")
            return
        
        # Calculate distance to target
        dist_to_target = abs(self.position_tracker.x - target_x)
        
        # Calculate steering to stay straight while backing
        # Use marker if visible for steering reference, otherwise keep straight
        steer = 0.0
        if self.marker_visible and self.marker_obs is not None:
            # Use marker as visual reference for steering
            steer = -DELIVERY_BACKWARD_STEER_GAIN * self.marker_obs.error
            steer = max(-DELIVERY_BACKWARD_STEER_MAX, min(DELIVERY_BACKWARD_STEER_MAX, steer))
        
        # Adjust speed based on distance to target
        if dist_to_target < 0.1:
            speed = DELIVERY_BACKWARD_SPEED * 0.5  # Slow down near target
        else:
            speed = DELIVERY_BACKWARD_SPEED
        
        # Print status periodically
        if int(self.state_age() * 4) % 4 == 0:
            print(f"[DELIVERY] Backing to {target_zone}: x={self.position_tracker.x:.2f}m, target={target_x:+.2f}m, dist={dist_to_target:.2f}m")
        
        self.publish_cmd(speed, steer)

    def handle_backup_after_grab(self):
        """Backup briefly then rotate slightly to offset return trajectory"""
        total_time = BACKUP_AFTER_GRAB_SEC + PREDELIVERY_ROTATE_SEC
        
        if self.state_age() < BACKUP_AFTER_GRAB_SEC:
            # Phase 1: Brief backup to seat cube in grabber
            self.publish_cmd(BACKUP_AFTER_GRAB_SPEED, 0.0)
            return
        
        if self.state_age() < total_time:
            # Phase 2: Slight rotation to avoid cube collisions on return
            if self.state_age() < BACKUP_AFTER_GRAB_SEC + 0.1:  # Small transition window
                print(f"[PREDELIVERY] Backup complete, rotating to offset trajectory")
            self.publish_cmd(0.0, PREDELIVERY_ROTATE_DIRECTION * PREDELIVERY_ROTATE_ANG)
            return
        
        # Both phases complete - stop and proceed
        self.stop_robot_once()
        print(f"[PREDELIVERY] Maneuver complete: backup {BACKUP_AFTER_GRAB_SEC}s + rotate {PREDELIVERY_ROTATE_SEC}s")
        print(f"[PREDELIVERY] Position: x={self.position_tracker.x:.2f}m, y={self.position_tracker.y:.2f}m")
        
        # Now decide: position-based delivery or marker-based delivery
        if POSITION_BASED_DELIVERY:
            target_zone = self.position_tracker.get_zone_for_cube(self.grab_color or "red")
            self.set_state("BACKWARD_DELIVERY", f"backing into {target_zone}")
        else:
            self.set_state("TURN_TO_DELIVERY_MARKER", f"find marker ids={sorted(self.delivery_marker_ids)}")

    def handle_turn_to_delivery_marker(self):
        if self.state_age() > MARKER_TURN_TIMEOUT_SEC:
            self.stop_robot_once()
            if MARKER_TIMEOUT_STOP:
                self.set_state("STOPPED", "delivery marker timeout, press H or restart")
            else:
                self.set_state("TURN_TO_DELIVERY_MARKER", "delivery marker timeout, keep turning")
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
        age = self.state_age()
        steer = 0.0
        marker_ok = (
            self.marker_visible
            and self.marker_obs is not None
            and self.marker_obs.marker_id in self.delivery_marker_ids
        )

        if marker_ok:
            self.delivery_marker_seen_once = True
            self.delivery_marker_lost_frames = 0
            steer = -DELIVERY_STEER_GAIN * self.marker_obs.error
            steer = max(-DELIVERY_STEER_MAX, min(DELIVERY_STEER_MAX, steer))
            if age >= DELIVERY_FORWARD_MIN_SEC and self.marker_obs.width >= DELIVERY_MARKER_TOO_CLOSE_WIDTH:
                self.stop_robot_once()
                self.set_state("SERVO_UP_RELEASE", f"marker too close width={self.marker_obs.width:.0f}, release")
                return
        else:
            self.delivery_marker_lost_frames += 1
            if (
                age >= DELIVERY_FORWARD_MIN_SEC
                and self.delivery_marker_seen_once
                and self.delivery_marker_lost_frames >= DELIVERY_MARKER_LOST_RELEASE_FRAMES
            ):
                self.stop_robot_once()
                self.set_state("SERVO_UP_RELEASE", "delivery marker disappeared, release")
                return

        if age >= DELIVERY_FORWARD_MAX_SEC:
            self.stop_robot_once()
            self.set_state("SERVO_UP_RELEASE", "delivery forward timeout 15s, release")
            return

        self.publish_cmd(DELIVERY_FORWARD_SPEED, steer)

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
        self.moved_cubes += 1  # NEW: Count delivered cubes
        self.servo.servo_up()
        
        # Update zone based on delivery
        if self.grab_color == "red":
            self.current_zone = "RED_ZONE"
            self.last_zone_reason = "released_red_in_red_zone"
        elif self.grab_color == "blue":
            self.current_zone = "BLUE_ZONE"
            self.last_zone_reason = "released_blue_in_blue_zone"
        
        print(f"[MISSION] Cubes delivered: {self.moved_cubes}")
        print(f"[POS] Position after delivery: x={self.position_tracker.x:.2f}m, y={self.position_tracker.y:.2f}m")
        
        # NEW: Check if mission complete
        if self.moved_cubes >= MAX_CUBES:
            print(f"[MISSION] All {MAX_CUBES} cubes delivered!")
            self.set_state("STOPPED", "mission complete")
            return
        
        if RETURN_CENTER_AFTER_RELEASE:
            self.set_state("RETURN_CENTER_AFTER_RELEASE", "use marker width to return near center")
        else:
            self.start_turn_relative(math.radians(TURN_AFTER_RELEASE_DEG), "TURN_AFTER_RELEASE_90")

    def handle_return_center_after_release(self):
        if self.state_age() > RETURN_CENTER_TIMEOUT_SEC:
            self.stop_robot_once()
            self.search_direction = SEARCH_START_DIRECTION
            self.set_state("SEARCH", f"return-center timeout, restart search, cycles={self.completed_cycles}")
            return

        if not self.marker_visible or self.marker_obs is None:
            self.publish_cmd(0.0, SEARCH_ANG)
            return

        err = self.marker_obs.error
        steer = -RETURN_CENTER_STEER_GAIN * err
        steer = max(-RETURN_CENTER_STEER_MAX, min(RETURN_CENTER_STEER_MAX, steer))
        width_err = self.marker_obs.width - RETURN_CENTER_TARGET_MARKER_WIDTH

        if abs(width_err) <= RETURN_CENTER_WIDTH_TOL:
            if abs(err) <= MARKER_ALIGN_PIXEL_TOL * 1.5:
                self.stop_robot_once()
                self.search_direction = SEARCH_START_DIRECTION
                self.set_state("SEARCH", f"center returned by marker width={self.marker_obs.width:.1f}, cycles={self.completed_cycles}")
                return
            self.publish_cmd(0.0, steer)
            return

        if width_err > 0:
            self.publish_cmd(-RETURN_CENTER_SPEED, steer)
        else:
            self.publish_cmd(RETURN_CENTER_SPEED, steer)

    def handle_turn_after_release_90(self):
        err = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(math.degrees(err)) <= TURN_AFTER_RELEASE_TOL_DEG:
            self.stop_robot_once()
            self.search_direction = SEARCH_START_DIRECTION
            self.set_state("SEARCH", f"restart ccw search, cycles={self.completed_cycles}")
            return
        angular = self.clamp_abs(err * 0.75, 0.07, TURN_AFTER_RELEASE_SPEED)
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
            "BACKWARD_DELIVERY": self.handle_backward_delivery,  # NEW
            "BACKUP_AFTER_GRAB": self.handle_backup_after_grab,
            "TURN_TO_DELIVERY_MARKER": self.handle_turn_to_delivery_marker,
            "DRIVE_TO_DELIVERY_ZONE": self.handle_drive_to_delivery_zone,
            "SERVO_UP_RELEASE": self.handle_servo_up_release,
            "BACKUP_AFTER_RELEASE": self.handle_backup_after_release,
            "RETURN_CENTER_AFTER_RELEASE": self.handle_return_center_after_release,
            "TURN_AFTER_RELEASE_90": self.handle_turn_after_release_90,
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
            f"zone={self.current_zone} want={self.current_target_color} lock={self.locked_search_color} "
            f"marker={self.marker_obs.marker_id if self.marker_obs else 'none'} "
            f"merr={self.marker_obs.error if self.marker_obs else 0:.0f} "
            f"zone_reason={self.last_zone_reason} "
            f"front={self.front_dist:.3f}m yaw={math.degrees(self.local_yaw):.0f} "
            f"pos=({self.position_tracker.x:.2f},{self.position_tracker.y:.2f}) "  # NEW
            f"cubes={self.moved_cubes}/{MAX_CUBES} "  # NEW
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
    node = SimpleCubeMissionV42()
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