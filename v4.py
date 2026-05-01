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


SEARCH_ANG = 0.20
ALIGN_MIN_ANG = 0.035
ALIGN_MAX_ANG = 0.18
APPROACH_FAST = 0.055
APPROACH_SLOW = 0.028
APPROACH_CREEP = 0.018

CONTROL_DT = 0.05
STATUS_DT = 0.8

MIN_CONTOUR_AREA = 260.0
MIN_TRACK_AREA = 330.0
MIN_LOCK_AREA = 650.0
MIN_BBOX_W = 15
MIN_BBOX_H = 15
MIN_ASPECT = 0.52
MAX_ASPECT = 1.90
MIN_FILL_RATIO = 0.25
MIN_EXTENT = 0.21
MIN_SOLIDITY = 0.68
MIN_CENTER_Y_RATIO = 0.22
BOTTOM_FLOOR_BONUS_RATIO = 0.45

MAX_RAW_JUMP_PX = 95.0
CONFIRM_FRAMES = 3
CENTER_HOLD_FRAMES = 3
ALIGN_DONE_PX = 22
APPROACH_ROTATE_ONLY_PX = 72
REALIGN_PX = 105
LOST_TIMEOUT = 0.55

MIN_HOLES_FOR_RANGE = 3
SLOW_HOLE_PITCH_PX = 15.0
STOP_HOLE_PITCH_PX = 23.0
STOP_BBOX_H_PX = 205
CLOSE_LOST_BBOX_H_PX = 155
HARD_FRONT_STOP_DIST = 0.14
CLOSE_LOST_FRONT_DIST = 0.20
EMERGENCY_STOP_DIST = 0.095


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


class CubeDetector:
    def __init__(self):
        self.image_width = 0
        self.image_height = 0

    def build_red_mask(self, hsv, bgr):
        lower_red_1 = np.array([0, 95, 45], dtype=np.uint8)
        upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([168, 95, 45], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2),
        )

        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]
        rgb_mask = np.zeros_like(hsv_mask)
        red_dom = (r >= 70) & (r > g + 28) & (r > b + 28)
        rgb_mask[red_dom] = 255
        return cv2.bitwise_and(hsv_mask, rgb_mask)

    def build_blue_mask(self, hsv, bgr):
        lower_blue_core = np.array([100, 90, 35], dtype=np.uint8)
        upper_blue_core = np.array([130, 255, 255], dtype=np.uint8)
        lower_blue_wide = np.array([95, 75, 30], dtype=np.uint8)
        upper_blue_wide = np.array([138, 255, 255], dtype=np.uint8)
        hsv_core = cv2.inRange(hsv, lower_blue_core, upper_blue_core)
        hsv_wide = cv2.inRange(hsv, lower_blue_wide, upper_blue_wide)

        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]
        rgb_core = np.zeros_like(hsv_core)
        rgb_wide = np.zeros_like(hsv_core)
        blue_core = (b >= 60) & (b > r + 22) & (b > g + 12)
        blue_wide = (b >= 55) & (b > r + 18) & (b > g - 8)
        rgb_core[blue_core] = 255
        rgb_wide[blue_wide] = 255

        core = cv2.bitwise_and(hsv_core, rgb_core)
        wide = cv2.bitwise_and(hsv_wide, rgb_wide)
        return cv2.bitwise_or(core, wide)

    def preprocess_mask(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask

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
            hue_ratio = float(np.mean((hue >= 104) & (hue <= 126) & (sat >= 105)))
            dom_br = float(mean_b - mean_r)
            dom_bg = float(mean_b - mean_g)
            conf = 0.30 * float(mean_s) + 0.30 * dom_br + 0.20 * dom_bg + 70.0 * hue_ratio
            ok = (
                hue_ratio >= 0.48
                and mean_s >= 120.0
                and mean_v >= 45.0
                and mean_b >= 65.0
                and dom_br >= 34.0
                and dom_bg >= 8.0
            )
            return ok, float(conf)

        hue_ratio = float(np.mean(((hue <= 13) | (hue >= 168)) & (sat >= 105)))
        dom_rb = float(mean_r - mean_b)
        dom_rg = float(mean_r - mean_g)
        conf = 0.30 * float(mean_s) + 0.30 * dom_rb + 0.20 * dom_rg + 70.0 * hue_ratio
        ok = (
            hue_ratio >= 0.50
            and mean_s >= 125.0
            and mean_v >= 55.0
            and mean_r >= 82.0
            and dom_rb >= 38.0
            and dom_rg >= 24.0
        )
        return ok, float(conf)

    def detect_holes(self, gray_roi, shape_mask_roi):
        if gray_roi.size == 0 or shape_mask_roi.size == 0:
            return 0, 0.0

        inner_mask = cv2.erode(shape_mask_roi, np.ones((5, 5), np.uint8), iterations=1)
        valid_pixels = gray_roi[inner_mask > 0]
        if valid_pixels.size < 30:
            return 0, 0.0

        dark_threshold = int(np.clip(np.percentile(valid_pixels, 22), 18, 95))
        dark_mask = cv2.inRange(gray_roi, 0, dark_threshold)
        dark_mask = cv2.bitwise_and(dark_mask, inner_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_area = max(float(np.count_nonzero(inner_mask)), 1.0)
        points = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(7.0, 0.00025 * mask_area):
                continue
            if area > 0.026 * mask_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.22:
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

            cy = y + h / 2.0
            if cy < self.image_height * MIN_CENTER_Y_RATIO:
                continue

            aspect = w / float(h)
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue

            bbox_area = float(w * h)
            hull = cv2.convexHull(contour)
            hull_area = max(cv2.contourArea(hull), 1.0)
            solidity = float(area / hull_area)
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
            bottom = y + h
            floor_bonus = 180.0 if bottom >= self.image_height * BOTTOM_FLOOR_BONUS_RATIO else 0.0
            high_penalty = 260.0 if cy < self.image_height * 0.34 and h < self.image_height * 0.20 else 0.0

            score = (
                1.85 * area
                + 430.0 * fill_ratio
                + 330.0 * extent
                + 260.0 * solidity
                + 2.0 * color_conf
                + 12.0 * min(holes, 10)
                + 2.0 * hole_pitch
                + floor_bonus
                - 0.18 * center_error
                - high_penalty
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
            )
            if best is None or obs.score > best.score:
                best = obs

        return best

    def detect(self, frame):
        self.image_height, self.image_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        red_mask_raw = self.build_red_mask(hsv, frame)
        blue_mask_raw = self.build_blue_mask(hsv, frame)
        blue_best = self.detect_best_for_color('blue', blue_mask_raw, frame, hsv, gray)
        red_best = self.detect_best_for_color('red', red_mask_raw, frame, hsv, gray)

        if blue_best is not None:
            return blue_best, red_mask_raw, blue_mask_raw
        return red_best, red_mask_raw, blue_mask_raw


class TrackingCubeV4(Node):
    def __init__(self):
        super().__init__('tracking_cube_v4')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)

        self.control_timer = self.create_timer(CONTROL_DT, self.control_loop)
        self.status_timer = self.create_timer(STATUS_DT, self.status_loop)

        self.detector = CubeDetector()
        self.state = 'WAIT_FOR_DATA'
        self.state_enter_time = time.monotonic()

        self.has_image = False
        self.has_scan = False
        self.has_odom = False
        self.image_width = 0
        self.image_height = 0
        self.front_dist = float('inf')

        self.target_visible = False
        self.target_obs: Optional[CubeObservation] = None
        self.prev_raw_obs: Optional[CubeObservation] = None
        self.target_seen_frames = 0
        self.centered_frames = 0
        self.last_target_seen_time = 0.0
        self.last_target_dir = 1.0
        self.last_target_color = ''
        self.last_bbox_h = 0
        self.last_area = 0.0
        self.last_pitch = 0.0

        self.filtered_cx = None
        self.filtered_cy = None
        self.filtered_bbox_h = None
        self.filtered_pitch = None

        self.shutdown_requested = False
        self.shutdown_reason = ''
        self.shutdown_count = 0
        self.wait_status_last = None

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        print('[BOOT] TrackingCube v4')
        print('[CMD] type H then Enter for emergency stop')

    def ready(self):
        return self.has_image and self.has_scan and self.has_odom

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_once(self):
        self.publish_cmd(0.0, 0.0)

    def stop_reliable(self, repeat=15, delay=0.025):
        for _ in range(repeat):
            self.stop_once()
            time.sleep(delay)

    def clamp_abs(self, value, min_abs, max_abs):
        if abs(value) < 1e-9:
            return 0.0
        return math.copysign(min(max(abs(value), min_abs), max_abs), value)

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

        front_count = max(2, int(round(math.radians(10.0) / max(msg.angle_increment, 1e-6))))
        front_indices = list(range(0, min(front_count + 1, n)))
        front_indices += list(range(max(0, n - front_count), n))
        vals = [ranges[i] for i in front_indices if math.isfinite(ranges[i]) and ranges[i] > 0.04]
        raw_front = min(vals) if vals else float('inf')

        if not self.has_scan:
            self.front_dist = raw_front
        else:
            alpha = 0.55
            self.front_dist = alpha * self.front_dist + (1.0 - alpha) * raw_front
        self.has_scan = True

    def odom_callback(self, _msg):
        self.has_odom = True

    def image_callback(self, msg):
        self.has_image = True
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('empty frame')
        except Exception:
            self.clear_target()
            return

        self.image_height, self.image_width = frame.shape[:2]
        obs, _, _ = self.detector.detect(frame)
        if obs is None:
            self.clear_target()
            return

        stable = False
        if self.prev_raw_obs is not None and self.prev_raw_obs.color == obs.color:
            raw_jump = abs(obs.cx - self.prev_raw_obs.cx)
            area_ratio = obs.area / max(self.prev_raw_obs.area, 1.0)
            stable = raw_jump <= MAX_RAW_JUMP_PX and 0.35 <= area_ratio <= 2.8

        self.target_seen_frames = self.target_seen_frames + 1 if stable else 1
        self.prev_raw_obs = obs

        alpha = 0.62
        if self.filtered_cx is None or self.last_target_color != obs.color or not stable:
            self.filtered_cx = obs.cx
            self.filtered_cy = obs.cy
            self.filtered_bbox_h = float(obs.bbox_h)
            self.filtered_pitch = float(obs.hole_pitch)
        else:
            self.filtered_cx = alpha * self.filtered_cx + (1.0 - alpha) * obs.cx
            self.filtered_cy = alpha * self.filtered_cy + (1.0 - alpha) * obs.cy
            self.filtered_bbox_h = alpha * self.filtered_bbox_h + (1.0 - alpha) * float(obs.bbox_h)
            self.filtered_pitch = alpha * self.filtered_pitch + (1.0 - alpha) * float(obs.hole_pitch)

        obs.cx = float(self.filtered_cx)
        obs.cy = float(self.filtered_cy)
        obs.bbox_h = int(max(obs.bbox_h, round(self.filtered_bbox_h)))
        obs.hole_pitch = float(max(obs.hole_pitch, self.filtered_pitch))

        self.target_visible = obs.area >= MIN_TRACK_AREA
        self.target_obs = obs if self.target_visible else None
        if not self.target_visible:
            self.target_seen_frames = 0
            return

        self.last_target_seen_time = time.monotonic()
        self.last_target_color = obs.color
        self.last_bbox_h = obs.bbox_h
        self.last_area = obs.area
        self.last_pitch = obs.hole_pitch

        err = obs.cx - self.image_width / 2.0
        if abs(err) > 2.0:
            self.last_target_dir = -1.0 if err > 0.0 else 1.0

    def clear_target(self):
        self.target_visible = False
        self.target_obs = None
        self.target_seen_frames = 0
        self.centered_frames = 0
        self.prev_raw_obs = None

    def target_error_pixels(self):
        if not self.target_visible or self.target_obs is None or self.image_width <= 0:
            return None
        return self.target_obs.cx - self.image_width / 2.0

    def target_recently_seen(self):
        return (time.monotonic() - self.last_target_seen_time) <= LOST_TIMEOUT

    def target_lockworthy(self):
        if not self.target_visible or self.target_obs is None or self.image_height <= 0:
            return False
        obs = self.target_obs
        return (
            obs.area >= MIN_LOCK_AREA
            and obs.cy >= self.image_height * MIN_CENTER_Y_RATIO
            and obs.fill_ratio >= MIN_FILL_RATIO
            and obs.extent >= MIN_EXTENT
            and obs.solidity >= MIN_SOLIDITY
        )

    def target_confirmed(self):
        return self.target_lockworthy() and self.target_seen_frames >= CONFIRM_FRAMES

    def target_close_enough(self):
        if not self.target_visible or self.target_obs is None:
            return False
        by_holes = self.target_obs.holes >= MIN_HOLES_FOR_RANGE and self.target_obs.hole_pitch >= STOP_HOLE_PITCH_PX
        by_bbox = self.target_obs.bbox_h >= STOP_BBOX_H_PX
        by_lidar = self.front_dist <= HARD_FRONT_STOP_DIST
        return by_holes or by_bbox or by_lidar

    def near_lost_should_stop(self):
        return (
            self.last_bbox_h >= CLOSE_LOST_BBOX_H_PX
            or self.last_pitch >= STOP_HOLE_PITCH_PX * 0.82
            or self.front_dist <= CLOSE_LOST_FRONT_DIST
        )

    def handle_search(self):
        if self.front_dist <= EMERGENCY_STOP_DIST:
            self.set_state('STOPPED', 'front obstacle too close')
            return

        if self.target_confirmed():
            self.stop_once()
            self.set_state('ALIGN', f'locked {self.target_obs.color}')
            return

        self.publish_cmd(0.0, SEARCH_ANG)

    def handle_align(self):
        if self.front_dist <= EMERGENCY_STOP_DIST:
            self.set_state('STOPPED', 'front obstacle too close')
            return

        if not self.target_visible or self.target_obs is None:
            if self.target_recently_seen() and not self.near_lost_should_stop():
                self.publish_cmd(0.0, self.last_target_dir * 0.08)
                return
            if self.near_lost_should_stop():
                self.set_state('STOPPED', 'target lost at close range')
                return
            self.set_state('SEARCH', 'target lost')
            return

        err = self.target_error_pixels()
        if err is None:
            self.stop_once()
            return

        if abs(err) <= ALIGN_DONE_PX:
            self.centered_frames += 1
            self.stop_once()
            if self.centered_frames >= CENTER_HOLD_FRAMES:
                self.set_state('APPROACH', f'centered {self.target_obs.color}')
            return

        self.centered_frames = 0
        err_norm = err / max(self.image_width / 2.0, 1.0)
        angular = self.clamp_abs(-0.30 * err_norm, ALIGN_MIN_ANG, ALIGN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_approach(self):
        if self.front_dist <= EMERGENCY_STOP_DIST:
            self.set_state('STOPPED', 'emergency front stop')
            return

        if not self.target_visible or self.target_obs is None:
            if self.near_lost_should_stop():
                self.set_state('STOPPED', 'target lost very close')
                return
            if self.target_recently_seen():
                self.stop_once()
                return
            self.set_state('SEARCH', 'target lost during approach')
            return

        if self.target_close_enough():
            self.set_state('STOPPED', f'arrived near {self.target_obs.color}')
            return

        err = self.target_error_pixels()
        if err is None:
            self.stop_once()
            return

        if abs(err) >= REALIGN_PX:
            self.stop_once()
            self.set_state('ALIGN', 'target off center')
            return

        err_norm = err / max(self.image_width / 2.0, 1.0)
        if abs(err) > APPROACH_ROTATE_ONLY_PX:
            angular = self.clamp_abs(-0.28 * err_norm, ALIGN_MIN_ANG, ALIGN_MAX_ANG)
            self.publish_cmd(0.0, angular)
            return

        if self.target_obs.bbox_h >= CLOSE_LOST_BBOX_H_PX or self.target_obs.hole_pitch >= SLOW_HOLE_PITCH_PX:
            linear = APPROACH_CREEP
        elif self.target_obs.bbox_h >= 120 or self.front_dist <= 0.28:
            linear = APPROACH_SLOW
        else:
            linear = APPROACH_FAST

        angular = 0.0 if abs(err) <= ALIGN_DONE_PX else self.clamp_abs(-0.18 * err_norm, 0.02, 0.09)
        self.publish_cmd(linear, angular)

    def control_loop(self):
        if self.shutdown_requested:
            self.stop_once()
            self.shutdown_count += 1
            if self.shutdown_count >= 10:
                self.stop_reliable()
                rclpy.shutdown()
            return

        if not self.ready():
            self.stop_once()
            return

        if self.state == 'WAIT_FOR_DATA':
            self.set_state('SEARCH', 'data ready')
            return
        if self.state == 'SEARCH':
            self.handle_search()
        elif self.state == 'ALIGN':
            self.handle_align()
        elif self.state == 'APPROACH':
            self.handle_approach()
        elif self.state == 'STOPPED':
            self.stop_once()
        else:
            self.stop_once()

    def status_loop(self):
        if not self.ready():
            wait_now = (self.has_scan, self.has_odom, self.has_image)
            if wait_now != self.wait_status_last:
                print(f'[WAIT] scan={self.has_scan} odom={self.has_odom} image={self.has_image}')
                self.wait_status_last = wait_now
            return

        if self.target_visible and self.target_obs is not None:
            err = self.target_error_pixels()
            print(
                f'[STATUS] {self.state} '
                f'color={self.target_obs.color} '
                f'err={err:.0f} '
                f'area={self.target_obs.area:.0f} '
                f'bbox_h={self.target_obs.bbox_h} '
                f'holes={self.target_obs.holes} '
                f'pitch={self.target_obs.hole_pitch:.1f} '
                f'front={self.front_dist:.2f}'
            )
        else:
            print(f'[STATUS] {self.state} color=none front={self.front_dist:.2f}')

    def destroy_node(self):
        try:
            self.stop_reliable()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrackingCubeV4()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.request_shutdown('KeyboardInterrupt')
        node.stop_reliable()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
