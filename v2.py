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
    fill_ratio: float
    extent: float
    solidity: float
    color_conf: float
    score: float


class CubeFinderV2(Node):
    def __init__(self):
        super().__init__('cube_finder_v2')

        # Speed: faster than the old V2/V3, but still controllable.
        self.SEARCH_ANG = 0.78
        self.SEARCH_TRACK_MAX_ANG = 0.92
        self.ALIGN_MIN_ANG = 0.12
        self.ALIGN_MAX_ANG = 0.55
        self.RECOVER_ANG = 0.42
        self.APPROACH_FAST = 0.15
        self.APPROACH_SLOW = 0.10

        # Vision gates.
        self.MIN_TRACK_AREA = 260.0
        self.MIN_LOCK_AREA = 460.0
        self.MIN_BBOX = 16
        self.MIN_CENTER_Y_RATIO = 0.20
        self.MIN_FILL_RATIO = 0.24
        self.MIN_EXTENT = 0.20
        self.MIN_SOLIDITY = 0.72
        self.MAX_ASPECT = 1.95
        self.MIN_ASPECT = 0.52
        self.MAX_RAW_JUMP_PX = 120.0
        self.CONFIRM_FRAMES = 3
        self.CENTER_LOCK_PX = 95
        self.ALIGN_DONE_PX = 20
        self.ROTATE_ONLY_PX = 120
        self.REALIGN_PX = 95

        # Stop policy. Goal: get very close, even slightly touching is acceptable.
        self.MIN_HOLES_FOR_RANGE = 3
        self.SLOW_HOLE_PITCH_PX = 16.0
        self.STOP_HOLE_PITCH_PX = 21.5
        self.STOP_BBOX_H_PX = 215
        self.CLOSE_LOST_BBOX_H_PX = 165
        self.HARD_FRONT_STOP_DIST = 0.105
        self.CLOSE_LOST_FRONT_DIST = 0.15
        self.EMERGENCY_STOP_DIST = 0.085
        self.LOST_TIMEOUT = 0.45

        self.CONTROL_DT = 0.05
        self.STATUS_DT = 1.0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.control_timer = self.create_timer(self.CONTROL_DT, self.control_loop)
        self.status_timer = self.create_timer(self.STATUS_DT, self.status_loop)

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        self.state = 'WAIT_FOR_DATA'
        self.state_enter_time = time.monotonic()

        self.has_scan = False
        self.has_image = False
        self.front_dist = float('inf')
        self.left_front_dist = float('inf')
        self.right_front_dist = float('inf')

        self.image_width = None
        self.image_height = None
        self.target_visible = False
        self.target_obs: Optional[CubeObservation] = None
        self.prev_raw_obs: Optional[CubeObservation] = None
        self.target_seen_frames = 0
        self.last_target_seen_time = 0.0
        self.last_target_dir = -1.0
        self.last_target_color = None
        self.last_bbox_h = 0
        self.last_target_pitch = 0.0
        self.last_target_area = 0.0

        self.filtered_cx = None
        self.filtered_cy = None
        self.filtered_pitch = None
        self.filtered_bbox_h = None

        self.shutdown_requested = False
        self.shutdown_reason = ''
        self.shutdown_count = 0
        self.wait_status_last = None

        print('[INFO] start | H to stop and exit')

    # ------------------------------- utils -------------------------------
    def ready(self):
        return self.has_scan and self.has_image

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def clamp_abs(self, value, min_abs, max_abs):
        mag = min(max(abs(value), min_abs), max_abs)
        return math.copysign(mag, value)

    def publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop_robot_reliable(self, repeat=6, delay=0.02):
        for _ in range(repeat):
            self.publish_cmd(0.0, 0.0)
            time.sleep(delay)

    def stop_robot_once(self):
        self.publish_cmd(0.0, 0.0)

    def set_state(self, new_state, text=''):
        if self.state == new_state:
            return
        self.state = new_state
        self.state_enter_time = time.monotonic()
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

    def target_recently_seen(self):
        return (time.monotonic() - self.last_target_seen_time) <= self.LOST_TIMEOUT

    def target_error_pixels(self):
        if not self.target_visible or self.target_obs is None or self.image_width is None:
            return None
        return self.target_obs.cx - self.image_width / 2.0

    def target_lockworthy(self):
        if not self.target_visible or self.target_obs is None or self.image_width is None:
            return False
        obs = self.target_obs
        err = abs(obs.cx - self.image_width / 2.0)
        return (
            obs.area >= self.MIN_LOCK_AREA
            and err <= self.CENTER_LOCK_PX
            and obs.fill_ratio >= self.MIN_FILL_RATIO
            and obs.extent >= self.MIN_EXTENT
            and obs.solidity >= self.MIN_SOLIDITY
        )

    def target_confirmed(self):
        return self.target_lockworthy() and self.target_seen_frames >= self.CONFIRM_FRAMES

    def target_close_enough(self):
        if not self.target_visible or self.target_obs is None:
            return False
        by_holes = (
            self.target_obs.holes >= self.MIN_HOLES_FOR_RANGE
            and self.target_obs.hole_pitch >= self.STOP_HOLE_PITCH_PX
        )
        by_bbox = self.target_obs.bbox_h >= self.STOP_BBOX_H_PX
        by_lidar = self.front_dist <= self.HARD_FRONT_STOP_DIST
        return by_holes or by_bbox or by_lidar

    def near_lost_should_stop(self):
        return (
            self.last_bbox_h >= self.CLOSE_LOST_BBOX_H_PX
            or self.last_target_pitch >= self.STOP_HOLE_PITCH_PX * 0.85
            or self.front_dist <= self.CLOSE_LOST_FRONT_DIST
        )

    # ------------------------------ callbacks ----------------------------
    def console_loop(self):
        while True:
            try:
                cmd = input().strip().lower()
            except Exception:
                return
            if cmd == 'h':
                self.request_shutdown('收到 H 指令，停车并退出')
                return
            if cmd in ('q', 'quit', 'exit'):
                self.request_shutdown('收到退出指令')
                return

    def scan_callback(self, msg):
        front_vals = list(msg.ranges[0:10]) + list(msg.ranges[350:360])
        left_front_vals = list(msg.ranges[15:45])
        right_front_vals = list(msg.ranges[315:345])

        def valid_min(vals):
            good = [x for x in vals if math.isfinite(x) and x > 0.05]
            return min(good) if good else float('inf')

        raw_front = valid_min(front_vals)
        raw_left = valid_min(left_front_vals)
        raw_right = valid_min(right_front_vals)

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
        obs = self.detect_best_cube(frame)

        if obs is None:
            self.target_visible = False
            self.target_obs = None
            self.target_seen_frames = 0
            self.prev_raw_obs = None
            return

        stable = False
        if self.prev_raw_obs is not None and self.prev_raw_obs.color == obs.color:
            raw_jump = abs(obs.cx - self.prev_raw_obs.cx)
            area_ratio = obs.area / max(self.prev_raw_obs.area, 1.0)
            if raw_jump <= self.MAX_RAW_JUMP_PX and 0.35 <= area_ratio <= 2.9:
                stable = True

        self.target_seen_frames = self.target_seen_frames + 1 if stable else 1
        self.prev_raw_obs = obs

        alpha = 0.55
        if self.filtered_cx is None or self.last_target_color != obs.color or not stable:
            self.filtered_cx = obs.cx
            self.filtered_cy = obs.cy
            self.filtered_pitch = obs.hole_pitch
            self.filtered_bbox_h = float(obs.bbox_h)
        else:
            self.filtered_cx = alpha * self.filtered_cx + (1.0 - alpha) * obs.cx
            self.filtered_cy = alpha * self.filtered_cy + (1.0 - alpha) * obs.cy
            self.filtered_pitch = alpha * self.filtered_pitch + (1.0 - alpha) * obs.hole_pitch
            self.filtered_bbox_h = alpha * self.filtered_bbox_h + (1.0 - alpha) * float(obs.bbox_h)

        obs.cx = float(self.filtered_cx)
        obs.cy = float(self.filtered_cy)
        obs.hole_pitch = float(max(obs.hole_pitch, self.filtered_pitch))
        obs.bbox_h = int(max(obs.bbox_h, round(self.filtered_bbox_h)))

        self.target_visible = obs.area >= self.MIN_TRACK_AREA
        self.target_obs = obs if self.target_visible else None
        if not self.target_visible:
            self.target_seen_frames = 0
            return

        self.last_target_seen_time = time.monotonic()
        self.last_target_dir = -1.0 if (obs.cx - self.image_width / 2.0) > 0.0 else 1.0
        self.last_target_color = obs.color
        self.last_bbox_h = obs.bbox_h
        self.last_target_pitch = obs.hole_pitch
        self.last_target_area = obs.area

    # ------------------------------ vision ------------------------------
    def build_red_mask(self, hsv, bgr):
        lower_red_1 = np.array([0, 110, 55], dtype=np.uint8)
        upper_red_1 = np.array([9, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([172, 110, 55], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2),
        )

        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]
        rgb_mask = np.zeros_like(hsv_mask)
        red_dom = (r >= 90) & (r > g + 36) & (r > b + 52)
        rgb_mask[red_dom] = 255
        mask = cv2.bitwise_and(hsv_mask, rgb_mask)
        return mask

    def build_blue_mask(self, hsv, bgr):
        lower_blue = np.array([107, 115, 40], dtype=np.uint8)
        upper_blue = np.array([126, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]
        rgb_mask = np.zeros_like(hsv_mask)
        blue_dom = (b >= 70) & (b > g + 20) & (b > r + 45)
        rgb_mask[blue_dom] = 255
        mask = cv2.bitwise_and(hsv_mask, rgb_mask)
        return mask

    def color_signature(self, color, roi_bgr, roi_hsv, roi_mask):
        pixels = roi_bgr[roi_mask > 0]
        hsv_pixels = roi_hsv[roi_mask > 0]
        if pixels.size == 0 or hsv_pixels.size == 0:
            return False, 0.0

        mean_b, mean_g, mean_r = np.mean(pixels, axis=0)
        mean_h, mean_s, mean_v = np.mean(hsv_pixels, axis=0)
        hue = hsv_pixels[:, 0]
        sat = hsv_pixels[:, 1]

        if color == 'blue':
            hue_ratio = float(np.mean((hue >= 108) & (hue <= 124) & (sat >= 120)))
            dom_br = float(mean_b - mean_r)
            dom_bg = float(mean_b - mean_g)
            conf = 0.34 * float(mean_s) + 0.28 * dom_br + 0.18 * dom_bg + 60.0 * hue_ratio
            ok = (
                hue_ratio >= 0.58
                and mean_s >= 145.0
                and mean_v >= 45.0
                and mean_b >= 75.0
                and dom_br >= 45.0
                and dom_bg >= 18.0
            )
            return ok, float(conf)

        hue_ratio = float(np.mean(((hue <= 10) | (hue >= 170)) & (sat >= 115)))
        dom_rg = float(mean_r - mean_g)
        dom_rb = float(mean_r - mean_b)
        conf = 0.33 * float(mean_s) + 0.28 * dom_rb + 0.18 * dom_rg + 60.0 * hue_ratio
        ok = (
            hue_ratio >= 0.46
            and mean_s >= 135.0
            and mean_v >= 65.0
            and mean_r >= 92.0
            and dom_rb >= 46.0
            and dom_rg >= 26.0
        )
        return ok, float(conf)

    def preprocess_mask(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask

    def detect_best_for_color(self, color, color_mask, frame, hsv, gray):
        mask = self.preprocess_mask(color_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_TRACK_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < self.MIN_BBOX or h < self.MIN_BBOX:
                continue

            cy = y + h / 2.0
            if self.image_height is not None and cy < self.image_height * self.MIN_CENTER_Y_RATIO:
                continue

            aspect = w / float(h)
            if aspect < self.MIN_ASPECT or aspect > self.MAX_ASPECT:
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
            roi_shape_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(roi_shape_mask, [shifted], -1, 255, thickness=-1)
            fill_ratio = float(np.count_nonzero(roi_shape_mask) / bbox_area)
            if fill_ratio < self.MIN_FILL_RATIO:
                continue

            extent = float(area / bbox_area)
            if extent < self.MIN_EXTENT:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
            if len(approx) < 4 or len(approx) > 12:
                continue

            roi_gray = gray[y:y + h, x:x + w]
            roi_hsv = hsv[y:y + h, x:x + w]
            roi_bgr = frame[y:y + h, x:x + w]
            roi_mask = roi_shape_mask

            color_ok, color_conf = self.color_signature(color, roi_bgr, roi_hsv, roi_mask)
            if not color_ok:
                continue

            holes, hole_pitch = self.detect_holes(roi_gray, roi_mask)

            center_penalty = 0.0
            if self.image_width is not None:
                err = abs((x + w / 2.0) - self.image_width / 2.0)
                center_penalty = 0.14 * err

            score = (
                1.75 * area
                + 420.0 * fill_ratio
                + 320.0 * extent
                + 250.0 * solidity
                + 1.6 * color_conf
                + 10.0 * min(holes, 10)
                + 2.0 * hole_pitch
                - center_penalty
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

    def detect_best_cube(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        blue_best = self.detect_best_for_color('blue', self.build_blue_mask(hsv, frame), frame, hsv, gray)
        red_best = self.detect_best_for_color('red', self.build_red_mask(hsv, frame), frame, hsv, gray)

        # User requirement: if both are valid, prefer blue.
        if blue_best is not None:
            return blue_best
        return red_best

    def detect_holes(self, gray_roi, color_mask_roi):
        if gray_roi.size == 0 or color_mask_roi.size == 0:
            return 0, 0.0

        inner_mask = cv2.erode(color_mask_roi, np.ones((5, 5), np.uint8), iterations=1)
        valid_pixels = gray_roi[inner_mask > 0]
        if valid_pixels.size < 30:
            return 0, 0.0

        dark_threshold = int(np.clip(np.percentile(valid_pixels, 22), 18, 95))
        dark_mask = cv2.inRange(gray_roi, 0, dark_threshold)
        dark_mask = cv2.bitwise_and(dark_mask, inner_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pts = []
        mask_area = max(float(np.count_nonzero(inner_mask)), 1.0)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(7.0, 0.00025 * mask_area):
                continue
            if area > 0.025 * mask_area:
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
            cx = float(m['m10'] / m['m00'])
            cy = float(m['m01'] / m['m00'])
            pts.append((cx, cy))

        if len(pts) < 2:
            return len(pts), 0.0

        arr = np.array(pts, dtype=np.float32)
        nn = []
        for i in range(len(arr)):
            d = np.linalg.norm(arr - arr[i], axis=1)
            d = d[d > 1.0]
            if d.size > 0:
                nn.append(float(np.min(d)))
        hole_pitch = float(np.median(nn)) if nn else 0.0
        return len(pts), hole_pitch

    # ------------------------------ control -----------------------------
    def handle_search(self):
        if self.front_dist <= self.EMERGENCY_STOP_DIST:
            self.request_shutdown('前方过近，紧急停车')
            return

        if self.target_visible and self.target_obs is not None and self.image_width is not None:
            err = self.target_error_pixels()
            if self.target_confirmed():
                self.stop_robot_once()
                self.set_state('ALIGN_TARGET', f'锁定 {self.target_obs.color} 方块，开始对准')
                return

            if abs(err) <= self.CENTER_LOCK_PX:
                self.publish_cmd(0.0, -0.22 if err > 0 else 0.22)
                return

            ang = self.clamp_abs(-0.0034 * err, 0.18, self.SEARCH_TRACK_MAX_ANG)
            self.publish_cmd(0.0, ang)
            return

        self.publish_cmd(0.0, self.SEARCH_ANG)

    def handle_align(self):
        if self.front_dist <= self.EMERGENCY_STOP_DIST:
            self.request_shutdown('前方过近，紧急停车')
            return

        if not self.target_visible or self.target_obs is None or self.image_width is None:
            if self.target_recently_seen():
                self.publish_cmd(0.0, self.last_target_dir * self.RECOVER_ANG)
                return
            self.set_state('SEARCH_TARGET', '目标丢失，重新搜索')
            return

        err = self.target_error_pixels()
        if abs(err) <= self.ALIGN_DONE_PX:
            if self.target_seen_frames >= 2:
                self.stop_robot_once()
                self.set_state('APPROACH_TARGET', f'{self.target_obs.color} 方块已对正，开始前进')
                return
            self.stop_robot_once()
            return

        ang = self.clamp_abs(-0.0038 * err, self.ALIGN_MIN_ANG, self.ALIGN_MAX_ANG)
        self.publish_cmd(0.0, ang)

    def handle_approach(self):
        if self.front_dist <= self.EMERGENCY_STOP_DIST:
            self.request_shutdown('前方过近，紧急停车')
            return

        if not self.target_visible or self.target_obs is None or self.image_width is None:
            if self.near_lost_should_stop():
                self.request_shutdown('近距离丢失目标，停车')
                return
            if self.target_recently_seen():
                self.publish_cmd(0.0, self.last_target_dir * self.RECOVER_ANG)
                return
            self.set_state('SEARCH_TARGET', '前进时目标丢失，重新搜索')
            return

        if self.target_close_enough():
            self.request_shutdown(
                f'到达 {self.target_obs.color} 方块前方，holes={self.target_obs.holes} pitch={self.target_obs.hole_pitch:.1f}'
            )
            return

        err = self.target_error_pixels()
        if abs(err) >= self.REALIGN_PX:
            self.set_state('ALIGN_TARGET', '偏差过大，重新对准')
            return

        if abs(err) > self.ROTATE_ONLY_PX:
            ang = self.clamp_abs(-0.0033 * err, self.ALIGN_MIN_ANG, self.ALIGN_MAX_ANG)
            self.publish_cmd(0.0, ang)
            return

        if self.target_obs.hole_pitch >= self.SLOW_HOLE_PITCH_PX or self.target_obs.bbox_h >= 150:
            linear = self.APPROACH_SLOW
        else:
            linear = self.APPROACH_FAST

        if abs(err) <= self.ALIGN_DONE_PX:
            angular = 0.0
        else:
            angular = self.clamp_abs(-0.0018 * err, 0.04, 0.18)

        self.publish_cmd(linear, angular)

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
            self.set_state('SEARCH_TARGET', '传感器准备完成，开始搜索')
            return

        if self.state == 'SEARCH_TARGET':
            self.handle_search()
        elif self.state == 'ALIGN_TARGET':
            self.handle_align()
        elif self.state == 'APPROACH_TARGET':
            self.handle_approach()
        else:
            self.stop_robot_once()

    def status_loop(self):
        if not self.ready():
            wait_now = (self.has_scan, self.has_image)
            if wait_now != self.wait_status_last:
                print(f'[WAIT] scan={self.has_scan} image={self.has_image}')
                self.wait_status_last = wait_now
            return

        if self.shutdown_requested:
            print(f'[STATUS] stopping | reason={self.shutdown_reason}')
            return

        if self.target_visible and self.target_obs is not None and self.image_width is not None:
            err = self.target_error_pixels()
            print(
                f'[STATUS] {self.state} '
                f'color={self.target_obs.color} '
                f'err={err:.0f} '
                f'area={self.target_obs.area:.0f} '
                f'bbox_h={self.target_obs.bbox_h} '
                f'holes={self.target_obs.holes} '
                f'pitch={self.target_obs.hole_pitch:.1f} '
                f'conf={self.target_obs.color_conf:.0f} '
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
    node = CubeFinderV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.request_shutdown('KeyboardInterrupt')
        try:
            node.stop_robot_reliable(repeat=15, delay=0.03)
        except Exception:
            pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
