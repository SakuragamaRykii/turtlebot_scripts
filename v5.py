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


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

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

STOP_DISTANCE_CM = 6.0
STOP_DISTANCE_M = STOP_DISTANCE_CM / 100.0
EMERGENCY_STOP_CM = 3.0
EMERGENCY_STOP_M = EMERGENCY_STOP_CM / 100.0

ALIGN_PIXEL_TOL = 18
APPROACH_ROTATE_ONLY_PX = 95
REACQUIRE_PIXEL_TOL = 120
CENTER_HOLD_FRAMES = 3
CONFIRM_FRAMES = 2
LOST_TARGET_TIMEOUT = 0.55

CLOSE_BBOX_H_PX = 230
CLOSE_LOST_BBOX_H_PX = 180

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
    bgr_mask[
        (r > 110) &
        (r > g + 35) &
        (r > b + 35)
    ] = 255

    return clean_mask(cv2.bitwise_and(hsv_mask, bgr_mask))


def build_blue_mask(hsv, bgr):
    lower = np.array([85, 35, 40], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)

    hsv_mask = cv2.inRange(hsv, lower, upper)

    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)

    bgr_mask = np.zeros_like(hsv_mask)
    bgr_mask[
        (b > 80) &
        (b > r + 15) &
        (b > g - 5)
    ] = 255

    return clean_mask(cv2.bitwise_and(hsv_mask, bgr_mask))


def find_square(mask, min_area):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

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


class CubeDetectorV5:
    def __init__(self):
        self.red_tracker = StableTracker(alpha=0.10, max_missing=10, max_jump=55)
        self.blue_tracker = StableTracker(alpha=0.16, max_missing=10, max_jump=80)
        self.last_red_mask = None
        self.last_blue_mask = None

    def detect(self, frame):
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_mask = build_red_mask(hsv, frame)
        blue_mask = build_blue_mask(hsv, frame)
        self.last_red_mask = red_mask
        self.last_blue_mask = blue_mask

        red_box = find_square(red_mask, MIN_RED_AREA)
        blue_box = find_square(blue_mask, MIN_BLUE_AREA)

        red_box = self.red_tracker.update(red_box)
        blue_box = self.blue_tracker.update(blue_box)

        blue_obs = self.box_to_obs('blue', blue_box, self.blue_tracker.missing)
        red_obs = self.box_to_obs('red', red_box, self.red_tracker.missing)

        if blue_obs is not None:
            return blue_obs
        return red_obs

    def box_to_obs(self, color, box, missing):
        if box is None:
            return None
        x, y, w, h, score = box
        return CubeObservation(
            color=color,
            cx=float(x + w / 2.0),
            cy=float(y + h / 2.0),
            area=float(w * h),
            bbox_x=int(x),
            bbox_y=int(y),
            bbox_w=int(w),
            bbox_h=int(h),
            score=float(score),
            missing=int(missing),
        )


class TrackingCubeV5(Node):
    def __init__(self):
        super().__init__('tracking_cube_v5')

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

        self.control_timer = self.create_timer(CONTROL_DT, self.control_loop)
        self.status_timer = self.create_timer(STATUS_DT, self.status_loop)

        self.detector = CubeDetectorV5()

        self.has_scan = False
        self.has_odom = False
        self.has_image = False

        self.front_dist = float('inf')
        self.image_width = None
        self.image_height = None

        self.target_visible = False
        self.target_obs: Optional[CubeObservation] = None
        self.target_seen_frames = 0
        self.centered_frames = 0
        self.last_target_seen_time = 0.0
        self.last_target_dir = 1.0
        self.last_bbox_h = 0
        self.last_color = 'none'

        self.state = 'WAIT_FOR_DATA'
        self.state_enter_time = time.monotonic()
        self.wait_status_last = None

        self.shutdown_requested = False
        self.shutdown_reason = ''
        self.shutdown_count = 0

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        print('[BOOT] TrackingCube v5')
        print(f'[PARAM] STOP_DISTANCE_CM={STOP_DISTANCE_CM:.1f}  SPEED_SCALE={SPEED_SCALE:.1f}')
        print('[CMD] type H then Enter to stop and exit')

    def ready(self):
        return self.has_scan and self.has_odom and self.has_image

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

    def odom_callback(self, _msg):
        self.has_odom = True

    def image_callback(self, msg):
        self.has_image = True
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('decoded frame is None')
        except Exception:
            self.clear_target()
            return

        self.image_height, self.image_width = frame.shape[:2]
        obs = self.detector.detect(frame)

        if obs is None:
            self.clear_target()
            return

        self.target_visible = True
        self.target_obs = obs
        self.target_seen_frames += 1
        self.last_target_seen_time = time.monotonic()
        self.last_bbox_h = obs.bbox_h
        self.last_color = obs.color

        err = obs.cx - self.image_width / 2.0
        if abs(err) > 2.0:
            self.last_target_dir = -1.0 if err > 0.0 else 1.0

    def clear_target(self):
        self.target_visible = False
        self.target_obs = None
        self.target_seen_frames = 0
        self.centered_frames = 0

    def target_error_pixels(self):
        if not self.target_visible or self.target_obs is None or self.image_width is None:
            return None
        return self.target_obs.cx - self.image_width / 2.0

    def target_recently_seen(self):
        return (time.monotonic() - self.last_target_seen_time) <= LOST_TARGET_TIMEOUT

    def target_confirmed(self):
        return (
            self.target_visible
            and self.target_obs is not None
            and self.target_obs.missing <= 2
            and self.target_seen_frames >= CONFIRM_FRAMES
        )

    def target_close_enough(self):
        return self.front_dist <= STOP_DISTANCE_M or self.last_bbox_h >= CLOSE_BBOX_H_PX

    def lost_close_should_stop(self):
        return self.front_dist <= STOP_DISTANCE_M + 0.03 or self.last_bbox_h >= CLOSE_LOST_BBOX_H_PX

    def handle_search(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if self.target_confirmed():
            self.stop_robot_once()
            self.set_state('ALIGN', f'found {self.target_obs.color}')
            return

        self.publish_cmd(0.0, SEARCH_ANG)

    def handle_align(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if not self.target_visible or self.target_obs is None:
            if self.lost_close_should_stop():
                self.set_state('STOPPED', 'target lost near stop distance')
                return
            if self.target_recently_seen():
                self.publish_cmd(0.0, self.last_target_dir * RECOVER_ANG)
                return
            self.set_state('SEARCH', 'target lost')
            return

        err = self.target_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) <= ALIGN_PIXEL_TOL:
            self.centered_frames += 1
            self.stop_robot_once()
            if self.centered_frames >= CENTER_HOLD_FRAMES:
                self.set_state('APPROACH', f'centered {self.target_obs.color}')
            return

        self.centered_frames = 0
        err_norm = err / max(self.image_width / 2.0, 1.0)
        angular = self.clamp_abs(-0.30 * err_norm, ALIGN_MIN_ANG, ALIGN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_approach(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if not self.target_visible or self.target_obs is None:
            if self.lost_close_should_stop():
                self.set_state('STOPPED', 'target lost near stop distance')
                return
            if self.target_recently_seen():
                self.stop_robot_once()
                return
            self.set_state('SEARCH', 'target lost during approach')
            return

        if self.target_close_enough():
            self.set_state('STOPPED', f'arrived near {self.target_obs.color}')
            return

        err = self.target_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) >= REACQUIRE_PIXEL_TOL:
            self.stop_robot_once()
            self.set_state('ALIGN', 'target off center')
            return

        err_norm = err / max(self.image_width / 2.0, 1.0)
        if abs(err) > APPROACH_ROTATE_ONLY_PX:
            angular = self.clamp_abs(-0.32 * err_norm, ALIGN_MIN_ANG, ALIGN_MAX_ANG)
            self.publish_cmd(0.0, angular)
            return

        linear = APPROACH_SLOW_SPEED if self.front_dist <= STOP_DISTANCE_M + 0.10 or self.target_obs.bbox_h >= 160 else APPROACH_FAST_SPEED
        angular = 0.0 if abs(err) <= ALIGN_PIXEL_TOL else self.clamp_abs(-0.24 * err_norm, 0.02 * SPEED_SCALE, 0.10 * SPEED_SCALE)
        self.publish_cmd(linear, angular)

    def control_loop(self):
        if self.shutdown_requested:
            self.stop_robot_once()
            self.shutdown_count += 1
            if self.shutdown_count >= 10:
                self.stop_robot_reliable()
                rclpy.shutdown()
            return

        if not self.ready():
            self.stop_robot_once()
            return

        if self.state == 'WAIT_FOR_DATA':
            self.stop_robot_once()
            self.set_state('SEARCH', 'data ready')
            return

        if self.state == 'SEARCH':
            self.handle_search()
        elif self.state == 'ALIGN':
            self.handle_align()
        elif self.state == 'APPROACH':
            self.handle_approach()
        elif self.state == 'STOPPED':
            self.stop_robot_once()
        else:
            self.stop_robot_once()

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
                f'bbox=({self.target_obs.bbox_w}x{self.target_obs.bbox_h}) '
                f'score={self.target_obs.score:.0f} '
                f'missing={self.target_obs.missing} '
                f'front={self.front_dist:.3f}m '
                f'stop={STOP_DISTANCE_CM:.1f}cm'
            )
        else:
            print(f'[STATUS] {self.state} color=none front={self.front_dist:.3f}m stop={STOP_DISTANCE_CM:.1f}cm')

    def destroy_node(self):
        try:
            self.stop_robot_reliable()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrackingCubeV5()
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
