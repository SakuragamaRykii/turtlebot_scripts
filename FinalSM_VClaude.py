#!/usr/bin/env python3
"""
TurtleBot3 WafflePi – Cube Zone Sorter
=======================================
States
------
WAIT_FOR_DATA  : Spin until LiDAR + odometry + camera are all live.
INIT_SWEEP     : Rotate 360° with LiDAR to measure arena wall distances.
IDLE           : Stationary. Waits for 's' key to begin search.
SEARCH         : Rotate 360°, log cube colours and zone correctness.
TURN_TO_APPROACH : Rotate to face the misplaced cube.
APPROACH       : Drive straight toward the cube; stop when close.
TURN_HOME      : Rotate to face the origin (0, 0).
RETURN_HOME    : Drive back to the origin.

Key bindings (console, any time)
---------------------------------
s  – (from IDLE) start search sweep
h  – halt immediately and return to IDLE

Zone convention
---------------
The arena is split down the middle by the robot's Y-axis at start-up.
  right half  (|local_yaw| < 90°)  → RED  zone
  left  half  (|local_yaw| >= 90°) → BLUE zone
Adjust ZONE_BOUNDARY_YAW below if your arena layout differs.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CubeEntry:
    color: str        # 'red' | 'blue'
    angle_rad: float  # local yaw when cube was spotted
    zone: str         # 'red_zone' | 'blue_zone'
    correct: bool     # True if cube is already in its matching zone


@dataclass
class ArenaMap:
    """Median wall distances (m) from the starting position."""
    front: float = 1.0
    back: float  = 1.0
    left: float  = 1.0
    right: float = 1.0

    @property
    def width(self) -> float:
        return self.left + self.right

    @property
    def depth(self) -> float:
        return self.front + self.back


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class CubeSorterNode(Node):

    # ── motion parameters ────────────────────────────────────────────────────
    DRIVE_SPEED       = 0.07    # m/s  forward cruise
    BACK_SPEED        = -0.06   # m/s  reverse
    SWEEP_ANG         = 0.20    # rad/s during 360° sweeps
    ALIGN_ANG_MIN     = 0.04    # rad/s minimum align angular speed
    ALIGN_ANG_MAX     = 0.18    # rad/s maximum align angular speed
    TURN_ANG_MIN      = 0.05    # rad/s minimum turn angular speed
    TURN_ANG_MAX      = 0.18    # rad/s maximum turn angular speed

    # ── safety distances ─────────────────────────────────────────────────────
    WALL_CAUTION_DIST = 0.30    # m  slow / avoid
    WALL_STOP_DIST    = 0.20    # m  stop forward motion
    EMERGENCY_DIST    = 0.15    # m  hard stop any motion

    # ── angle tolerances ─────────────────────────────────────────────────────
    YAW_TOL           = math.radians(3.0)
    SWEEP_DONE_TOL    = math.radians(5.0)

    # ── timing ───────────────────────────────────────────────────────────────
    CONTROL_DT        = 0.05    # s  control loop period
    STATUS_DT         = 1.0     # s  status print period

    # ── vision parameters ────────────────────────────────────────────────────
    MIN_CONTOUR_AREA  = 260     # px²
    MIN_BBOX_W        = 14      # px
    MIN_BBOX_H        = 14      # px
    MIN_FILL_RATIO    = 0.28
    MIN_EXTENT        = 0.22
    MIN_SOLIDITY      = 0.70
    MIN_CENTER_Y      = 0.18    # fraction of image height (ignore sky)
    CONFIRM_FRAMES    = 3       # stable frames before logging a cube
    LOST_TIMEOUT      = 0.80    # s before giving up on a lost cube

    # Stop approach when cube bbox height reaches this many pixels
    APPROACH_STOP_BBOX_H = 150  # px
    # … or when LiDAR front distance drops to this
    APPROACH_STOP_DIST   = 0.25  # m
    # Pixel tolerance for "centred" alignment
    ALIGN_PIXEL_TOL      = 22   # px
    # Pixel error above which we rotate-only (no translation)
    ALIGN_ROTATE_ONLY_PX = 55   # px

    # ── zone boundary ────────────────────────────────────────────────────────
    # Heading angles whose |value| < ZONE_BOUNDARY_YAW look into the RED zone.
    ZONE_BOUNDARY_YAW = math.pi / 2.0  # rad

    # ── search logging window ─────────────────────────────────────────────────
    SEARCH_LOG_WINDOW = math.radians(15.0)  # don't re-log within this arc

    # ── return-home arrival threshold ────────────────────────────────────────
    HOME_ARRIVE_DIST  = 0.10    # m

    # ────────────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__('cube_sorter')

        # ROS interfaces
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan',
                                 self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom',
                                 self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(CompressedImage,
                                 '/camera/image_raw/compressed',
                                 self.image_cb, qos_profile_sensor_data)
        self.create_timer(self.CONTROL_DT, self.control_loop)
        self.create_timer(self.STATUS_DT,  self.status_loop)

        # Sensor-ready flags
        self.has_scan  = False
        self.has_odom  = False
        self.has_image = False

        # LiDAR (smoothed)
        self.front_dist = float('inf')
        self.back_dist  = float('inf')
        self.left_dist  = float('inf')
        self.right_dist = float('inf')

        # Odometry – world frame
        self.world_x   = 0.0
        self.world_y   = 0.0
        self.world_yaw = 0.0
        # Odometry – local frame (zeroed at first odom message)
        self.local_x   = 0.0
        self.local_y   = 0.0
        self.local_yaw = 0.0
        self._init_wx  = None
        self._init_wy  = None
        self._init_wyaw = None

        # Camera
        self.img_w: Optional[int] = None
        self.img_h: Optional[int] = None
        self.target_visible = False
        self.target_obs: Optional[dict] = None
        self.target_frames = 0
        self.last_seen_time = 0.0
        self.last_turn_dir  = 1.0   # +1 = CCW, -1 = CW

        # Filtered camera positions
        self._fcx: Optional[float] = None
        self._fcy: Optional[float] = None
        self._fbbox_h: float = 0.0
        self._prev_obs: Optional[dict] = None

        # Arena map (set during INIT_SWEEP)
        self.arena: Optional[ArenaMap] = None
        self._init_front: List[float] = []
        self._init_back:  List[float] = []
        self._init_left:  List[float] = []
        self._init_right: List[float] = []
        self._init_accum  = 0.0
        self._init_prev   = 0.0

        # Cube lists
        self.correct_cubes: List[CubeEntry] = []
        self.wrong_cubes:   List[CubeEntry] = []

        # Search sweep state
        self._sweep_accum = 0.0
        self._sweep_prev  = 0.0
        self._sweep_logged: List[float] = []

        # Approach / turn state
        self.approach_target: Optional[CubeEntry] = None
        self._turn_target_yaw = 0.0

        # State machine
        self.state = 'WAIT_FOR_DATA'
        self._state_enter = time.monotonic()

        # Console thread
        self._keys: List[str] = []
        self._key_lock = threading.Lock()
        threading.Thread(target=self._console_loop, daemon=True).start()

        print('[BOOT] CubeSorter started')
        print('[CMD]  s = start search | h = halt to IDLE')

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def _q2yaw(q) -> float:
        return math.atan2(2 * (q.w * q.z + q.x * q.y),
                          1 - 2 * (q.y ** 2 + q.z ** 2))

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _clamp_abs(self, v, mn, mx):
        if abs(v) < 1e-9:
            return 0.0
        return math.copysign(self._clamp(abs(v), mn, mx), v)

    def _pub(self, lin=0.0, ang=0.0):
        m = Twist()
        m.linear.x  = float(lin)
        m.angular.z = float(ang)
        self.cmd_pub.publish(m)

    def _stop(self):
        self._pub(0.0, 0.0)

    def _ready(self) -> bool:
        return self.has_scan and self.has_odom and self.has_image

    def _set_state(self, new: str, note: str = ''):
        if self.state == new:
            return
        old = self.state
        self.state = new
        self._state_enter = time.monotonic()
        suffix = f' | {note}' if note else ''
        print(f'[STATE] {old} -> {new}{suffix}')

        # Entry side-effects
        if new == 'INIT_SWEEP':
            self._init_prev  = self.local_yaw
            self._init_accum = 0.0
            self._init_front.clear(); self._init_back.clear()
            self._init_left.clear();  self._init_right.clear()
        elif new == 'SEARCH':
            self._sweep_prev  = self.local_yaw
            self._sweep_accum = 0.0
            self._sweep_logged.clear()
            self.correct_cubes.clear()
            self.wrong_cubes.clear()
        elif new == 'TURN_TO_APPROACH':
            self._turn_target_yaw = self.approach_target.angle_rad
        elif new == 'TURN_HOME':
            dx = -self.local_x
            dy = -self.local_y
            self._turn_target_yaw = math.atan2(dy, dx)

    # ── console ───────────────────────────────────────────────────────────────

    def _console_loop(self):
        while True:
            try:
                line = input().strip().lower()
            except EOFError:
                return
            with self._key_lock:
                self._keys.append(line)

    def _pop_keys(self) -> List[str]:
        with self._key_lock:
            k, self._keys = self._keys, []
        return k

    # ── zone logic ────────────────────────────────────────────────────────────

    def _yaw_to_zone(self, yaw: float) -> str:
        """Return which arena zone the robot is looking into at this heading."""
        if abs(self._norm(yaw)) < self.ZONE_BOUNDARY_YAW:
            return 'red_zone'
        return 'blue_zone'

    @staticmethod
    def _cube_correct(color: str, zone: str) -> bool:
        return (color == 'red'  and zone == 'red_zone') or \
               (color == 'blue' and zone == 'blue_zone')

    # ── safety checks ─────────────────────────────────────────────────────────

    def _emergency(self) -> bool:
        return (self.front_dist <= self.EMERGENCY_DIST or
                self.back_dist  <= self.EMERGENCY_DIST)

    def _front_blocked(self) -> bool:
        return self.front_dist <= self.WALL_STOP_DIST

    # ── sensor callbacks ──────────────────────────────────────────────────────

    def scan_cb(self, msg: LaserScan):
        r = msg.ranges
        n = len(r)

        def vmin(indices):
            vs = [r[i] for i in indices
                  if 0 <= i < n and math.isfinite(r[i]) and r[i] > 0.05]
            return min(vs) if vs else float('inf')

        front  = list(range(0, 15))  + list(range(n - 15, n))
        back   = list(range(n // 2 - 15, n // 2 + 15))
        left   = list(range(80, 100))
        right  = list(range(260, 280))

        rf, rb, rl, rr = vmin(front), vmin(back), vmin(left), vmin(right)

        a = 0.5
        if not self.has_scan:
            self.front_dist, self.back_dist  = rf, rb
            self.left_dist,  self.right_dist = rl, rr
        else:
            self.front_dist = a * self.front_dist + (1 - a) * rf
            self.back_dist  = a * self.back_dist  + (1 - a) * rb
            self.left_dist  = a * self.left_dist  + (1 - a) * rl
            self.right_dist = a * self.right_dist + (1 - a) * rr
        self.has_scan = True

    def odom_cb(self, msg: Odometry):
        self.world_x   = msg.pose.pose.position.x
        self.world_y   = msg.pose.pose.position.y
        self.world_yaw = self._q2yaw(msg.pose.pose.orientation)

        if self._init_wx is None:
            self._init_wx, self._init_wy   = self.world_x, self.world_y
            self._init_wyaw                = self.world_yaw
            print('[INFO] Odometry origin recorded')

        dx = self.world_x - self._init_wx
        dy = self.world_y - self._init_wy
        c  = math.cos(-self._init_wyaw)
        s  = math.sin(-self._init_wyaw)
        self.local_x   = c * dx - s * dy
        self.local_y   = s * dx + c * dy
        self.local_yaw = self._norm(self.world_yaw - self._init_wyaw)
        self.has_odom  = True

    def image_cb(self, msg: CompressedImage):
        self.has_image = True
        try:
            arr   = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError
        except Exception:
            self.target_visible = False
            self.target_obs     = None
            self.target_frames  = 0
            return

        h, w = frame.shape[:2]
        self.img_h, self.img_w = h, w

        obs = self._detect_cube(frame)
        if obs is None:
            self.target_visible = False
            self.target_obs     = None
            self.target_frames  = 0
            self._prev_obs      = None
            return

        # Temporal stability
        stable = False
        if self._prev_obs and obs['color'] == self._prev_obs['color']:
            jump = abs(obs['cx'] - self._prev_obs['cx'])
            ar   = obs['area'] / max(self._prev_obs['area'], 1.0)
            stable = (jump <= 90.0 and 0.35 <= ar <= 2.8)

        self.target_frames = (self.target_frames + 1) if stable else 1
        self._prev_obs = obs

        # EMA filter on position / size
        alpha = 0.70
        if self._fcx is None or not stable:
            self._fcx, self._fcy = obs['cx'], obs['cy']
            self._fbbox_h = float(obs['bbox_h'])
        else:
            self._fcx     = alpha * self._fcx     + (1 - alpha) * obs['cx']
            self._fcy     = alpha * self._fcy     + (1 - alpha) * obs['cy']
            self._fbbox_h = alpha * self._fbbox_h + (1 - alpha) * obs['bbox_h']

        obs['cx']     = float(self._fcx)
        obs['cy']     = float(self._fcy)
        obs['bbox_h'] = int(round(self._fbbox_h))

        self.target_visible = obs['area'] >= self.MIN_CONTOUR_AREA
        self.target_obs     = obs if self.target_visible else None
        if not self.target_visible:
            self.target_frames = 0
            return

        self.last_seen_time = time.monotonic()
        err = obs['cx'] - w / 2.0
        if abs(err) > 2.0:
            self.last_turn_dir = -1.0 if err > 0 else 1.0

    # ── colour detection ──────────────────────────────────────────────────────

    def _red_mask(self, hsv, bgr):
        m = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   95, 45]), np.array([11,  255, 255])),
            cv2.inRange(hsv, np.array([170, 95, 45]), np.array([180, 255, 255])))
        r, g, b = bgr[:,:,2], bgr[:,:,1], bgr[:,:,0]
        rgb = np.zeros(m.shape, dtype=np.uint8)
        rgb[(r >= 70) & (r > g + 28) & (r > b + 28)] = 255
        return cv2.bitwise_and(m, rgb)

    def _blue_mask(self, hsv, bgr):
        m = cv2.inRange(hsv, np.array([100, 95, 35]), np.array([128, 255, 255]))
        r, g, b = bgr[:,:,2], bgr[:,:,1], bgr[:,:,0]
        rgb = np.zeros(m.shape, dtype=np.uint8)
        rgb[(b >= 60) & (b > r + 22) & (b > g + 12)] = 255
        return cv2.bitwise_and(m, rgb)

    def _colour_ok(self, color: str, roi_bgr, roi_hsv, mask) -> bool:
        px  = roi_bgr[mask > 0]
        hpx = roi_hsv[mask > 0]
        if px.size == 0:
            return False
        mb, mg, mr = np.mean(px, axis=0)
        _, ms, mv  = np.mean(hpx, axis=0)
        hue, sat   = hpx[:, 0], hpx[:, 1]
        if color == 'blue':
            hr = float(np.mean((hue >= 106) & (hue <= 124) & (sat >= 120)))
            return (hr >= 0.50 and ms >= 140 and mv >= 50
                    and mb >= 75 and (mb - mr) >= 45 and (mb - mg) >= 25)
        hr = float(np.mean(((hue <= 12) | (hue >= 168)) & (sat >= 120)))
        return (hr >= 0.65 and ms >= 140 and mv >= 55
                and mr >= 85 and (mr - mb) >= 48 and (mr - mg) >= 35)

    def _detect_cube(self, frame) -> Optional[dict]:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        k    = np.ones((5, 5), np.uint8)
        best = None

        for color, build in [('red', self._red_mask), ('blue', self._blue_mask)]:
            mask = build(hsv, frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < self.MIN_CONTOUR_AREA:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < self.MIN_BBOX_W or bh < self.MIN_BBOX_H:
                    continue

                cy_val = y + bh / 2.0
                if cy_val < h * self.MIN_CENTER_Y:
                    continue

                aspect = bw / float(bh)
                if not (0.55 <= aspect <= 1.80):
                    continue

                bbox_area = float(bw * bh)
                hull      = cv2.convexHull(cnt)
                solidity  = area / max(cv2.contourArea(hull), 1.0)
                if solidity < self.MIN_SOLIDITY:
                    continue

                shifted = cnt - np.array([[x, y]])
                roi_m   = np.zeros((bh, bw), np.uint8)
                cv2.drawContours(roi_m, [shifted], -1, 255, -1)
                fill   = float(np.count_nonzero(roi_m) / bbox_area)
                extent = float(area / bbox_area)
                if fill < self.MIN_FILL_RATIO or extent < self.MIN_EXTENT:
                    continue

                peri   = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                if not (4 <= len(approx) <= 10):
                    continue

                if not self._colour_ok(color, frame[y:y+bh, x:x+bw],
                                       hsv[y:y+bh, x:x+bw], roi_m):
                    continue

                score = 1.5 * area + 400 * fill + 300 * extent + 250 * solidity
                obs = {'color': color, 'cx': float(x + bw / 2.0),
                       'cy': cy_val, 'area': float(area),
                       'bbox_h': int(bh), 'score': float(score)}
                if best is None or obs['score'] > best['score']:
                    best = obs
        return best

    # ── state handlers ────────────────────────────────────────────────────────

    def _do_init_sweep(self):
        delta = self._norm(self.local_yaw - self._init_prev)
        self._init_accum += abs(delta)
        self._init_prev   = self.local_yaw

        self._init_front.append(self.front_dist)
        self._init_back.append(self.back_dist)
        self._init_left.append(self.left_dist)
        self._init_right.append(self.right_dist)

        if self._init_accum >= 2 * math.pi - self.SWEEP_DONE_TOL:
            self._stop()

            def med(lst):
                good = [v for v in lst if math.isfinite(v) and v > 0.05]
                return float(np.median(good)) if good else 1.0

            self.arena = ArenaMap(
                front=med(self._init_front), back=med(self._init_back),
                left=med(self._init_left),   right=med(self._init_right))
            print(f'[INIT] Arena dimensions recorded:\n'
                  f'         front={self.arena.front:.2f}m  '
                  f'back={self.arena.back:.2f}m  '
                  f'left={self.arena.left:.2f}m  '
                  f'right={self.arena.right:.2f}m\n'
                  f'         width={self.arena.width:.2f}m  '
                  f'depth={self.arena.depth:.2f}m')
            self._set_state('IDLE', 'Arena mapped. Press s to start search.')
            return

        self._pub(0.0, self.SWEEP_ANG)

    def _do_idle(self):
        self._stop()

    def _do_search(self):
        delta = self._norm(self.local_yaw - self._sweep_prev)
        self._sweep_accum += abs(delta)
        self._sweep_prev   = self.local_yaw

        # Log a cube if stable and not already logged nearby
        if (self.target_visible and self.target_obs is not None
                and self.target_frames >= self.CONFIRM_FRAMES):
            yaw   = self.local_yaw
            color = self.target_obs['color']
            too_close = any(
                abs(self._norm(yaw - a)) < self.SEARCH_LOG_WINDOW
                for a in self._sweep_logged)

            if not too_close:
                zone    = self._yaw_to_zone(yaw)
                correct = self._cube_correct(color, zone)
                entry   = CubeEntry(color=color, angle_rad=yaw,
                                    zone=zone, correct=correct)
                self._sweep_logged.append(yaw)
                if correct:
                    self.correct_cubes.append(entry)
                    print(f'[SEARCH] {color} cube found at '
                          f'{yaw:.2f} rad – {zone} (correct)')
                else:
                    self.wrong_cubes.append(entry)
                    print(f'[SEARCH] {color} cube found at '
                          f'{yaw:.2f} rad – {zone} (WRONG ZONE)')

        if self._sweep_accum >= 2 * math.pi - self.SWEEP_DONE_TOL:
            self._stop()
            print(f'[SEARCH] 360° sweep complete. '
                  f'Correct: {len(self.correct_cubes)}  '
                  f'Misplaced: {len(self.wrong_cubes)}')
            if not self.wrong_cubes:
                print('[SEARCH] All cubes are in the right zones.')
                self._set_state('IDLE', 'All cubes are in the right zones.')
            else:
                self.approach_target = self.wrong_cubes[0]
                self._set_state(
                    'TURN_TO_APPROACH',
                    f'Targeting {self.approach_target.color} cube at '
                    f'{self.approach_target.angle_rad:.2f} rad')
            return

        self._pub(0.0, self.SWEEP_ANG)

    def _do_turn_to_approach(self):
        err = self._norm(self._turn_target_yaw - self.local_yaw)
        if abs(err) <= self.YAW_TOL:
            self._stop()
            self._set_state('APPROACH', 'Aligned to cube heading – driving forward')
            return
        ang = self._clamp_abs(0.8 * err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
        self._pub(0.0, ang)

    def _do_approach(self):
        # Emergency wall check
        if self._emergency():
            self._stop()
            print('[APPROACH] Emergency stop – wall too close!')
            self._set_state('IDLE', 'Emergency wall stop during approach')
            return

        # Lost-target recovery
        if not self.target_visible or self.target_obs is None:
            if time.monotonic() - self.last_seen_time <= self.LOST_TIMEOUT:
                self._pub(0.0, self.last_turn_dir * 0.10)
                return
            print('[APPROACH] Target lost – returning to IDLE')
            self._set_state('IDLE', 'Cube lost during approach')
            return

        if self.img_w is None:
            return

        cx_img = self.img_w / 2.0
        px_err = self.target_obs['cx'] - cx_img
        err_n  = px_err / cx_img

        # Arrival check
        close_cam   = self.target_obs['bbox_h'] >= self.APPROACH_STOP_BBOX_H
        close_lidar = self.front_dist <= self.APPROACH_STOP_DIST
        if close_cam or close_lidar:
            self._stop()
            print(f'[APPROACH] Reached {self.approach_target.color} cube! '
                  f'bbox_h={self.target_obs["bbox_h"]}px  '
                  f'front={self.front_dist:.2f}m')
            if self.wrong_cubes:
                self.wrong_cubes.pop(0)
            self._set_state('TURN_HOME', 'Cube reached – returning to origin')
            return

        # Wall safety while moving forward
        if self._front_blocked():
            self._stop()
            print('[APPROACH] Path blocked by wall')
            self._set_state('IDLE', 'Wall blocked approach path')
            return

        # Rotate-only if large misalignment
        if abs(px_err) > self.ALIGN_ROTATE_ONLY_PX:
            ang = self._clamp_abs(-0.35 * err_n,
                                  self.ALIGN_ANG_MIN, self.ALIGN_ANG_MAX)
            self._pub(0.0, ang)
        else:
            ang = (0.0 if abs(px_err) <= self.ALIGN_PIXEL_TOL
                   else self._clamp_abs(-0.22 * err_n, 0.02, 0.10))
            self._pub(self.DRIVE_SPEED, ang)

    def _do_turn_home(self):
        err = self._norm(self._turn_target_yaw - self.local_yaw)
        if abs(err) <= self.YAW_TOL:
            self._stop()
            self._set_state('RETURN_HOME', 'Facing origin – driving back')
            return
        ang = self._clamp_abs(0.8 * err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
        self._pub(0.0, ang)

    def _do_return_home(self):
        if self._emergency():
            self._stop()
            print('[RETURN] Emergency stop near wall!')
            self._set_state('IDLE', 'Emergency wall stop during return')
            return

        dx   = -self.local_x
        dy   = -self.local_y
        dist = math.hypot(dx, dy)

        if dist < self.HOME_ARRIVE_DIST:
            self._stop()
            print(f'[RETURN] Back at origin. '
                  f'Remaining misplaced cubes: {len(self.wrong_cubes)}')
            if self.wrong_cubes:
                self.approach_target = self.wrong_cubes[0]
                self._set_state(
                    'TURN_TO_APPROACH',
                    f'Next: {self.approach_target.color} cube at '
                    f'{self.approach_target.angle_rad:.2f} rad')
            else:
                print('[RETURN] All misplaced cubes handled!')
                self._set_state('IDLE', 'All misplaced cubes handled')
            return

        target_hdg = math.atan2(dy, dx)
        hdg_err    = self._norm(target_hdg - self.local_yaw)

        if abs(hdg_err) > math.radians(20.0):
            ang = self._clamp_abs(0.8 * hdg_err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
            self._pub(0.0, ang)
        else:
            ang = self._clamp_abs(0.6 * hdg_err, 0.0, 0.10)
            self._pub(self.DRIVE_SPEED, ang)

    # ── main control loop ─────────────────────────────────────────────────────

    def control_loop(self):
        # Process key presses
        for k in self._pop_keys():
            if k == 'h':
                self._stop()
                print('[CMD] H – halting to IDLE')
                self._set_state('IDLE', 'Manual halt')
                return
            if k == 's' and self.state == 'IDLE':
                self._set_state('SEARCH', 'S – starting search sweep')

        if not self._ready():
            self._stop()
            return

        if self.state == 'WAIT_FOR_DATA':
            self._stop()
            self._set_state('INIT_SWEEP', 'All sensors ready – mapping arena')
            return

        handlers = {
            'INIT_SWEEP':       self._do_init_sweep,
            'IDLE':             self._do_idle,
            'SEARCH':           self._do_search,
            'TURN_TO_APPROACH': self._do_turn_to_approach,
            'APPROACH':         self._do_approach,
            'TURN_HOME':        self._do_turn_home,
            'RETURN_HOME':      self._do_return_home,
        }
        h = handlers.get(self.state)
        if h:
            h()
        else:
            self._stop()

    # ── status loop ───────────────────────────────────────────────────────────

    def status_loop(self):
        if not self._ready():
            print(f'[WAIT] scan={self.has_scan} odom={self.has_odom} '
                  f'image={self.has_image}')
            return

        cube = ''
        if self.target_visible and self.target_obs and self.img_w:
            e = self.target_obs['cx'] - self.img_w / 2.0
            cube = (f' | cube={self.target_obs["color"]} '
                    f'err={e:.0f}px bbox_h={self.target_obs["bbox_h"]}px')

        print(f'[STATUS] {self.state}  '
              f'pos=({self.local_x:.2f}, {self.local_y:.2f})  '
              f'yaw={math.degrees(self.local_yaw):.1f}°  '
              f'F={self.front_dist:.2f}m  '
              f'wrong={len(self.wrong_cubes)}'
              f'{cube}')

    # ── cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        for _ in range(12):
            self._stop()
            time.sleep(0.03)
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = CubeSorterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('[STOP] KeyboardInterrupt')
    finally:
        for _ in range(15):
            try:
                node._stop()
            except Exception:
                pass
            time.sleep(0.03)
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