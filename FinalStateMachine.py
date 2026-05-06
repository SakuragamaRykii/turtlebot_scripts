#!/usr/bin/env python3
"""
TurtleBot3 – Simplified Cube Zone Sorter
-----------------------------------------
- Immediately stops search on the first misplaced cube
- Center‑crop alignment (200×480) during approach
- Auto‑grab when cube slips under camera
- Direct delivery to y = -0.1 (red) / +0.1 (blue)
- Forward speed = 0.12 m/s
"""

import math
import threading
import time
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan

# Try to import GPIO for servo control
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print('[WARN] RPi.GPIO not available - servo control disabled')


class CubeSorterNode(Node):
    # ── motion parameters ────────────────────────────────────────────────────
    DRIVE_SPEED = 0.12               # m/s  forward cruise
    BACK_SPEED  = -0.12              # m/s  reverse
    SWEEP_ANG   = 0.40               # rad/s during search rotation
    ALIGN_ANG_MIN = 0.08             # rad/s minimum align angular speed
    ALIGN_ANG_MAX = 0.36             # rad/s maximum align angular speed
    TURN_ANG_MIN  = 0.10             # rad/s minimum turn angular speed
    TURN_ANG_MAX  = 0.36             # rad/s maximum turn angular speed

    # ── safety distances ─────────────────────────────────────────────────────
    WALL_STOP_DIST  = 0.20           # m  stop forward motion
    EMERGENCY_DIST  = 0.15           # m  hard stop any motion

    # ── angle tolerances ─────────────────────────────────────────────────────
    YAW_TOL = math.radians(5.0)

    # ── timing ───────────────────────────────────────────────────────────────
    CONTROL_DT = 0.05                # s  control loop period
    STATUS_DT  = 1.0                 # s  status print period

    # ── vision parameters ────────────────────────────────────────────────────
    MIN_CONTOUR_AREA = 260
    MIN_BBOX_W = 14
    MIN_BBOX_H = 14
    MIN_FILL_RATIO = 0.28
    MIN_EXTENT = 0.22
    MIN_SOLIDITY = 0.70
    MIN_CENTER_Y = 0.18              # ignore top part of image
    CONFIRM_FRAMES = 3               # stable frames before logging a cube

    # Camera intrinsic
    FOCAL_LENGTH_PX = 550.0
    CUBE_REAL_WIDTH  = 0.058         # m
    CUBE_REAL_HEIGHT = 0.058         # m
    MAX_CUBE_DISTANCE = 2.0          # m

    # Approach / grab conditions
    GRAB_DISTANCE = 0.15             # m
    APPROACH_STOP_DIST = 0.20        # m (LiDAR backup)

    # Alignment tolerances (pixels)
    ALIGN_PIXEL_TOL = 22
    ALIGN_ROTATE_ONLY_PX = 55

    # ── zone boundary ────────────────────────────────────────────────────────
    ZONE_BOUNDARY_X = 0.0            # red side: x >= 0, blue: x < 0

    # ── center crop for approach ─────────────────────────────────────────────
    CENTER_CROP_W = 200
    CENTER_CROP_H = 480
    
    # ── delivery offsets ─────────────────────────────────────────────────────
    DELIVERY_Y_RED  = -0.1           # y target for red cube
    DELIVERY_Y_BLUE =  0.1           # y target for blue cube

    # ── servo control ────────────────────────────────────────────────────────
    SERVO_PIN = 12
    SERVO_OPEN_DUTY = 5.5            # fully lifted / open
    SERVO_CLAMPED_DUTY = 9.0         # clamped down
    SERVO_FREQ = 50                  # Hz

    # ── return-home arrival threshold ────────────────────────────────────────
    HOME_ARRIVE_DIST = 0.15          # m

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

        # Initialize servo – ALWAYS OPEN (lifted) on boot
        self.servo = None
        if HAS_GPIO:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.SERVO_PIN, GPIO.OUT)
                self.servo = GPIO.PWM(self.SERVO_PIN, self.SERVO_FREQ)
                self.servo.start(self.SERVO_OPEN_DUTY)   # hold open continuously
                print(f'[SERVO] Initialized on GPIO {self.SERVO_PIN} – held OPEN')
            except Exception as e:
                print(f'[SERVO] Failed to initialize: {e}')
        else:
            print('[SERVO] No GPIO available – servo commands are simulated')

        # Sensor‑ready flags
        self.has_scan  = False
        self.has_odom  = False
        self.has_image = False

        # LiDAR (smoothed)
        self.front_dist = float('inf')
        self.back_dist  = float('inf')
        self.left_dist  = float('inf')
        self.right_dist = float('inf')

        # Odometry – local frame (Forward=+Y, Right=+X)
        self.local_x   = 0.0
        self.local_y   = 0.0
        self.local_yaw = 0.0
        self._init_wx   = None
        self._init_wy   = None
        self._init_wyaw = None

        # Camera
        self.img_w: Optional[int] = 640
        self.img_h: Optional[int] = 480
        self.target_visible = False
        self.target_obs: Optional[dict] = None
        self.target_frames = 0
        self.last_seen_time = 0.0
        self.last_turn_dir  = 1.0
        self.all_visible_cubes: List[dict] = []

        # Filtered camera positions
        self._fcx: Optional[float] = None
        self._fcy: Optional[float] = None
        self._fbbox_h: float = 0.0
        self._fbbox_w: float = 0.0
        self._prev_obs: Optional[dict] = None

        # Arena (hard‑coded for safe bounds checking)
        self.arena_left  = 1.0
        self.arena_right = 1.0
        self.arena_front = 1.0
        self.arena_back  = 1.0

        # Current target cube (simple dict during approach)
        self.cube_target: Optional[Dict[str, Any]] = None
        self.cube_held_color: Optional[str] = None

        # Delivery point
        self.delivery_x = 0.0
        self.delivery_y = 0.0

        # State machine
        self.state = 'WAIT_FOR_DATA'
        self._state_enter = time.monotonic()
        self._approach_had_visible = False   # flag for auto‑grab on lost cube

        # Console thread (keys: s = start search, h = halt)
        self._keys: List[str] = []
        self._key_lock = threading.Lock()
        threading.Thread(target=self._console_loop, daemon=True).start()

        print('[BOOT] Simplified Cube Sorter ready.')
        print('[BOOT] Servo held OPEN (lifted). Press s to start search.')

    # ── servo helpers ────────────────────────────────────────────────────────
    def _grab_cube(self):
        """Clamp the servo (lower grabber)."""
        print('[GRAB] CLAMPING cube...')
        if self.servo:
            self.servo.ChangeDutyCycle(self.SERVO_CLAMPED_DUTY)
            time.sleep(0.5)
        else:
            print('[GRAB] SIMULATED')

    def _release_cube(self):
        """Open the servo (lift grabber)."""
        print('[RELEASE] OPENING grabber...')
        if self.servo:
            self.servo.ChangeDutyCycle(self.SERVO_OPEN_DUTY)
            time.sleep(0.5)
        else:
            print('[RELEASE] SIMULATED')

    # ── distance & position estimation ───────────────────────────────────────
    def _estimate_distance(self, bbox_w: float, bbox_h: float) -> float:
        pixel_size = bbox_w if bbox_w > bbox_h else bbox_h
        if pixel_size < 1:
            return float('inf')
        return (self.FOCAL_LENGTH_PX * (self.CUBE_REAL_WIDTH if bbox_w > bbox_h else self.CUBE_REAL_HEIGHT)) / pixel_size

    def _cube_position_from_camera(self, bbox_cx: float,
                                   distance: float) -> (float, float):
        """Convert image coordinates to arena frame relative to robot."""
        cx_offset = bbox_cx - self.img_w / 2.0
        angle_offset = math.atan2(cx_offset, self.FOCAL_LENGTH_PX)
        cube_x = distance * math.sin(angle_offset)
        cube_y = distance * math.cos(angle_offset)
        return (self.local_x + cube_x * math.cos(self.local_yaw) - cube_y * math.sin(self.local_yaw),
                self.local_y + cube_x * math.sin(self.local_yaw) + cube_y * math.cos(self.local_yaw))

    # ── angle helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _norm(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def _q2yaw(q) -> float:
        return math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y**2+q.z**2))

    def _clamp_abs(self, v, mn, mx):
        if abs(v) < 1e-9:
            return 0.0
        return math.copysign(max(mn, min(mx, abs(v))), v)

    def _pub(self, lin=0.0, ang=0.0):
        m = Twist()
        m.linear.x = float(lin)
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

        # Reset approach visibility flag when entering APPROACH
        if new == 'APPROACH':
            self._approach_had_visible = False

    # ── sensor callbacks ─────────────────────────────────────────────────────
    def scan_cb(self, msg: LaserScan):
        r = msg.ranges
        n = len(r)
        def vmin(indices):
            vs = [r[i] for i in indices if 0 <= i < n and math.isfinite(r[i]) and r[i] > 0.05]
            return min(vs) if vs else float('inf')
        front  = list(range(0,15)) + list(range(n-15, n))
        back   = list(range(n//2-15, n//2+15))
        left   = list(range(80,100))
        right  = list(range(260,280))
        rf, rb, rl, rr = vmin(front), vmin(back), vmin(left), vmin(right)
        a = 0.5
        if not self.has_scan:
            self.front_dist, self.back_dist = rf, rb
            self.left_dist, self.right_dist = rl, rr
        else:
            self.front_dist = a*self.front_dist + (1-a)*rf
            self.back_dist  = a*self.back_dist  + (1-a)*rb
            self.left_dist  = a*self.left_dist  + (1-a)*rl
            self.right_dist = a*self.right_dist + (1-a)*rr
        self.has_scan = True

    def odom_cb(self, msg: Odometry):
        self.local_x = msg.pose.pose.position.x - (self._init_wx or 0)
        self.local_y = msg.pose.pose.position.y - (self._init_wy or 0)
        self.local_yaw = self._norm(self._q2yaw(msg.pose.pose.orientation) - (self._init_wyaw or 0))
        if self._init_wx is None:
            self._init_wx = msg.pose.pose.position.x
            self._init_wy = msg.pose.pose.position.y
            self._init_wyaw = self._q2yaw(msg.pose.pose.orientation)
            self.local_x, self.local_y = 0.0, 0.0
            print('[INFO] Odometry origin recorded')
        self.has_odom = True

    def image_cb(self, msg: CompressedImage):
        self.has_image = True
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None: raise ValueError
        except Exception:
            self.target_visible = False
            self.target_obs = None
            self.all_visible_cubes.clear()
            return

        # Use center crop only during alignment and approach states
        use_crop = self.state in ('TURN_TO_APPROACH', 'APPROACH')
        if use_crop:
            roi_x = (self.img_w - self.CENTER_CROP_W) // 2
            crop = frame[:, roi_x:roi_x + self.CENTER_CROP_W, :]
            detections = self._detect_all_cubes(crop)
            # Adjust cx back to full‑image coordinates
            for d in detections:
                d['cx'] += roi_x
        else:
            detections = self._detect_all_cubes(frame)

        self.all_visible_cubes = detections

        # Select the best cube to track (closest, correct colour if approaching)
        best_obs = None
        if not detections:
            self.target_visible = False
            self.target_obs = None
            self.target_frames = 0
            self._prev_obs = None
            return

        if self.cube_target and use_crop:
            # While approaching, lock onto the colour we are after
            matching = [d for d in detections if d['color'] == self.cube_target['color']]
            if matching:
                best_obs = min(matching, key=lambda d: d['distance'])
            else:
                best_obs = min(detections, key=lambda d: d['distance'])
        else:
            best_obs = min(detections, key=lambda d: d['distance'])

        # Temporal filtering
        stable = False
        if self._prev_obs and best_obs['color'] == self._prev_obs.get('color'):
            jump = abs(best_obs['cx'] - self._prev_obs['cx'])
            ar = best_obs['area'] / max(self._prev_obs['area'], 1.0)
            stable = (jump <= 90.0 and 0.35 <= ar <= 2.8)
        self.target_frames = (self.target_frames + 1) if stable else 1
        self._prev_obs = best_obs

        alpha = 0.70
        if self._fcx is None or not stable:
            self._fcx = best_obs['cx']
            self._fcy = best_obs['cy']
            self._fbbox_h = float(best_obs['bbox_h'])
            self._fbbox_w = float(best_obs['bbox_w'])
        else:
            self._fcx = alpha * self._fcx + (1-alpha) * best_obs['cx']
            self._fcy = alpha * self._fcy + (1-alpha) * best_obs['cy']
            self._fbbox_h = alpha * self._fbbox_h + (1-alpha) * best_obs['bbox_h']
            self._fbbox_w = alpha * self._fbbox_w + (1-alpha) * best_obs['bbox_w']

        best_obs['cx'] = float(self._fcx)
        best_obs['cy'] = float(self._fcy)
        best_obs['bbox_h'] = int(round(self._fbbox_h))
        best_obs['bbox_w'] = int(round(self._fbbox_w))
        best_obs['distance'] = self._estimate_distance(self._fbbox_w, self._fbbox_h)

        world_x, world_y = self._cube_position_from_camera(self._fcx, best_obs['distance'])
        best_obs['world_x'] = world_x
        best_obs['world_y'] = world_y

        self.target_visible = (best_obs['area'] >= self.MIN_CONTOUR_AREA and
                               best_obs['distance'] <= self.MAX_CUBE_DISTANCE)
        self.target_obs = best_obs if self.target_visible else None

        if not self.target_visible:
            self.target_frames = 0
            return

        self.last_seen_time = time.monotonic()
        err = best_obs['cx'] - self.img_w/2.0
        if abs(err) > 2.0:
            self.last_turn_dir = -1.0 if err > 0 else 1.0

    # ── colour detection (unchanged) ─────────────────────────────────────────
    def _red_mask(self, hsv, bgr):
        m = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,95,45]), np.array([11,255,255])),
            cv2.inRange(hsv, np.array([170,95,45]), np.array([180,255,255])))
        r, g, b = bgr[:,:,2], bgr[:,:,1], bgr[:,:,0]
        rgb = np.zeros(m.shape, dtype=np.uint8)
        rgb[(r>=70) & (r>g+28) & (r>b+28)] = 255
        return cv2.bitwise_and(m, rgb)

    def _blue_mask(self, hsv, bgr):
        m = cv2.inRange(hsv, np.array([100,95,35]), np.array([128,255,255]))
        r, g, b = bgr[:,:,2], bgr[:,:,1], bgr[:,:,0]
        rgb = np.zeros(m.shape, dtype=np.uint8)
        rgb[(b>=60) & (b>r+22) & (b>g+12)] = 255
        return cv2.bitwise_and(m, rgb)

    def _colour_ok(self, color, roi_bgr, roi_hsv, mask):
        px = roi_bgr[mask>0]
        hpx = roi_hsv[mask>0]
        if px.size == 0: return False
        mb, mg, mr = np.mean(px, axis=0)
        _, ms, mv = np.mean(hpx, axis=0)
        hue, sat = hpx[:,0], hpx[:,1]
        if color == 'blue':
            hr = float(np.mean((hue>=106) & (hue<=124) & (sat>=120)))
            return (hr>=0.50 and ms>=140 and mv>=50 and mb>=75 and (mb-mr)>=45 and (mb-mg)>=25)
        hr = float(np.mean(((hue<=12)|(hue>=168)) & (sat>=120)))
        return (hr>=0.65 and ms>=140 and mv>=55 and mr>=85 and (mr-mb)>=48 and (mr-mg)>=35)

    def _detect_all_cubes(self, frame) -> List[dict]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        k = np.ones((5,5), np.uint8)
        cubes = []
        for color, build in [('red', self._red_mask), ('blue', self._blue_mask)]:
            mask = build(hsv, frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < self.MIN_CONTOUR_AREA: continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < self.MIN_BBOX_W or bh < self.MIN_BBOX_H: continue
                if y + bh/2 < h * self.MIN_CENTER_Y: continue
                if not (0.55 <= bw/float(bh) <= 1.80): continue
                hull = cv2.convexHull(cnt)
                solidity = area / max(cv2.contourArea(hull), 1.0)
                if solidity < self.MIN_SOLIDITY: continue
                shifted = cnt - np.array([[x, y]])
                roi_m = np.zeros((bh, bw), np.uint8)
                cv2.drawContours(roi_m, [shifted], -1, 255, -1)
                bbox_area = float(bw * bh)
                fill = float(np.count_nonzero(roi_m))/bbox_area
                extent = float(area)/bbox_area
                if fill < self.MIN_FILL_RATIO or extent < self.MIN_EXTENT: continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03*peri, True)
                if not (4 <= len(approx) <= 10): continue
                if not self._colour_ok(color, frame[y:y+bh, x:x+bw], hsv[y:y+bh, x:x+bw], roi_m): continue
                distance = self._estimate_distance(float(bw), float(bh))
                if distance > self.MAX_CUBE_DISTANCE: continue
                cx = float(x + bw/2.0)
                cy = float(y + bh/2.0)
                world_x, world_y = self._cube_position_from_camera(cx, distance)
                # arena boundary check
                if not (-self.arena_left-0.15 < world_x < self.arena_right+0.15 and
                        -self.arena_back-0.15 < world_y < self.arena_front+0.15):
                    continue
                cubes.append({
                    'color': color, 'cx': cx, 'cy': cy,
                    'area': float(area), 'bbox_h': int(bh), 'bbox_w': int(bw),
                    'distance': distance, 'world_x': world_x, 'world_y': world_y
                })
        return cubes

    # ── state handlers ───────────────────────────────────────────────────────
    def _do_idle(self):
        self._stop()

    def _do_search(self):
        # Rotate in place
        self._pub(0.0, self.SWEEP_ANG)

        # If we already have visible cubes, check for any misplaced one
        for obs in self.all_visible_cubes:
            color = obs['color']
            world_x = obs['world_x']
            zone = 'red_zone' if world_x >= self.ZONE_BOUNDARY_X else 'blue_zone'
            correct = (color == 'red' and zone == 'red_zone') or (color == 'blue' and zone == 'blue_zone')
            if not correct:
                # Found a misplaced cube
                self._stop()
                self.cube_target = {
                    'color': color,
                    'angle_rad': self.local_yaw   # robot's heading when cube was seen
                }
                print(f'[SEARCH] Locked {color} cube (yaw={math.degrees(self.local_yaw):.1f}°)')
                self._set_state('TURN_TO_APPROACH', f'Turning to {color} cube')
                return

    def _do_turn_to_approach(self):
        if not self.cube_target:
            self._set_state('IDLE', 'No target'); return
        err = self._norm(self.cube_target['angle_rad'] - self.local_yaw)
        if abs(err) <= self.YAW_TOL:
            self._stop()
            self._set_state('APPROACH', 'Aligned')
            return
        # If cube already visible in center crop, jump directly to approach
        if self.target_visible and self.target_obs and self.target_obs['color'] == self.cube_target['color']:
            self._stop()
            self._set_state('APPROACH', 'Cube visible')
            return
        ang = self._clamp_abs(0.8*err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
        self._pub(0.0, ang)

    def _do_approach(self):
        if self._emergency():
            self._stop(); self._set_state('IDLE', 'Emergency'); return

        # Stop condition: LiDAR / distance too close
        if self.front_dist <= self.WALL_STOP_DIST or \
           (self.target_obs and self.target_obs['distance'] <= self.GRAB_DISTANCE):
            self._stop()
            self._set_state('GRAB_CUBE', 'Close enough'); return

        # Lost cube → assume it's under the camera
        if not self.target_visible:
            if self._approach_had_visible and (time.monotonic() - self._state_enter) > 1.0:
                print('[APPROACH] Cube lost from FOV – assumed underneath')
                self._stop()
                self._set_state('GRAB_CUBE', 'Lost cube – grab')
                return
            else:
                # Not yet confident we lost it; rotate slowly to reacquire
                self._pub(0.0, self.last_turn_dir * 0.15)
                return
        else:
            self._approach_had_visible = True

        # Visual servoing
        cx_img = self.img_w / 2.0
        px_err = self.target_obs['cx'] - cx_img
        err_n = px_err / cx_img

        if abs(px_err) > self.ALIGN_ROTATE_ONLY_PX:
            ang = self._clamp_abs(-0.35 * err_n, self.ALIGN_ANG_MIN, self.ALIGN_ANG_MAX)
            self._pub(0.0, ang)
        else:
            ang = 0.0 if abs(px_err) <= self.ALIGN_PIXEL_TOL else self._clamp_abs(-0.22*err_n, 0.04, 0.20)
            self._pub(self.DRIVE_SPEED, ang)

    def _do_grab_cube(self):
        self._grab_cube()
        self.cube_held_color = self.cube_target['color'] if self.cube_target else None
        self._set_state('PLAN_DELIVERY', 'Planning delivery')

    def _do_plan_delivery(self):
        # Direct delivery: go to y = -0.1 (red) or +0.1 (blue), keeping x on correct side
        if self.cube_held_color == 'red':
            self.delivery_x = max(self.local_x, 0.05)
            self.delivery_y = self.DELIVERY_Y_RED
        else:
            self.delivery_x = min(self.local_x, -0.05)
            self.delivery_y = self.DELIVERY_Y_BLUE
        print(f'[PLAN] Delivery target: ({self.delivery_x:.2f}, {self.delivery_y:.2f})')
        self._set_state('TURN_TO_DELIVER')

    def _do_turn_to_deliver(self):
        dx = self.delivery_x - self.local_x
        dy = self.delivery_y - self.local_y
        target_yaw = math.atan2(dx, dy)
        err = self._norm(target_yaw - self.local_yaw)
        if abs(err) <= self.YAW_TOL:
            self._stop()
            self._set_state('DELIVER', 'Facing delivery point'); return
        ang = self._clamp_abs(0.8*err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
        self._pub(0.0, ang)

    def _do_deliver(self):
        if self._emergency():
            self._stop(); self._set_state('IDLE', 'Emergency'); return
        dist = math.hypot(self.delivery_x - self.local_x, self.delivery_y - self.local_y)
        if dist < 0.1:
            self._stop()
            self._set_state('RELEASE_CUBE'); return
        if self.front_dist <= self.WALL_STOP_DIST:
            self._stop()
            self._set_state('RELEASE_CUBE', 'Wall forced'); return
        dx = self.delivery_x - self.local_x
        dy = self.delivery_y - self.local_y
        target_hdg = math.atan2(dx, dy)
        hdg_err = self._norm(target_hdg - self.local_yaw)
        if abs(hdg_err) > math.radians(15):
            ang = self._clamp_abs(0.7*hdg_err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
            self._pub(self.DRIVE_SPEED*0.3, ang)
        else:
            ang = self._clamp_abs(0.4*hdg_err, 0.0, 0.10)
            self._pub(self.DRIVE_SPEED, ang)

    def _do_release_cube(self):
        self._release_cube()
        self.cube_held_color = None
        self.cube_target = None
        self._set_state('TURN_HOME', 'Returning to origin')

    def _do_turn_home(self):
        dx = -self.local_x
        dy = -self.local_y
        target_yaw = math.atan2(dx, dy)
        err = self._norm(target_yaw - self.local_yaw)
        if abs(err) <= self.YAW_TOL:
            self._stop()
            self._set_state('RETURN_HOME', 'Facing origin'); return
        ang = self._clamp_abs(0.8*err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
        self._pub(0.0, ang)

    def _do_return_home(self):
        if self._emergency():
            self._stop(); self._set_state('IDLE', 'Emergency'); return
        dx = -self.local_x
        dy = -self.local_y
        dist = math.hypot(dx, dy)
        if dist < self.HOME_ARRIVE_DIST:
            self._stop()
            print('[RETURN] Home. Ready for next cube.')
            self._set_state('IDLE'); return
        target_hdg = math.atan2(dx, dy)
        hdg_err = self._norm(target_hdg - self.local_yaw)
        if abs(hdg_err) > math.radians(20):
            ang = self._clamp_abs(0.8*hdg_err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
            self._pub(0.0, ang)
        else:
            ang = self._clamp_abs(0.6*hdg_err, 0.0, 0.10)
            self._pub(self.DRIVE_SPEED, ang)

    # ── emergency & safety ───────────────────────────────────────────────────
    def _emergency(self) -> bool:
        return (self.front_dist <= self.EMERGENCY_DIST or self.back_dist <= self.EMERGENCY_DIST)

    # ── console input ────────────────────────────────────────────────────────
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
            k = self._keys
            self._keys = []
            return k

    # ── main control loop ────────────────────────────────────────────────────
    def control_loop(self):
        for k in self._pop_keys():
            if k == 'h':
                self._stop(); self._release_cube()
                self._set_state('IDLE', 'Manual halt'); return
            if k == 's' and self.state == 'IDLE':
                self.cube_target = None
                self.cube_held_color = None
                self._set_state('SEARCH', 'Searching for misplaced cube')

        if not self._ready():
            self._stop(); return

        # On first data, set hard‑coded arena and go to IDLE
        if self.state == 'WAIT_FOR_DATA':
            self._stop()
            self.arena_left = self.arena_right = self.arena_front = self.arena_back = 1.0
            print('[INIT] Hard‑coded arena: 2x2 m')
            self._set_state('IDLE')
            return

        handlers = {
            'IDLE':             self._do_idle,
            'SEARCH':           self._do_search,
            'TURN_TO_APPROACH': self._do_turn_to_approach,
            'APPROACH':         self._do_approach,
            'GRAB_CUBE':        self._do_grab_cube,
            'PLAN_DELIVERY':    self._do_plan_delivery,
            'TURN_TO_DELIVER':  self._do_turn_to_deliver,
            'DELIVER':          self._do_deliver,
            'RELEASE_CUBE':     self._do_release_cube,
            'TURN_HOME':        self._do_turn_home,
            'RETURN_HOME':      self._do_return_home,
        }
        h = handlers.get(self.state)
        if h:
            h()
        else:
            self._stop()

    # ── status print ─────────────────────────────────────────────────────────
    def status_loop(self):
        if not self._ready():
            print(f'[WAIT] scan={self.has_scan} odom={self.has_odom} image={self.has_image}')
            return
        dist_str = ''
        if self.target_visible and self.target_obs:
            dist_str = f' dist={self.target_obs["distance"]:.2f}m'
        held = f' held={self.cube_held_color}' if self.cube_held_color else ''
        print(f'[STATUS] {self.state} pos=({self.local_x:.2f},{self.local_y:.2f}) '
              f'yaw={math.degrees(self.local_yaw):.0f}° '
              f'F={self.front_dist:.2f}m{held}{dist_str}')

    # ── cleanup ──────────────────────────────────────────────────────────────
    def destroy_node(self):
        if self.servo:
            self.servo.ChangeDutyCycle(self.SERVO_OPEN_DUTY)
            time.sleep(0.3)
            self.servo.stop()
            GPIO.cleanup()
        for _ in range(12):
            self._stop(); time.sleep(0.03)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CubeSorterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('[STOP] KeyboardInterrupt')
    finally:
        for _ in range(15):
            try: node._stop()
            except: pass
            time.sleep(0.03)
        try: node.destroy_node()
        except: pass
        try:
            if rclpy.ok(): rclpy.shutdown()
        except: pass


if __name__ == '__main__':
    main()