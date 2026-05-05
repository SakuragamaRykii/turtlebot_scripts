#!/usr/bin/env python3
"""
TurtleBot3 WafflePi – Cube Zone Sorter (GRABBER + SMART DELIVERY) - FIXED
==================================================================
FIXED: Records cube angles during search and uses them for approach
       without requiring continuous visual contact during turns.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CubeEntry:
    color: str        # 'red' | 'blue'
    angle_rad: float  # local yaw when cube was spotted (ROBOT'S HEADING)
    zone: str         # 'red_zone' | 'blue_zone'
    correct: bool     # True if cube is already in its matching zone
    x_pos: float = 0.0  # X position in arena
    y_pos: float = 0.0  # Y position in arena
    distance: float = 0.0  # Estimated distance from robot when spotted
    on_center_line: bool = False  # Whether cube is on or near y=0


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
    
    def is_within_bounds(self, x: float, y: float, margin: float = 0.1) -> bool:
        """Check if a point is within arena boundaries."""
        return (-self.left + margin < x < self.right - margin and
                -self.back + margin < y < self.front - margin)


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
    YAW_TOL           = math.radians(5.0)  # Increased from 3° for more reliable turning
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
    LOST_TIMEOUT      = 2.0     # s before giving up on a lost cube (INCREASED)
    APPROACH_LOST_TIMEOUT = 3.0 # s - extra time during approach before giving up

    # ── PiCamera parameters for distance estimation ──────────────────────────
    CUBE_REAL_WIDTH = 0.058     # m - actual cube width (5.8cm)
    CUBE_REAL_HEIGHT = 0.058    # m - actual cube height (5.8cm)
    FOCAL_LENGTH_PX = 550.0     # pixels - NEEDS CALIBRATION
    
    # Distance filtering
    MAX_CUBE_DISTANCE = 2.0     # m - ignore cubes beyond this distance
    ARENA_BOUNDARY_MARGIN = 0.15 # m - margin for arena boundary checks

    # Stop approach when cube is this close
    GRAB_DISTANCE = 0.15        # m - distance to grab cube
    APPROACH_STOP_BBOX_H = 180  # px - backup stop condition
    APPROACH_STOP_DIST   = 0.20  # m - LiDAR backup stop
    
    # Pixel tolerance for "centred" alignment
    ALIGN_PIXEL_TOL      = 22   # px
    # Pixel error above which we rotate-only (no translation)
    ALIGN_ROTATE_ONLY_PX = 55   # px

    # ── zone boundary ────────────────────────────────────────────────────────
    ZONE_BOUNDARY_X = 0.0  # m - dividing line at X=0
    
    # ── center line parameters ───────────────────────────────────────────────
    CENTER_LINE_Y = 0.0     # Target Y position (the center line)
    CENTER_LINE_TOL = 0.05  # m - tolerance for reaching center line
    CENTER_LINE_DETECT_DIST = 0.20  # m - how close to center to flag obstacle
    
    # ── delivery path adjustment ─────────────────────────────────────────────
    DELIVERY_X_OFFSET = 0.30  # m - offset when avoiding obstacles
    MAX_DELIVERY_ATTEMPTS = 3  # Maximum attempts to find clear path

    # ── servo control ────────────────────────────────────────────────────────
    SERVO_PIN = 12          # GPIO pin for servo (UPDATED to your pin)
    SERVO_CLAMPED = 1800    # PWM value for clamped position
    SERVO_OPEN = 1100       # PWM value for open position
    SERVO_FREQ = 50         # Hz

    # ── search logging window ─────────────────────────────────────────────────
    SEARCH_LOG_WINDOW = math.radians(20.0)  # INCREASED - don't re-log within this arc
    SEARCH_POSITION_WINDOW = 0.20  # m - INCREASED don't re-log cubes within this distance

    # ── return-home arrival threshold ────────────────────────────────────────
    HOME_ARRIVE_DIST  = 0.15    # m - INCREASED tolerance for reaching origin

    # ── blind approach parameters ────────────────────────────────────────────
    BLIND_APPROACH_SPEED = 0.05  # m/s - slower speed when approaching blind
    BLIND_APPROACH_MAX_DIST = 1.5  # m - max distance to drive blind

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

        # Initialize servo
        self.servo = None
        if HAS_GPIO:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.SERVO_PIN, GPIO.OUT)
                self.servo = GPIO.PWM(self.SERVO_PIN, self.SERVO_FREQ)
                self.servo.start(self.SERVO_OPEN / 20000.0 * 100)  # Convert to duty cycle
                print(f'[SERVO] Initialized on GPIO {self.SERVO_PIN}')
            except Exception as e:
                print(f'[SERVO] Failed to initialize: {e}')
        else:
            print('[SERVO] No GPIO available - servo commands will be simulated')

        # Sensor-ready flags
        self.has_scan  = False
        self.has_odom  = False
        self.has_image = False

        # LiDAR (smoothed)
        self.front_dist = float('inf')
        self.back_dist  = float('inf')
        self.left_dist  = float('inf')
        self.right_dist = float('inf')

        # Odometry – local frame (Forward=+Y, Right=+X)
        self.world_x   = 0.0
        self.world_y   = 0.0
        self.world_yaw = 0.0
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
        self.last_turn_dir  = 1.0
        
        # Multiple cube tracking for center line detection
        self.all_visible_cubes: List[dict] = []

        # Filtered camera positions
        self._fcx: Optional[float] = None
        self._fcy: Optional[float] = None
        self._fbbox_h: float = 0.0
        self._fbbox_w: float = 0.0
        self._prev_obs: Optional[dict] = None

        # Arena map (set during INIT_SWEEP or hard-coded)
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
        self.all_cubes: List[CubeEntry] = []  # All detected cubes

        # Search sweep state
        self._sweep_accum = 0.0
        self._sweep_prev  = 0.0
        self._sweep_logged: List[Tuple[float, float]] = []  # (yaw, distance)
        self._search_start_yaw = 0.0  # Yaw when search started

        # Approach / delivery state
        self.approach_target: Optional[CubeEntry] = None
        self._turn_target_yaw = 0.0
        self._turn_start_time = 0.0  # When we started turning
        self._blind_approach_start_pos = (0.0, 0.0)  # Position when blind approach started
        self._blind_approach_dist_traveled = 0.0  # Distance traveled in blind approach
        self.delivery_x = 0.0  # Target X for delivery
        self.delivery_y = 0.0  # Target Y for delivery (always near 0)
        self.delivery_attempts = 0  # Counter for path planning attempts
        
        # Cube held state
        self.cube_held: Optional[CubeEntry] = None

        # State machine
        self.state = 'WAIT_FOR_DATA'
        self._state_enter = time.monotonic()

        # Console thread
        self._keys: List[str] = []
        self._key_lock = threading.Lock()
        threading.Thread(target=self._console_loop, daemon=True).start()

        print('[BOOT] CubeSorter started (GRABBER + SMART DELIVERY - FIXED)')
        print('[BOOT] Coordinate system: Forward = +Y, Right = +X')
        print('[BOOT] FIX: Records cube angles for blind approach')
        print('[CMD]  s = start search | h = halt to IDLE')

    # ── servo control ─────────────────────────────────────────────────────────
    
    def _grab_cube(self):
        """Activate servo to clamp cube"""
        print('[GRAB] Clamping cube...')
        if self.servo:
            self.servo.ChangeDutyCycle(self.SERVO_CLAMPED / 20000.0 * 100)
            time.sleep(0.5)
        else:
            print('[GRAB] SIMULATED - servo not available')
    
    def _release_cube(self):
        """Open servo to release cube"""
        print('[RELEASE] Opening grabber...')
        if self.servo:
            self.servo.ChangeDutyCycle(self.SERVO_OPEN / 20000.0 * 100)
            time.sleep(0.5)
        else:
            print('[RELEASE] SIMULATED - servo not available')

    # ── distance estimation ──────────────────────────────────────────────────
    
    def _estimate_distance(self, bbox_w: float, bbox_h: float) -> float:
        """Estimate distance to cube using known real-world dimensions."""
        if bbox_w > bbox_h:
            pixel_size = bbox_w
            real_size = self.CUBE_REAL_WIDTH
        else:
            pixel_size = bbox_h
            real_size = self.CUBE_REAL_HEIGHT
        
        if pixel_size < 1:
            return float('inf')
        
        distance = (self.FOCAL_LENGTH_PX * real_size) / pixel_size
        return distance
    
    def _cube_position_from_camera(self, bbox_cx: float, bbox_cy: float, 
                                   distance: float) -> Tuple[float, float]:
        """Calculate the cube's position in arena coordinates."""
        if not self.img_w or not self.img_h:
            return 0.0, 0.0
        
        cx_offset = bbox_cx - self.img_w / 2.0
        angle_offset = math.atan2(cx_offset, self.FOCAL_LENGTH_PX)
        
        cube_x = distance * math.sin(angle_offset)
        cube_y = distance * math.cos(angle_offset)
        
        cube_arena_x = self.local_x + cube_x * math.cos(self.local_yaw) - cube_y * math.sin(self.local_yaw)
        cube_arena_y = self.local_y + cube_x * math.sin(self.local_yaw) + cube_y * math.cos(self.local_yaw)
        
        return cube_arena_x, cube_arena_y

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
            self._search_start_yaw = self.local_yaw
            self._sweep_accum = 0.0
            self._sweep_logged.clear()
            self.correct_cubes.clear()
            self.wrong_cubes.clear()
            self.all_cubes.clear()
            print(f'[SEARCH] Starting search sweep at yaw={math.degrees(self.local_yaw):.1f}°')
        elif new == 'TURN_TO_APPROACH':
            if self.approach_target:
                self._turn_target_yaw = self.approach_target.angle_rad
                self._turn_start_time = time.monotonic()
                print(f'[TURN] Target cube was at yaw={math.degrees(self._turn_target_yaw):.1f}°')
                print(f'[TURN] Current yaw={math.degrees(self.local_yaw):.1f}°')
            else:
                print('[TURN] ERROR: No approach target set!')
        elif new == 'GRAB_CUBE':
            self._grab_cube()
        elif new == 'PLAN_DELIVERY':
            self._plan_delivery_path()
        elif new == 'RELEASE_CUBE':
            self._release_cube()
        elif new == 'TURN_HOME':
            dx = -self.local_x
            dy = -self.local_y
            self._turn_target_yaw = math.atan2(dx, dy)
            self._turn_start_time = time.monotonic()
        elif new == 'APPROACH':
            # Record start position for blind approach tracking
            self._blind_approach_start_pos = (self.local_x, self.local_y)
            self._blind_approach_dist_traveled = 0.0
            print(f'[APPROACH] Starting approach from ({self.local_x:.2f}, {self.local_y:.2f})')

    # ── delivery path planning ───────────────────────────────────────────────
    
    def _check_center_line_obstacles(self) -> List[dict]:
        obstacles = []
        for cube in self.all_cubes:
            if cube.on_center_line:
                obstacles.append({
                    'x': cube.x_pos, 'y': cube.y_pos,
                    'color': cube.color,
                    'distance': abs(cube.y_pos - self.CENTER_LINE_Y)
                })
        for obs in self.all_visible_cubes:
            if obs.get('world_y', 0) is not None:
                if abs(obs.get('world_y', 0) - self.CENTER_LINE_Y) < self.CENTER_LINE_DETECT_DIST:
                    obstacles.append({
                        'x': obs.get('world_x', 0), 'y': obs.get('world_y', 0),
                        'color': obs['color'],
                        'distance': abs(obs.get('world_y', 0) - self.CENTER_LINE_Y)
                    })
        return obstacles
    
    def _plan_delivery_path(self):
        if not self.cube_held:
            self._set_state('TURN_HOME', 'No cube held')
            return
        
        target_x = self.local_x
        target_y = self.CENTER_LINE_Y
        
        if self.cube_held.color == 'red':
            target_x = max(self.ZONE_BOUNDARY_X + 0.05, target_x)
        else:
            target_x = min(self.ZONE_BOUNDARY_X - 0.05, target_x)
        
        if self.arena:
            target_x = self._clamp(target_x, -self.arena.left + 0.15, self.arena.right - 0.15)
        
        obstacles = self._check_center_line_obstacles()
        
        if obstacles:
            print(f'[PLAN] Found {len(obstacles)} obstacle(s) on center line')
            for attempt in range(self.MAX_DELIVERY_ATTEMPTS):
                collision = False
                for obs in obstacles:
                    if abs(target_x - obs['x']) < self.CENTER_LINE_DETECT_DIST:
                        collision = True
                        print(f'[PLAN] Collision with {obs["color"]} cube at x={obs["x"]:.2f}')
                        break
                if not collision:
                    break
                if attempt == 0:
                    target_x += self.DELIVERY_X_OFFSET
                elif attempt == 1:
                    target_x -= 2 * self.DELIVERY_X_OFFSET
                else:
                    target_x += self.DELIVERY_X_OFFSET * (attempt - 1)
                if self.arena:
                    target_x = self._clamp(target_x, -self.arena.left + 0.15, self.arena.right - 0.15)
        
        self.delivery_x = target_x
        self.delivery_y = target_y
        self.delivery_attempts = 0
        
        print(f'[PLAN] Delivery target: ({self.delivery_x:.2f}, {self.delivery_y:.2f})m')
        self._set_state('TURN_TO_DELIVER', f'Turning to face delivery point')

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

    def _is_in_red_zone(self, x: float) -> bool:
        return x >= self.ZONE_BOUNDARY_X
    
    def _is_in_blue_zone(self, x: float) -> bool:
        return x < self.ZONE_BOUNDARY_X
    
    def _get_zone(self, x: float) -> str:
        return 'red_zone' if self._is_in_red_zone(x) else 'blue_zone'

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

        dx_world = self.world_x - self._init_wx
        dy_world = self.world_y - self._init_wy
        
        cos_yaw = math.cos(self._init_wyaw)
        sin_yaw = math.sin(self._init_wyaw)
        
        self.local_x = dx_world * cos_yaw + dy_world * sin_yaw
        self.local_y = -dx_world * sin_yaw + dy_world * cos_yaw
        self.local_yaw = self._norm(self.world_yaw - self._init_wyaw)
        self.has_odom  = True
        
        # Track blind approach distance
        if self.state == 'APPROACH':
            dx_traveled = self.local_x - self._blind_approach_start_pos[0]
            dy_traveled = self.local_y - self._blind_approach_start_pos[1]
            self._blind_approach_dist_traveled = math.hypot(dx_traveled, dy_traveled)

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
            self.all_visible_cubes.clear()
            return

        h, w = frame.shape[:2]
        self.img_h, self.img_w = h, w

        all_obs = self._detect_all_cubes(frame)
        self.all_visible_cubes = all_obs
        
        best_obs = None
        if not all_obs:
            self.target_visible = False
            self.target_obs = None
            self.target_frames = 0
            self._prev_obs = None
            return
        
        # If we're approaching a target, prioritize that color
        if self.approach_target and self.state in ['APPROACH', 'TURN_TO_APPROACH']:
            target_color = self.approach_target.color
            matching = [o for o in all_obs if o['color'] == target_color]
            if matching:
                best_obs = min(matching, key=lambda o: o.get('distance', float('inf')))
            else:
                best_obs = min(all_obs, key=lambda o: o.get('distance', float('inf')))
        else:
            best_obs = min(all_obs, key=lambda o: o.get('distance', float('inf')))

        if best_obs is None:
            self.target_visible = False
            self.target_obs = None
            self.target_frames = 0
            self._prev_obs = None
            return

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
            self._fcx = alpha * self._fcx + (1 - alpha) * best_obs['cx']
            self._fcy = alpha * self._fcy + (1 - alpha) * best_obs['cy']
            self._fbbox_h = alpha * self._fbbox_h + (1 - alpha) * best_obs['bbox_h']
            self._fbbox_w = alpha * self._fbbox_w + (1 - alpha) * best_obs['bbox_w']

        best_obs['cx'] = float(self._fcx)
        best_obs['cy'] = float(self._fcy)
        best_obs['bbox_h'] = int(round(self._fbbox_h))
        best_obs['bbox_w'] = int(round(self._fbbox_w))
        best_obs['distance'] = self._estimate_distance(self._fbbox_w, self._fbbox_h)
        
        world_x, world_y = self._cube_position_from_camera(
            self._fcx, self._fcy, best_obs['distance'])
        best_obs['world_x'] = world_x
        best_obs['world_y'] = world_y

        self.target_visible = (best_obs['area'] >= self.MIN_CONTOUR_AREA and 
                               best_obs['distance'] <= self.MAX_CUBE_DISTANCE)
        self.target_obs = best_obs if self.target_visible else None
        
        if not self.target_visible:
            self.target_frames = 0
            return

        self.last_seen_time = time.monotonic()
        err = best_obs['cx'] - w / 2.0
        if abs(err) > 2.0:
            self.last_turn_dir = -1.0 if err > 0 else 1.0

    # ── colour detection (unchanged) ──────────────────────────────────────────

    def _red_mask(self, hsv, bgr):
        m = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 95, 45]), np.array([11, 255, 255])),
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

    def _detect_all_cubes(self, frame) -> List[dict]:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        k    = np.ones((5, 5), np.uint8)
        all_cubes = []

        for color, build in [('red', self._red_mask), ('blue', self._blue_mask)]:
            mask = build(hsv, frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

                hull      = cv2.convexHull(cnt)
                solidity  = area / max(cv2.contourArea(hull), 1.0)
                if solidity < self.MIN_SOLIDITY:
                    continue

                shifted = cnt - np.array([[x, y]])
                roi_m   = np.zeros((bh, bw), np.uint8)
                cv2.drawContours(roi_m, [shifted], -1, 255, -1)
                bbox_area = float(bw * bh)
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

                distance = self._estimate_distance(float(bw), float(bh))
                if distance > self.MAX_CUBE_DISTANCE:
                    continue
                
                cx = float(x + bw / 2.0)
                cy = cy_val
                world_x, world_y = self._cube_position_from_camera(cx, cy, distance)
                
                if self.arena and not self.arena.is_within_bounds(world_x, world_y, self.ARENA_BOUNDARY_MARGIN):
                    continue

                obs = {
                    'color': color,
                    'cx': cx, 'cy': cy,
                    'area': float(area),
                    'bbox_h': int(bh), 'bbox_w': int(bw),
                    'distance': distance,
                    'world_x': world_x, 'world_y': world_y
                }
                all_cubes.append(obs)

        return all_cubes

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
                left=med(self._init_left), right=med(self._init_right))
            print(f'[INIT] Arena: {self.arena.front:.2f}F/{self.arena.back:.2f}B '
                  f'{self.arena.left:.2f}L/{self.arena.right:.2f}R')
            self._set_state('IDLE', 'Arena mapped. Press s to search.')
            return

        self._pub(0.0, self.SWEEP_ANG)

    def _do_idle(self):
        self._stop()

    def _do_search(self):
        delta = self._norm(self.local_yaw - self._sweep_prev)
        self._sweep_accum += abs(delta)
        self._sweep_prev   = self.local_yaw

        # Log all visible cubes with their detection angles
        if self.all_visible_cubes:
            for obs in self.all_visible_cubes:
                yaw = self.local_yaw  # CURRENT robot heading when cube is seen
                color = obs['color']
                distance = obs.get('distance', 0)
                world_x = obs.get('world_x', 0)
                world_y = obs.get('world_y', 0)
                
                # Check if already logged nearby
                too_close = any(
                    abs(self._norm(yaw - a)) < self.SEARCH_LOG_WINDOW and
                    abs(distance - d) < self.SEARCH_POSITION_WINDOW
                    for a, d in self._sweep_logged)
                
                if too_close:
                    continue
                
                self._sweep_logged.append((yaw, distance))
                
                # Determine zone based on cube position
                zone = self._get_zone(world_x)
                correct = self._cube_correct(color, zone)
                
                # Check if on center line
                on_center = abs(world_y - self.CENTER_LINE_Y) < self.CENTER_LINE_DETECT_DIST
                
                # CRITICAL: Store the yaw angle (robot's heading) when cube was spotted
                entry = CubeEntry(
                    color=color,
                    angle_rad=yaw,  # <-- THIS IS THE KEY: robot's heading when cube seen
                    zone=zone,
                    correct=correct,
                    x_pos=world_x,
                    y_pos=world_y,
                    distance=distance,
                    on_center_line=on_center
                )
                
                self.all_cubes.append(entry)
                if correct:
                    self.correct_cubes.append(entry)
                else:
                    self.wrong_cubes.append(entry)
                
                status = 'ON CENTER' if on_center else ('correct' if correct else 'WRONG ZONE')
                print(f'[SEARCH] {color} at ({world_x:.2f}, {world_y:.2f})m '
                      f'dist={distance:.2f}m yaw={math.degrees(yaw):.1f}° - {zone} ({status})')

        if self._sweep_accum >= 2 * math.pi - self.SWEEP_DONE_TOL:
            self._stop()
            print(f'[SEARCH] Complete sweep. Correct: {len(self.correct_cubes)}  '
                  f'Misplaced: {len(self.wrong_cubes)}')
            
            # Print summary of all found cubes
            for i, cube in enumerate(self.wrong_cubes):
                print(f'[SEARCH]   Wrong #{i+1}: {cube.color} at '
                      f'({cube.x_pos:.2f}, {cube.y_pos:.2f})m, '
                      f'spotted at yaw={math.degrees(cube.angle_rad):.1f}°')
            
            # Prioritize off-center cubes
            misplaced_off_center = [c for c in self.wrong_cubes if not c.on_center_line]
            misplaced_on_center = [c for c in self.wrong_cubes if c.on_center_line]
            self.wrong_cubes = misplaced_off_center + misplaced_on_center
            
            if not self.wrong_cubes:
                print('[SEARCH] All cubes in right zones!')
                self._set_state('IDLE', 'All cubes correct')
            else:
                self.approach_target = self.wrong_cubes[0]
                print(f'[SEARCH] First target: {self.approach_target.color} at '
                      f'yaw={math.degrees(self.approach_target.angle_rad):.1f}°')
                self._set_state('TURN_TO_APPROACH',
                    f'Turning to {math.degrees(self.approach_target.angle_rad):.1f}° for {self.approach_target.color} cube')
            return

        self._pub(0.0, self.SWEEP_ANG)

    def _do_turn_to_approach(self):
        """Turn to face the recorded angle of the target cube."""
        if not self.approach_target:
            print('[TURN] ERROR: No target!')
            self._set_state('IDLE', 'No target')
            return
        
        # Use the stored angle from when the cube was spotted
        target_yaw = self.approach_target.angle_rad
        err = self._norm(target_yaw - self.local_yaw)
        
        # Log progress every ~1 second
        if int(time.monotonic() * 2) % 2 == 0:
            print(f'[TURN] Turning to {math.degrees(target_yaw):.0f}° | '
                  f'current={math.degrees(self.local_yaw):.0f}° | '
                  f'error={math.degrees(err):.1f}°')
        
        if abs(err) <= self.YAW_TOL:
            self._stop()
            print(f'[TURN] Aligned! Facing {math.degrees(self.local_yaw):.1f}°')
            self._set_state('APPROACH', 'Driving toward cube (may be blind initially)')
            return
        
        # Check timeout - if turning too long, start approach anyway
        turn_duration = time.monotonic() - self._turn_start_time
        if turn_duration > 8.0:
            print(f'[TURN] Timeout after {turn_duration:.1f}s - starting approach')
            self._stop()
            self._set_state('APPROACH', 'Timeout - starting blind approach')
            return
        
        ang = self._clamp_abs(0.8 * err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
        self._pub(0.0, ang)

    def _do_approach(self):
        """Approach cube - can work blind using odometry if cube not visible."""
        if self._emergency():
            self._stop()
            self._set_state('IDLE', 'Emergency stop')
            return

        # If we can see the cube, use visual servoing
        if self.target_visible and self.target_obs is not None:
            self.last_seen_time = time.monotonic()
            
            # Check distance for grab
            distance = self.target_obs.get('distance', float('inf'))
            
            if distance <= self.GRAB_DISTANCE:
                self._stop()
                print(f'[APPROACH] Close enough! Distance: {distance:.3f}m')
                self.cube_held = self.approach_target
                if self.wrong_cubes and self.approach_target in self.wrong_cubes:
                    self.wrong_cubes.remove(self.approach_target)
                self._set_state('GRAB_CUBE', f'Grabbing {self.cube_held.color} cube')
                return
            
            # Visual servoing
            if self.img_w is None:
                return
            
            cx_img = self.img_w / 2.0
            px_err = self.target_obs['cx'] - cx_img
            err_n = px_err / cx_img

            if abs(px_err) > self.ALIGN_ROTATE_ONLY_PX:
                ang = self._clamp_abs(-0.35 * err_n, self.ALIGN_ANG_MIN, self.ALIGN_ANG_MAX)
                self._pub(0.0, ang)
            else:
                ang = (0.0 if abs(px_err) <= self.ALIGN_PIXEL_TOL
                       else self._clamp_abs(-0.22 * err_n, 0.02, 0.10))
                self._pub(self.DRIVE_SPEED, ang)
            
            return
        
        # BLIND APPROACH: Cube not visible - drive forward based on odometry
        time_since_seen = time.monotonic() - self.last_seen_time
        
        # Check if we've been blind too long
        if time_since_seen > self.APPROACH_LOST_TIMEOUT:
            print(f'[APPROACH] Lost cube for {time_since_seen:.1f}s')
            if self._blind_approach_dist_traveled > self.BLIND_APPROACH_MAX_DIST:
                print('[APPROACH] Traveled too far blind - giving up')
                self._set_state('IDLE', 'Cube lost during blind approach')
                return
            # If we had visual recently, keep going blind
            if time_since_seen > 5.0:
                self._set_state('IDLE', 'Cube lost too long')
                return
        
        # Check wall
        if self._front_blocked():
            self._stop()
            print('[APPROACH] Wall ahead during blind approach')
            self._set_state('IDLE', 'Wall blocked blind approach')
            return
        
        # Drive forward blind
        print(f'[APPROACH] Blind: traveled {self._blind_approach_dist_traveled:.2f}m, '
              f'lost for {time_since_seen:.1f}s')
        self._pub(self.BLIND_APPROACH_SPEED, 0.0)

    def _do_grab_cube(self):
        """Wait for grab to complete"""
        time.sleep(0.5)
        self._set_state('PLAN_DELIVERY', 'Planning delivery path')

    def _do_turn_to_deliver(self):
        dx = self.delivery_x - self.local_x
        dy = self.delivery_y - self.local_y
        target_yaw = math.atan2(dx, dy)
        err = self._norm(target_yaw - self.local_yaw)
        
        if abs(err) <= self.YAW_TOL:
            self._stop()
            self._set_state('DELIVER', 'Driving to delivery point')
            return
        
        ang = self._clamp_abs(0.8 * err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
        self._pub(0.0, ang)

    def _do_deliver(self):
        if self._emergency():
            self._stop()
            self._set_state('IDLE', 'Emergency during delivery')
            return

        dist_to_target = math.hypot(
            self.delivery_x - self.local_x,
            self.delivery_y - self.local_y)
        
        y_dist = abs(self.local_y - self.CENTER_LINE_Y)
        
        if y_dist <= self.CENTER_LINE_TOL:
            self._stop()
            print(f'[DELIVER] Arrived at center line! ({self.local_x:.3f}, {self.local_y:.3f})')
            self._set_state('RELEASE_CUBE', 'Releasing cube at center line')
            return
        
        if self._front_blocked():
            self._stop()
            print('[DELIVER] Wall blocking delivery')
            self._set_state('RELEASE_CUBE', 'Wall forced early release')
            return
        
        dx = self.delivery_x - self.local_x
        dy = self.delivery_y - self.local_y
        target_yaw = math.atan2(dx, dy)
        hdg_err = self._norm(target_yaw - self.local_yaw)
        
        if abs(hdg_err) > math.radians(15):
            ang = self._clamp_abs(0.7 * hdg_err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
            self._pub(self.DRIVE_SPEED * 0.3, ang)
        else:
            ang = self._clamp_abs(0.4 * hdg_err, 0.0, 0.10)
            speed = min(self.DRIVE_SPEED, dist_to_target * 0.5)
            self._pub(speed, ang)

    def _do_release_cube(self):
        time.sleep(0.5)
        self.cube_held = None
        self._set_state('TURN_HOME', 'Cube released - returning home')

    def _do_turn_home(self):
        dx = -self.local_x
        dy = -self.local_y
        target_yaw = math.atan2(dx, dy)
        err = self._norm(target_yaw - self.local_yaw)
        
        if abs(err) <= self.YAW_TOL:
            self._stop()
            self._set_state('RETURN_HOME', 'Facing origin')
            return
        
        turn_duration = time.monotonic() - self._turn_start_time
        if turn_duration > 8.0:
            print('[TURN_HOME] Timeout - starting return anyway')
            self._stop()
            self._set_state('RETURN_HOME', 'Timeout - returning')
            return
        
        ang = self._clamp_abs(0.8 * err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
        self._pub(0.0, ang)

    def _do_return_home(self):
        if self._emergency():
            self._stop()
            self._set_state('IDLE', 'Emergency during return')
            return

        dx = -self.local_x
        dy = -self.local_y
        dist = math.hypot(dx, dy)

        if dist < self.HOME_ARRIVE_DIST:
            self._stop()
            print(f'[RETURN] Home. Remaining misplaced: {len(self.wrong_cubes)}')
            if self.wrong_cubes:
                self.approach_target = self.wrong_cubes[0]
                self._set_state('TURN_TO_APPROACH',
                    f'Next: {self.approach_target.color} at yaw={math.degrees(self.approach_target.angle_rad):.1f}°')
            else:
                self._set_state('IDLE', 'All cubes placed!')
            return

        target_hdg = math.atan2(dx, dy)
        hdg_err = self._norm(target_hdg - self.local_yaw)

        if abs(hdg_err) > math.radians(20.0):
            ang = self._clamp_abs(0.8 * hdg_err, self.TURN_ANG_MIN, self.TURN_ANG_MAX)
            self._pub(0.0, ang)
        else:
            ang = self._clamp_abs(0.6 * hdg_err, 0.0, 0.10)
            self._pub(self.DRIVE_SPEED, ang)

    # ── main control loop ─────────────────────────────────────────────────────

    def control_loop(self):
        for k in self._pop_keys():
            if k == 'h':
                self._stop()
                self._release_cube()
                self._set_state('IDLE', 'Manual halt')
                return
            if k == 's' and self.state == 'IDLE':
                self._set_state('SEARCH', 'Starting search')

        if not self._ready():
            self._stop()
            return

        # HARD-CODED ARENA DIMENSIONS
        if self.state == 'WAIT_FOR_DATA':
            self._stop()
            self.arena = ArenaMap(
                front=1.0, back=1.0,
                left=1.0, right=1.0
            )
            print(f'[INIT] Arena hard-coded: {self.arena.width:.2f}x{self.arena.depth:.2f}m')
            self._set_state('IDLE', 'Arena ready. Press s to search.')
            return

        handlers = {
            'INIT_SWEEP':       self._do_init_sweep,
            'IDLE':             self._do_idle,
            'SEARCH':           self._do_search,
            'TURN_TO_APPROACH': self._do_turn_to_approach,
            'APPROACH':         self._do_approach,
            'GRAB_CUBE':        self._do_grab_cube,
            'PLAN_DELIVERY':    lambda: None,
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

    # ── status loop ───────────────────────────────────────────────────────────

    def status_loop(self):
        if not self._ready():
            print(f'[WAIT] scan={self.has_scan} odom={self.has_odom} image={self.has_image}')
            return

        dist_info = ''
        if self.target_visible and self.target_obs:
            dist_info = f' dist={self.target_obs.get("distance", 0):.2f}m'
        
        held = f' held={self.cube_held.color}' if self.cube_held else ''
        
        target_info = ''
        if self.approach_target and self.state in ['TURN_TO_APPROACH', 'APPROACH']:
            target_info = f' target_yaw={math.degrees(self.approach_target.angle_rad):.0f}°'

        print(f'[STATUS] {self.state} pos=({self.local_x:.2f},{self.local_y:.2f}) '
              f'yaw={math.degrees(self.local_yaw):.0f}° '
              f'F={self.front_dist:.2f}m wrong={len(self.wrong_cubes)}{held}{target_info}{dist_info}')

    # ── cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        if self.servo:
            self.servo.stop()
            GPIO.cleanup()
        for _ in range(12):
            self._stop()
            time.sleep(0.03)
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