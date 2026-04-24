#!/usr/bin/env python3
"""
TurtleBot3 Cube Sorting System
- Maps arena dimensions using LiDAR
- Detects cubes in wrong zones and navigates to them
- State machine based control system
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


@dataclass
class CubeInfo:
    """Information about a detected cube"""
    color: str
    angle: float  # Angle in radians relative to robot's initial orientation
    distance: float  # Estimated distance in meters
    cx: float  # Center x in image
    cy: float  # Center y in image
    area: float
    bbox_w: int
    bbox_h: int


@dataclass
class ObsCube:
    """Observation data from camera"""
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
    hole_diam: float
    fill_ratio: float
    extent: float
    solidity: float
    score: float
    color_conf: float


class CubeSorter(Node):
    def __init__(self):
        super().__init__('cube_sorter')

        # Configuration parameters
        self.ROTATION_SPEED = 0.3  # rad/s for search rotation
        self.FORWARD_SPEED = 0.08  # m/s for approach
        self.ALIGN_ROTATION_SPEED = 0.4  # rad/s for alignment
        self.APPROACH_STOP_DIST = 0.15  # Stop distance from cube in meters
        self.WALL_SAFE_DIST = 0.3  # Safe distance from walls
        self.CENTER_ARRIVAL_TOL = 0.1  # Tolerance for reaching center
        self.ANGLE_ARRIVAL_TOL = 0.05  # rad tolerance for angle alignment
        self.PIXEL_ALIGN_TOL = 30  # Pixel tolerance for cube alignment

        # Camera parameters
        self.MIN_CONTOUR_AREA = 500
        self.MIN_BBOX_W = 20
        self.MIN_BBOX_H = 20
        self.MAX_ASPECT_RATIO = 2.0
        self.MIN_ASPECT_RATIO = 0.5
        self.MIN_FILL_RATIO = 0.3
        self.MIN_EXTENT = 0.25
        self.MIN_SOLIDITY = 0.7
        self.MIN_CENTER_Y_RATIO = 0.15

        # ROS2 setup
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

        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.status_timer = self.create_timer(1.0, self.status_loop)

        # State variables
        self.state = 'INIT'
        self.prev_state = None
        self.state_start_time = time.monotonic()

        # Sensor data
        self.has_scan = False
        self.has_odom = False
        self.has_image = False
        self.front_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')
        self.back_dist = float('inf')

        # Odometry - Absolute position
        self.world_x = 0.0
        self.world_y = 0.0
        self.world_yaw = 0.0
        
        # Arena dimensions (initialized in INIT state)
        self.arena_length = 0.0  # X dimension
        self.arena_width = 0.0   # Y dimension
        self.arena_center = np.array([0.0, 0.0])
        
        # Local coordinate system (relative to initial position)
        self.init_x = None
        self.init_y = None
        self.init_yaw = None
        self.local_x = 0.0
        self.local_y = 0.0
        self.local_yaw = 0.0

        # Cube detection
        self.correct_cubes: List[CubeInfo] = []
        self.wrong_cubes: List[CubeInfo] = []
        self.current_search_angle = 0.0
        self.search_start_yaw = 0.0
        self.search_rotated = 0.0
        self.target_cube: Optional[CubeInfo] = None
        
        # Current detection
        self.current_cubes: List[ObsCube] = []
        self.image_width = None
        self.image_height = None

        # Console input thread
        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        self.get_logger().info('Cube Sorter initialized')
        print('[SYSTEM] Cube Sorter started - Press S to start search, H to halt')

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, q) -> float:
        """Convert quaternion to yaw angle"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_cmd(self, linear: float = 0.0, angular: float = 0.0):
        """Publish velocity command"""
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        """Stop the robot"""
        self.publish_cmd(0.0, 0.0)

    def set_state(self, new_state: str):
        """Change state with logging"""
        if self.state != new_state:
            self.prev_state = self.state
            self.state = new_state
            self.state_start_time = time.monotonic()
            print(f'[STATE] {new_state}')

    def console_loop(self):
        """Handle console input commands"""
        while True:
            try:
                cmd = input().strip().lower()
            except (EOFError, Exception):
                return
            
            if cmd == 's' and self.state == 'IDLE':
                self.set_state('SEARCH')
                self.correct_cubes.clear()
                self.wrong_cubes.clear()
                print('[CMD] Starting cube search')
            elif cmd == 'h':
                self.set_state('IDLE')
                self.stop_robot()
                print('[CMD] Halt command received - Returning to IDLE')
            elif cmd in ('q', 'quit', 'exit'):
                print('[CMD] Shutdown requested')
                self.stop_robot()
                rclpy.shutdown()

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR data"""
        ranges = np.array(msg.ranges)
        
        # Filter out invalid readings
        valid_mask = np.isfinite(ranges) & (ranges > 0.05)
        safe_ranges = np.where(valid_mask, ranges, float('inf'))
        
        # Get directional distances
        self.front_dist = float(np.min(safe_ranges[350:360].tolist() + safe_ranges[0:10].tolist()))
        self.left_dist = float(np.min(safe_ranges[75:105]))
        self.right_dist = float(np.min(safe_ranges[255:285]))
        self.back_dist = float(np.min(safe_ranges[165:195]))
        
        self.has_scan = True

    def odom_callback(self, msg: Odometry):
        """Process odometry data"""
        self.world_x = msg.pose.pose.position.x
        self.world_y = msg.pose.pose.position.y
        self.world_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        
        # Initialize local coordinate system on first odom message
        if self.init_x is None:
            self.init_x = self.world_x
            self.init_y = self.world_y
            self.init_yaw = self.world_yaw
            print(f'[INIT] Local coordinate system initialized at ({self.init_x:.2f}, {self.init_y:.2f})')
        
        # Calculate local coordinates
        dx = self.world_x - self.init_x
        dy = self.world_y - self.init_y
        cos_init = math.cos(-self.init_yaw)
        sin_init = math.sin(-self.init_yaw)
        
        self.local_x = cos_init * dx - sin_init * dy
        self.local_y = sin_init * dx + cos_init * dy
        self.local_yaw = self.normalize_angle(self.world_yaw - self.init_yaw)
        
        self.has_odom = True

    def image_callback(self, msg: CompressedImage):
        """Process camera image"""
        self.has_image = True
        
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
        except Exception:
            return
        
        self.image_height, self.image_width = frame.shape[:2]
        self.current_cubes = self.detect_cubes(frame)

    def build_color_mask(self, hsv: np.ndarray, bgr: np.ndarray, color: str) -> np.ndarray:
        """Build mask for specific color"""
        if color == 'red':
            lower1 = np.array([0, 100, 50])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 100, 50])
            upper2 = np.array([180, 255, 255])
            hsv_mask = cv2.bitwise_or(
                cv2.inRange(hsv, lower1, upper1),
                cv2.inRange(hsv, lower2, upper2)
            )
            # Additional RGB verification for red
            b, g, r = cv2.split(bgr)
            rgb_mask = (r >= 70) & (r > g + 25) & (r > b + 25)
            return cv2.bitwise_and(hsv_mask, rgb_mask.astype(np.uint8) * 255)
        
        elif color == 'blue':
            lower = np.array([100, 100, 40])
            upper = np.array([130, 255, 255])
            hsv_mask = cv2.inRange(hsv, lower, upper)
            # Additional RGB verification for blue
            b, g, r = cv2.split(bgr)
            rgb_mask = (b >= 60) & (b > r + 20) & (b > g + 10)
            return cv2.bitwise_and(hsv_mask, rgb_mask.astype(np.uint8) * 255)
        
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def verify_color(self, color: str, roi_bgr: np.ndarray, roi_hsv: np.ndarray, mask: np.ndarray) -> Tuple[bool, float]:
        """Verify color of detected region"""
        pixels = roi_bgr[mask > 0]
        hsv_pixels = roi_hsv[mask > 0]
        
        if pixels.size == 0 or hsv_pixels.size == 0:
            return False, 0.0
        
        mean_b, mean_g, mean_r = np.mean(pixels, axis=0)
        mean_h, mean_s, mean_v = np.mean(hsv_pixels, axis=0)
        hue = hsv_pixels[:, 0]
        sat = hsv_pixels[:, 1]
        
        if color == 'blue':
            hue_ratio = np.mean((hue >= 100) & (hue <= 130) & (sat >= 100))
            dom_rb = mean_b - mean_r
            dom_gb = mean_b - mean_g
            conf = 0.3 * mean_s + 0.35 * dom_rb + 0.2 * dom_gb + 50 * hue_ratio
            ok = (hue_ratio >= 0.5 and mean_s >= 130 and mean_v >= 50 and 
                  mean_b >= 70 and dom_rb >= 40 and dom_gb >= 20)
            return ok, conf
        
        else:  # red
            hue_ratio = np.mean(((hue <= 12) | (hue >= 168)) & (sat >= 100))
            dom_br = mean_r - mean_b
            dom_gr = mean_r - mean_g
            conf = 0.3 * mean_s + 0.35 * dom_br + 0.2 * dom_gr + 50 * hue_ratio
            ok = (hue_ratio >= 0.6 and mean_s >= 130 and mean_v >= 55 and 
                  mean_r >= 80 and dom_br >= 45 and dom_gr >= 30)
            return ok, conf

    def detect_cubes(self, frame: np.ndarray) -> List[ObsCube]:
        """Detect cubes in camera frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        detected = []
        
        for color in ['red', 'blue']:
            mask = self.build_color_mask(hsv, frame, color)
            
            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.MIN_CONTOUR_AREA:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                if w < self.MIN_BBOX_W or h < self.MIN_BBOX_H:
                    continue
                
                cy = y + h / 2.0
                if self.image_height and cy < self.image_height * self.MIN_CENTER_Y_RATIO:
                    continue
                
                aspect = w / float(h)
                if aspect < self.MIN_ASPECT_RATIO or aspect > self.MAX_ASPECT_RATIO:
                    continue
                
                # Shape analysis
                hull = cv2.convexHull(contour)
                hull_area = max(cv2.contourArea(hull), 1.0)
                solidity = area / hull_area
                if solidity < self.MIN_SOLIDITY:
                    continue
                
                bbox_area = float(w * h)
                fill_ratio = area / bbox_area
                if fill_ratio < self.MIN_FILL_RATIO:
                    continue
                
                extent = area / bbox_area
                if extent < self.MIN_EXTENT:
                    continue
                
                # Color verification
                roi_gray = gray[y:y+h, x:x+w]
                roi_hsv = hsv[y:y+h, x:x+w]
                roi_bgr = frame[y:y+h, x:x+w]
                
                shifted = contour - np.array([[x, y]])
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(roi_mask, [shifted], -1, 255, thickness=-1)
                
                color_ok, color_conf = self.verify_color(color, roi_bgr, roi_hsv, roi_mask)
                if not color_ok:
                    continue
                
                # Score calculation
                score = (1.5 * area + 300 * fill_ratio + 250 * extent + 
                        200 * solidity + 5 * color_conf)
                
                obs = ObsCube(
                    color=color,
                    cx=float(x + w / 2.0),
                    cy=float(cy),
                    area=float(area),
                    bbox_x=int(x),
                    bbox_y=int(y),
                    bbox_w=int(w),
                    bbox_h=int(h),
                    holes=0,
                    hole_pitch=0.0,
                    hole_diam=0.0,
                    fill_ratio=float(fill_ratio),
                    extent=float(extent),
                    solidity=float(solidity),
                    score=float(score),
                    color_conf=float(color_conf)
                )
                detected.append(obs)
        
        return detected

    def get_best_cube(self) -> Optional[ObsCube]:
        """Get the highest scoring cube from current detections"""
        if not self.current_cubes:
            return None
        return max(self.current_cubes, key=lambda c: c.score)

    def is_in_red_zone(self, angle: float) -> bool:
        """Check if angle points to red zone (right side)"""
        normalized = self.normalize_angle(angle)
        return -math.pi/2 <= normalized <= math.pi/2

    def is_in_blue_zone(self, angle: float) -> bool:
        """Check if angle points to blue zone (left side)"""
        return not self.is_in_red_zone(angle)

    def add_cube_observation(self, cube: ObsCube):
        """Add cube to appropriate list based on zone correctness"""
        # Estimate distance based on cube size in image
        if self.image_height:
            distance_est = 2.0 * (1.0 - cube.bbox_h / self.image_height)
        else:
            distance_est = 1.0
        
        # Calculate absolute angle of cube relative to initial orientation
        absolute_angle = self.normalize_angle(self.local_yaw)
        
        cube_info = CubeInfo(
            color=cube.color,
            angle=absolute_angle,
            distance=distance_est,
            cx=cube.cx,
            cy=cube.cy,
            area=cube.area,
            bbox_w=cube.bbox_w,
            bbox_h=cube.bbox_h
        )
        
        # Check if cube is in correct zone
        in_red_zone = self.is_in_red_zone(absolute_angle)
        
        if (cube.color == 'red' and in_red_zone) or (cube.color == 'blue' and not in_red_zone):
            self.correct_cubes.append(cube_info)
            print(f'[FOUND] Correct {cube.color} cube at {math.degrees(absolute_angle):.1f}°')
        else:
            self.wrong_cubes.append(cube_info)
            print(f'[FOUND] Wrong {cube.color} cube at {math.degrees(absolute_angle):.1f}° - Needs relocation')

    def handle_init(self):
        """Initialize arena dimensions using LiDAR"""
        if not self.has_scan or not self.has_odom:
            print('[INIT] Waiting for sensor data...')
            return
        
        # Measure arena dimensions
        total_dist = self.front_dist + self.back_dist
        total_width = self.left_dist + self.right_dist
        
        self.arena_length = total_dist
        self.arena_width = total_width
        self.arena_center = np.array([self.local_x, self.local_y])
        
        print(f'[INIT] Arena dimensions: Length={self.arena_length:.2f}m, Width={self.arena_width:.2f}m')
        print(f'[INIT] Robot positioned at center (0, 0)')
        print(f'[INIT] Arena boundaries: X=±{self.arena_length/2:.2f}m, Y=±{self.arena_width/2:.2f}m')
        
        self.set_state('IDLE')

    def handle_idle(self):
        """Stay stationary waiting for commands"""
        self.stop_robot()

    def handle_search(self):
        """Perform 360° rotation to find cubes"""
        if not self.has_odom:
            return
        
        # Start the search rotation
        if self.prev_state != 'SEARCH':
            self.search_start_yaw = self.local_yaw
            self.search_rotated = 0.0
            self.correct_cubes.clear()
            self.wrong_cubes.clear()
            print('[SEARCH] Starting full rotation scan...')
        
        # Rotate the robot
        self.publish_cmd(0.0, self.ROTATION_SPEED)
        
        # Track rotation progress
        delta_yaw = self.normalize_angle(self.local_yaw - self.search_start_yaw)
        self.search_rotated += abs(self.normalize_angle(delta_yaw - (self.search_rotated - self.search_rotated)))
        
        # Check for cubes during rotation
        best_cube = self.get_best_cube()
        if best_cube and best_cube.area > 800:  # Only record if cube is clearly visible
            # Check if we haven't already recorded this cube
            current_angle = self.normalize_angle(self.local_yaw)
            already_recorded = False
            
            for cube in self.correct_cubes + self.wrong_cubes:
                angle_diff = abs(self.normalize_angle(cube.angle - current_angle))
                if angle_diff < 0.1:  # Within ~6 degrees
                    already_recorded = True
                    break
            
            if not already_recorded:
                self.add_cube_observation(best_cube)
        
        # Check if full rotation is complete
        if abs(delta_yaw) < 0.05 and self.prev_state != 'SEARCH':
            # We've completed nearly a full rotation back to start
            if self.search_rotated > math.pi * 1.8:  # At least 324 degrees
                self.stop_robot()
                
                if not self.wrong_cubes:
                    print('[SEARCH] All cubes are in the correct zones!')
                    self.set_state('IDLE')
                else:
                    print(f'[SEARCH] Found {len(self.wrong_cubes)} cubes in wrong zones')
                    self.target_cube = self.wrong_cubes[0]
                    print(f'[SEARCH] Approaching {self.target_cube.color} cube at {math.degrees(self.target_cube.angle):.1f}°')
                    self.set_state('APPROACH')
        
        self.prev_state = self.state

    def check_wall_proximity(self) -> bool:
        """Check if robot is too close to any wall"""
        if not self.has_scan:
            return False
        
        # Check all directions
        if self.front_dist < self.WALL_SAFE_DIST:
            print(f'[WARNING] Front wall too close: {self.front_dist:.2f}m')
            return True
        if self.back_dist < self.WALL_SAFE_DIST:
            print(f'[WARNING] Back wall too close: {self.back_dist:.2f}m')
            return True
        if self.left_dist < self.WALL_SAFE_DIST:
            print(f'[WARNING] Left wall too close: {self.left_dist:.2f}m')
            return True
        if self.right_dist < self.WALL_SAFE_DIST:
            print(f'[WARNING] Right wall too close: {self.right_dist:.2f}m')
            return True
        
        return False

    def handle_approach(self):
        """Navigate towards the target cube"""
        if not self.target_cube or not self.has_odom:
            self.set_state('IDLE')
            return
        
        # Check wall proximity
        if self.check_wall_proximity():
            print('[APPROACH] Too close to wall, adjusting...')
            self.stop_robot()
            # Back up slightly
            self.publish_cmd(-0.05, 0.0)
            time.sleep(0.5)
            self.stop_robot()
        
        # Check if we can see the cube
        best_cube = self.get_best_cube()
        
        if best_cube and best_cube.area > 500:
            # Cube is visible - align and approach
            if self.image_width is None:
                return
            
            # Calculate pixel error from center
            center_x = self.image_width / 2.0
            pixel_error = best_cube.cx - center_x
            
            # Check if aligned
            if abs(pixel_error) < self.PIXEL_ALIGN_TOL:
                # Aligned - move forward
                if best_cube.area > 50000:  # Very close
                    print(f'[APPROACH] Reached {self.target_cube.color} cube!')
                    self.stop_robot()
                    # Remove this cube from wrong_cubes
                    if self.wrong_cubes:
                        self.wrong_cubes.pop(0)
                    
                    # Check if more cubes need processing
                    if self.wrong_cubes:
                        self.target_cube = self.wrong_cubes[0]
                        print(f'[APPROACH] Next target: {self.target_cube.color} cube')
                    else:
                        print('[APPROACH] All wrong cubes processed - Returning to center')
                        self.set_state('RETURN_TO_CENTER')
                else:
                    # Move forward slowly
                    self.publish_cmd(self.FORWARD_SPEED, 0.0)
            else:
                # Need to rotate to align
                angle_correction = -0.5 * (pixel_error / center_x)
                angular_speed = max(-self.ALIGN_ROTATION_SPEED, 
                                  min(self.ALIGN_ROTATION_SPEED, angle_correction))
                self.publish_cmd(0.0, angular_speed)
        else:
            # Cube not visible - rotate to last known angle
            if self.target_cube:
                angle_error = self.normalize_angle(self.target_cube.angle - self.local_yaw)
                if abs(angle_error) < self.ANGLE_ARRIVAL_TOL:
                    # At correct angle but no cube - move forward slowly
                    self.publish_cmd(self.FORWARD_SPEED * 0.5, 0.0)
                else:
                    # Rotate towards target
                    angular_speed = max(-self.ALIGN_ROTATION_SPEED,
                                      min(self.ALIGN_ROTATION_SPEED, 2.0 * angle_error))
                    self.publish_cmd(0.0, angular_speed)
            else:
                self.set_state('IDLE')

    def handle_return_to_center(self):
        """Return to the initial position (0, 0)"""
        if not self.has_odom:
            return
        
        # Calculate vector to center
        dx = -self.local_x  # Want to reach x=0
        dy = -self.local_y  # Want to reach y=0
        dist_to_center = math.sqrt(dx**2 + dy**2)
        
        if dist_to_center < self.CENTER_ARRIVAL_TOL:
            print('[RETURN] Reached center position')
            self.stop_robot()
            self.set_state('IDLE')
            return
        
        # Calculate angle to center
        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - self.local_yaw)
        
        # Check wall proximity during return
        if self.check_wall_proximity():
            print('[RETURN] Wall proximity warning - Adjusting path')
            self.stop_robot()
        
        # Align with center direction first
        if abs(angle_error) > self.ANGLE_ARRIVAL_TOL:
            angular_speed = max(-self.ALIGN_ROTATION_SPEED,
                              min(self.ALIGN_ROTATION_SPEED, 2.0 * angle_error))
            self.publish_cmd(0.0, angular_speed)
        else:
            # Move towards center
            speed = min(self.FORWARD_SPEED, dist_to_center)
            self.publish_cmd(speed, 0.0)

    def control_loop(self):
        """Main control loop"""
        if not self.has_scan or not self.has_odom:
            return
        
        # Execute current state
        if self.state == 'INIT':
            self.handle_init()
        elif self.state == 'IDLE':
            self.handle_idle()
        elif self.state == 'SEARCH':
            self.handle_search()
        elif self.state == 'APPROACH':
            self.handle_approach()
        elif self.state == 'RETURN_TO_CENTER':
            self.handle_return_to_center()

    def status_loop(self):
        """Periodic status reporting"""
        if not self.has_odom:
            return
        
        position = f'X={self.local_x:.2f} Y={self.local_y:.2f} Yaw={math.degrees(self.local_yaw):.1f}°'
        
        if self.state == 'SEARCH':
            print(f'[STATUS] {self.state} | {position} | Correct: {len(self.correct_cubes)} Wrong: {len(self.wrong_cubes)}')
        elif self.state == 'APPROACH' and self.target_cube:
            print(f'[STATUS] {self.state} | {position} | Target: {self.target_cube.color} cube')
        else:
            print(f'[STATUS] {self.state} | {position}')

    def destroy_node(self):
        """Clean shutdown"""
        self.stop_robot()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CubeSorter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('[SYSTEM] Shutdown requested')
        node.stop_robot()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()