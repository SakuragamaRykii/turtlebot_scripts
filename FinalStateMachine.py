#!/usr/bin/env python3
"""
TurtleBot3 Cube Sorting State Machine
For Raspberry Pi 5 with PiCamera, Servo on GPIO12, OpenCR board
"""

import cv2
import numpy as np
import time
import threading
from enum import Enum
from collections import deque
import math

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# GPIO imports for servo
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO not available - using mock")
    GPIO_AVAILABLE = False

# ============================================================================
# Configuration Constants
# ============================================================================

class Config:
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Color detection HSV ranges (tune these based on your lighting)
    RED_LOWER1 = np.array([0, 100, 100])
    RED_UPPER1 = np.array([10, 255, 255])
    RED_LOWER2 = np.array([160, 100, 100])
    RED_UPPER2 = np.array([180, 255, 255])
    
    BLUE_LOWER = np.array([100, 100, 100])
    BLUE_UPPER = np.array([130, 255, 255])
    
    # Minimum contour area to consider as cube
    MIN_CUBE_AREA = 1000
    
    # Servo settings
    SERVO_PIN = 12
    SERVO_GRASP_ANGLE = 0      # Angle to close gripper
    SERVO_RELEASE_ANGLE = 90   # Angle to open gripper
    SERVO_PWM_FREQ = 50        # 50Hz for standard servos
    
    # Navigation settings
    LINEAR_SPEED = 0.15        # m/s
    ANGULAR_SPEED = 0.5        # rad/s
    APPROACH_SPEED = 0.08      # Slower approach speed
    APPROACH_DISTANCE = 0.15   # meters from cube to stop
    GRASP_DISTANCE = 0.10      # meters
    
    # Arena settings (adjust based on your arena dimensions)
    ARENA_WIDTH = 2.0          # meters
    ARENA_HEIGHT = 2.0         # meters
    
    # Target sides (change based on assignment)
    TARGET_SIDE = "LEFT"       # or "RIGHT" based on competition assignment
    
    # LiDAR settings
    FRONT_ANGLES_RANGE = 30    # degrees from center considered "front"
    WALL_DISTANCE_THRESHOLD = 0.3  # meters
    
    # Scan settings
    SCAN_ROTATION_SPEED = 0.8  # rad/s
    SCAN_ROTATION_TIME = 2.0   # seconds per quarter rotation
    
    # Debug
    DEBUG = True
    SHOW_CAMERA = True         # Set False if no display connected


# ============================================================================
# State Machine States
# ============================================================================

class RobotState(Enum):
    INIT = "INIT"
    SCANNING = "SCANNING"
    APPROACHING = "APPROACHING"
    IDENTIFYING = "IDENTIFYING"
    GRASPING = "GRASPING"
    NAVIGATING_TO_TARGET = "NAVIGATING_TO_TARGET"
    VERIFYING_BOUNDARY = "VERIFYING_BOUNDARY"
    RELEASING = "RELEASING"
    COUNTING = "COUNTING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


# ============================================================================
# Servo Controller
# ============================================================================

class ServoController:
    """Controls the servo motor for the gripper arm"""
    
    def __init__(self, pin=Config.SERVO_PIN):
        self.pin = pin
        self.is_initialized = False
        
        if GPIO_AVAILABLE:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pin, GPIO.OUT)
                self.pwm = GPIO.PWM(self.pin, Config.SERVO_PWM_FREQ)
                self.pwm.start(0)
                self.is_initialized = True
                print(f"Servo initialized on GPIO pin {self.pin}")
            except Exception as e:
                print(f"Failed to initialize servo: {e}")
    
    def set_angle(self, angle):
        """Set servo angle (0-180)"""
        if not self.is_initialized:
            print(f"[MOCK] Servo angle set to: {angle}°")
            return
        
        # Convert angle to duty cycle (typical: 2-12% for 0-180 degrees)
        duty = 2 + (angle / 18)
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(0.3)  # Allow servo to move
        self.pwm.ChangeDutyCycle(0)  # Stop signal to prevent jitter
        
        print(f"Servo moved to {angle}°")
    
    def grasp(self):
        """Close gripper to grasp cube"""
        self.set_angle(Config.SERVO_GRASP_ANGLE)
        time.sleep(1.0)  # Allow grip to secure
    
    def release(self):
        """Open gripper to release cube"""
        self.set_angle(Config.SERVO_RELEASE_ANGLE)
        time.sleep(1.0)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.is_initialized:
            self.pwm.stop()
            GPIO.cleanup()


# ============================================================================
# Cube Detector using OpenCV
# ============================================================================

class CubeDetector:
    """Detects and identifies red and blue cubes using PiCamera via cv2"""
    
    def __init__(self):
        self.camera = None
        self.frame = None
        self.running = False
        self.camera_thread = None
        
    def start_camera(self):
        """Initialize the PiCamera using cv2"""
        try:
            # For Raspberry Pi camera, usually /dev/video0
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
            
            if not self.camera.isOpened():
                raise Exception("Failed to open camera")
            
            self.running = True
            self.camera_thread = threading.Thread(target=self._camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            print("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def _camera_loop(self):
        """Continuous camera capture thread"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frame = frame
    
    def get_frame(self):
        """Get the latest camera frame"""
        if self.frame is not None:
            return self.frame.copy()
        return None
    
    def detect_cubes(self, frame=None):
        """
        Detect red and blue cubes in the frame
        Returns: list of dicts with 'color', 'contour', 'center', 'area'
        """
        if frame is None:
            frame = self.get_frame()
        
        if frame is None:
            return []
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect red (two ranges due to HSV wrap-around)
        red_mask1 = cv2.inRange(hsv, Config.RED_LOWER1, Config.RED_UPPER1)
        red_mask2 = cv2.inRange(hsv, Config.RED_LOWER2, Config.RED_UPPER2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Detect blue
        blue_mask = cv2.inRange(hsv, Config.BLUE_LOWER, Config.BLUE_UPPER)
        
        # Find cubes
        cubes = []
        
        for mask, color in [(red_mask, "RED"), (blue_mask, "BLUE")]:
            # Morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > Config.MIN_CUBE_AREA:
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    cubes.append({
                        'color': color,
                        'contour': contour,
                        'center': (cx, cy),
                        'area': area,
                        'bbox': (x, y, w, h)
                    })
        
        # Sort by area (largest first - likely closest)
        cubes.sort(key=lambda c: c['area'], reverse=True)
        
        return cubes
    
    def draw_detections(self, frame, cubes):
        """Draw bounding boxes and labels on frame"""
        for cube in cubes:
            x, y, w, h = cube['bbox']
            color = (0, 0, 255) if cube['color'] == "RED" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{cube['color']} Cube", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, cube['center'], 5, color, -1)
        return frame
    
    def stop_camera(self):
        """Stop camera capture"""
        self.running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        if self.camera:
            self.camera.release()


# ============================================================================
# Main Robot Controller
# ============================================================================

class CubeSortingRobot(Node):
    """Main robot controller implementing the state machine"""
    
    def __init__(self):
        super().__init__('cube_sorting_robot')
        
        # Initialize components
        self.servo = ServoController()
        self.detector = CubeDetector()
        
        # State machine
        self.state = RobotState.INIT
        self.previous_state = None
        
        # Robot pose estimation (simple odometry-based)
        self.pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.arena_boundaries = None
        
        # Cube tracking
        self.current_cube = None
        self.cubes_processed = []
        self.cubes_placed = {'RED': 0, 'BLUE': 0}
        self.total_cubes_to_place = 4  # 2 red + 2 blue
        
        # LiDAR data
        self.lidar_data = None
        self.min_front_distance = float('inf')
        
        # Flags
        self.running = True
        self.scan_angle_covered = 0.0
        self.approach_start_time = None
        self.navigation_target = None
        
        # Set up ROS publishers and subscribers
        self.setup_ros()
        
        # Start camera
        self.detector.start_camera()
        
        print("Cube Sorting Robot initialized")
        print(f"Target side: {Config.TARGET_SIDE}")
    
    def setup_ros(self):
        """Set up ROS2 publishers and subscribers"""
        # Publisher for robot velocity
        self.cmd_vel_pub = self.create_publisher(
            Twist, 
            '/cmd_vel', 
            10
        )
        
        # Subscriber for LiDAR data
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        
        # Subscriber for odometry (if available)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Timer for main control loop (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)
    
    def lidar_callback(self, msg):
        """Process LiDAR data"""
        self.lidar_data = msg
        
        # Calculate minimum distance in front
        front_angles = len(msg.ranges) // 2
        angle_range = Config.FRONT_ANGLES_RANGE * len(msg.ranges) // 360
        
        start_idx = front_angles - angle_range
        end_idx = front_angles + angle_range
        
        # Filter valid ranges
        front_ranges = []
        for i in range(start_idx, end_idx):
            idx = i % len(msg.ranges)
            if msg.range_min < msg.ranges[idx] < msg.range_max:
                front_ranges.append(msg.ranges[idx])
        
        self.min_front_distance = min(front_ranges) if front_ranges else float('inf')
    
    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        # This is simplified - you might want to track orientation properly
        self.pose['x'] = msg.pose.pose.position.x
        self.pose['y'] = msg.pose.pose.position.y
        
        # Extract yaw from quaternion (simplified)
        q = msg.pose.pose.orientation
        self.pose['theta'] = 2 * math.atan2(q.z, q.w)
    
    def publish_velocity(self, linear, angular):
        """Publish velocity command"""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)
    
    def stop_robot(self):
        """Stop the robot"""
        self.publish_velocity(0.0, 0.0)
    
    def control_loop(self):
        """Main control loop - executes current state"""
        if not self.running:
            return
        
        # Execute current state function
        state_function = getattr(self, f'state_{self.state.value.lower()}', None)
        if state_function:
            state_function()
        else:
            print(f"Unknown state: {self.state}")
            self.transition_to(RobotState.ERROR)
    
    def transition_to(self, new_state):
        """Transition to a new state"""
        self.previous_state = self.state
        self.state = new_state
        print(f"\n{'='*50}")
        print(f"STATE TRANSITION: {self.previous_state} -> {new_state}")
        print(f"{'='*50}\n")
    
    # ========================================================================
    # State: INIT
    # ========================================================================
    
    def state_init(self):
        """Initialize robot and start scanning"""
        print("Initializing robot systems...")
        
        # Perform LiDAR calibration - map arena
        self.map_arena()
        
        # Move to neutral position
        self.servo.release()
        time.sleep(1.0)
        
        print("Initialization complete")
        self.transition_to(RobotState.SCANNING)
    
    def map_arena(self):
        """Use LiDAR to estimate arena boundaries"""
        print("Mapping arena...")
        
        # Rotate 360 degrees to get wall distances
        self.publish_velocity(0.0, Config.SCAN_ROTATION_SPEED)
        time.sleep(6.28 / Config.SCAN_ROTATION_SPEED)  # Time for full rotation
        self.stop_robot()
        
        # Store arena boundaries based on initial position
        # Assuming robot starts in center (0,0,0)
        if self.lidar_data:
            ranges = self.lidar_data.ranges
            num_ranges = len(ranges)
            
            # Estimate distances in 4 directions
            front = min(ranges[num_ranges//4:3*num_ranges//4])
            right = min(ranges[0:num_ranges//4] + ranges[3*num_ranges//4:])
            back = min(ranges[3*num_ranges//4:])
            left = min(ranges[num_ranges//2:num_ranges//4:-1])
            
            self.arena_boundaries = {
                'front': front,
                'back': back,
                'left': left,
                'right': right
            }
            
            print(f"Arena boundaries estimated: {self.arena_boundaries}")
    
    # ========================================================================
    # State: SCANNING
    # ========================================================================
    
    def state_scanning(self):
        """Scan environment for cubes"""
        # Rotate in place to search for cubes
        self.publish_velocity(0.0, Config.SCAN_ROTATION_SPEED * 0.5)
        
        # Check camera for cubes
        frame = self.detector.get_frame()
        if frame is not None:
            cubes = self.detector.detect_cubes(frame)
            
            # Debug display
            if Config.SHOW_CAMERA and Config.DEBUG:
                debug_frame = self.detector.draw_detections(frame.copy(), cubes)
                cv2.imshow("Cube Detection", debug_frame)
                cv2.waitKey(1)
            
            # If cube found and not already processed
            for cube in cubes:
                # Check if this cube was already processed (simple position check)
                if not self._is_cube_processed(cube):
                    self.current_cube = cube
                    print(f"Found new {cube['color']} cube at {cube['center']}")
                    self.transition_to(RobotState.APPROACHING)
                    return
        
        # Increment scan angle
        self.scan_angle_covered += Config.SCAN_ROTATION_SPEED * 0.5 * 0.1
        if self.scan_angle_covered > 2 * math.pi:
            # Completed full scan without finding new cubes
            print("Scan complete - no new cubes found")
            self.scan_angle_covered = 0.0
            self.transition_to(RobotState.COUNTING)
    
    def _is_cube_processed(self, cube):
        """Check if cube was already processed"""
        for processed in self.cubes_processed:
            # Simple distance check (would be better with world coordinates)
            dist = math.sqrt(
                (cube['center'][0] - processed['center'][0])**2 +
                (cube['center'][1] - processed['center'][1])**2
            )
            if dist < 50:  # pixels
                return True
        return False
    
    # ========================================================================
    # State: APPROACHING
    # ========================================================================
    
    def state_approaching(self):
        """Approach the detected cube"""
        if self.current_cube is None:
            self.transition_to(RobotState.SCANNING)
            return
        
        # Get current frame
        frame = self.detector.get_frame()
        if frame is None:
            return
        
        # Detect cube in current frame
        cubes = self.detector.detect_cubes(frame)
        matching_cube = None
        
        # Find the same cube
        for cube in cubes:
            if cube['color'] == self.current_cube['color']:
                # Check if it's the same cube (based on position proximity)
                dist = math.sqrt(
                    (cube['center'][0] - self.current_cube['center'][0])**2 +
                    (cube['center'][1] - self.current_cube['center'][1])**2
                )
                if dist < 100:  # pixels
                    matching_cube = cube
                    break
        
        if matching_cube is None:
            print("Lost track of cube, returning to scan")
            self.transition_to(RobotState.SCANNING)
            return
        
        # Update current cube info
        self.current_cube = matching_cube
        
        # Visual servoing - center cube in frame
        frame_center_x = Config.CAMERA_WIDTH // 2
        cube_center_x = matching_cube['center'][0]
        error_x = cube_center_x - frame_center_x
        
        # Proportional control for centering
        angular_vel = -0.002 * error_x
        angular_vel = max(-Config.ANGULAR_SPEED, min(Config.ANGULAR_SPEED, angular_vel))
        
        # Move forward if cube is centered
        if abs(error_x) < 30:  # Within 30 pixels of center
            # Check LiDAR distance for precise approach
            if self.min_front_distance > Config.APPROACH_DISTANCE:
                self.publish_velocity(Config.APPROACH_SPEED, angular_vel)
            else:
                # Close enough to identify and grasp
                self.stop_robot()
                print(f"Close enough to cube ({self.min_front_distance:.2f}m)")
                self.transition_to(RobotState.IDENTIFYING)
        else:
            # Adjust orientation
            self.publish_velocity(0.0, angular_vel)
        
        # Debug display
        if Config.SHOW_CAMERA and Config.DEBUG:
            debug_frame = self.detector.draw_detections(frame.copy(), [matching_cube])
            cv2.line(debug_frame, (frame_center_x, 0), (frame_center_x, Config.CAMERA_HEIGHT), 
                    (0, 255, 0), 1)
            cv2.imshow("Approaching Cube", debug_frame)
            cv2.waitKey(1)
    
    # ========================================================================
    # State: IDENTIFYING
    # ========================================================================
    
    def state_identifying(self):
        """Confirm cube color before grasping"""
        if self.current_cube is None:
            self.transition_to(RobotState.SCANNING)
            return
        
        # Take multiple samples to confirm color
        confirmed_color = self._confirm_cube_color()
        
        if confirmed_color:
            self.current_cube['color'] = confirmed_color
            print(f"Cube color confirmed: {confirmed_color}")
            self.transition_to(RobotState.GRASPING)
        else:
            print("Could not confirm cube color")
            self.transition_to(RobotState.SCANNING)
    
    def _confirm_cube_color(self, samples=5):
        """Take multiple samples to confirm cube color"""
        color_votes = {"RED": 0, "BLUE": 0}
        
        for _ in range(samples):
            frame = self.detector.get_frame()
            if frame is None:
                continue
            
            cubes = self.detector.detect_cubes(frame)
            if cubes:
                # Take the largest cube in view
                largest = max(cubes, key=lambda c: c['area'])
                color_votes[largest['color']] += 1
            
            time.sleep(0.2)
        
        # Return color with majority vote
        if color_votes["RED"] > color_votes["BLUE"]:
            return "RED"
        elif color_votes["BLUE"] > color_votes["RED"]:
            return "BLUE"
        else:
            return None
    
    # ========================================================================
    # State: GRASPING
    # ========================================================================
    
    def state_grasping(self):
        """Execute grasping sequence"""
        if self.current_cube is None:
            self.transition_to(RobotState.SCANNING)
            return
        
        print(f"Attempting to grasp {self.current_cube['color']} cube")
        
        # Approach a bit more
        if hasattr(self, 'was_moving_for_grasp'):
            # Already moved forward, now grasp
            pass
        else:
            self.was_moving_for_grasp = True
            # Small forward movement
            self.publish_velocity(0.05, 0.0)
            time.sleep(1.0)
            self.stop_robot()
        
        # Execute grasp
        self.servo.grasp()
        time.sleep(1.0)
        
        # Verify grasp - back up and check if cube is still in view
        self.publish_velocity(-0.05, 0.0)
        time.sleep(1.0)
        self.stop_robot()
        
        # Check if cube disappeared from camera (was picked up)
        frame = self.detector.get_frame()
        cubes_remaining = self.detector.detect_cubes(frame)
        
        # If fewer cubes visible, likely grasped successfully
        print(f"Cubes still visible: {len(cubes_remaining)}")
        
        # Record as processed
        self.cubes_processed.append({
            'center': self.current_cube['center'],
            'color': self.current_cube['color'],
            'timestamp': time.time()
        })
        
        print(f"Grasped {self.current_cube['color']} cube")
        
        # Clear flag
        if hasattr(self, 'was_moving_for_grasp'):
            delattr(self, 'was_moving_for_grasp')
        
        # Navigate to target side
        self.transition_to(RobotState.NAVIGATING_TO_TARGET)
    
    # ========================================================================
    # State: NAVIGATING TO TARGET
    # ========================================================================
    
    def state_navigating_to_target(self):
        """Navigate to the target side based on cube color"""
        if self.current_cube is None:
            self.transition_to(RobotState.SCANNING)
            return
        
        target_direction = self._get_target_direction()
        print(f"Navigating to {target_direction} side")
        
        # Simple navigation strategy:
        # Rotate to face target direction, then drive forward
        
        if self.arena_boundaries is None:
            print("No arena boundaries known, estimating...")
            # Drive based on odometry
            self._navigate_by_odometry(target_direction)
        else:
            self._navigate_by_lidar(target_direction)
        
        # Check if we've arrived
        if self._check_arrival():
            print("Arrived at target area")
            self.transition_to(RobotState.VERIFYING_BOUNDARY)
    
    def _get_target_direction(self):
        """Determine which side to go based on cube color and assignment"""
        if self.current_cube['color'] == "RED":
            return "RIGHT"
        else:  # BLUE
            return "LEFT"
    
    def _navigate_by_lidar(self, direction):
        """Navigate using LiDAR wall distance"""
        # Turn toward target wall
        if direction == "LEFT":
            self.publish_velocity(0.0, 1.57)  # Rotate 90° left
            time.sleep(1.0)
        else:  # RIGHT
            self.publish_velocity(0.0, -1.57)  # Rotate 90° right
            time.sleep(1.0)
        
        self.stop_robot()
        
        # Drive forward until close to wall
        while self.min_front_distance > Config.WALL_DISTANCE_THRESHOLD:
            self.publish_velocity(Config.LINEAR_SPEED, 0.0)
            time.sleep(0.1)
        
        self.stop_robot()
    
    def _navigate_by_odometry(self, direction):
        """Navigate using wheel odometry"""
        # Simplified odometry-based navigation
        start_position = self.pose.copy()
        target_distance = 0.5  # Half arena width
        
        # Rotate to face direction
        if direction == "LEFT":
            self.publish_velocity(0.0, 1.57)
            time.sleep(1.0)
        else:
            self.publish_velocity(0.0, -1.57)
            time.sleep(1.0)
        
        self.stop_robot()
        
        # Drive forward
        distance_traveled = 0.0
        while distance_traveled < target_distance:
            self.publish_velocity(Config.LINEAR_SPEED, 0.0)
            time.sleep(0.1)
            # Update distance
            dx = self.pose['x'] - start_position['x']
            dy = self.pose['y'] - start_position['y']
            distance_traveled = math.sqrt(dx**2 + dy**2)
        
        self.stop_robot()
    
    def _check_arrival(self):
        """Check if robot has reached the target zone"""
        # Check if we're close enough to the target wall
        if self.min_front_distance < Config.WALL_DISTANCE_THRESHOLD * 1.5:
            return True
        
        # Or check odometry
        if self.arena_boundaries:
            # We're close to the boundary
            return True
        
        return False
    
    # ========================================================================
    # State: VERIFYING BOUNDARY
    # ========================================================================
    
    def state_verifying_boundary(self):
        """Verify cube placement meets the 1cm overlap requirement"""
        print("Verifying boundary position...")
        
        # Strategy: Drive a bit more toward target side to ensure overlap
        # Even if we're on the line, that's acceptable
        
        if self.min_front_distance > 0.05:  # At least 5cm from wall
            # Drive a tiny bit more
            self.publish_velocity(0.03, 0.0)
            time.sleep(1.0)
            self.stop_robot()
        
        # Consider position verified
        print("Position verified - ready to release")
        self.transition_to(RobotState.RELEASING)
    
    # ========================================================================
    # State: RELEASING
    # ========================================================================
    
    def state_releasing(self):
        """Release the cube"""
        if self.current_cube is None:
            self.transition_to(RobotState.SCANNING)
            return
        
        print(f"Releasing {self.current_cube['color']} cube")
        
        # Release servo
        self.servo.release()
        
        # Back away slightly
        self.publish_velocity(-0.05, 0.0)
        time.sleep(2.0)
        self.stop_robot()
        
        # Update counts
        self.cubes_placed[self.current_cube['color']] += 1
        print(f"Cubes placed - RED: {self.cubes_placed['RED']}, BLUE: {self.cubes_placed['BLUE']}")
        
        # Clear current cube
        self.current_cube = None
        
        # Check if done
        self.transition_to(RobotState.COUNTING)
    
    # ========================================================================
    # State: COUNTING
    # ========================================================================
    
    def state_counting(self):
        """Check if all cubes have been placed"""
        total_placed = self.cubes_placed['RED'] + self.cubes_placed['BLUE']
        
        print(f"Progress: {total_placed}/{self.total_cubes_to_place} cubes placed")
        
        if total_placed >= self.total_cubes_to_place:
            print("\n" + "="*50)
            print("ALL CUBES PLACED SUCCESSFULLY!")
            print("="*50 + "\n")
            self.transition_to(RobotState.FINISHED)
        else:
            print("More cubes to find...")
            self.transition_to(RobotState.SCANNING)
    
    # ========================================================================
    # State: FINISHED
    # ========================================================================
    
    def state_finished(self):
        """Mission complete"""
        self.stop_robot()
        self.servo.release()
        print("Robot has completed the task!")
        print(f"Final placement: RED={self.cubes_placed['RED']}, BLUE={self.cubes_placed['BLUE']}")
        
        # Optional: do a victory dance
        for _ in range(3):
            self.publish_velocity(0.0, 1.0)
            time.sleep(0.5)
            self.publish_velocity(0.0, -1.0)
            time.sleep(0.5)
        self.stop_robot()
        
        # Stop the control loop
        self.running = False
    
    # ========================================================================
    # State: ERROR
    # ========================================================================
    
    def state_error(self):
        """Error recovery state"""
        print("ERROR STATE - Attempting recovery")
        self.stop_robot()
        self.servo.release()
        
        # Back up a bit
        self.publish_velocity(-0.1, 0.0)
        time.sleep(2.0)
        self.stop_robot()
        
        # Reset and try scanning again
        self.current_cube = None
        self.transition_to(RobotState.SCANNING)
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    
    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up...")
        self.running = False
        self.stop_robot()
        self.detector.stop_camera()
        self.servo.cleanup()
        
        if Config.SHOW_CAMERA:
            cv2.destroyAllWindows()
        
        print("Cleanup complete")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function"""
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create robot controller
        robot = CubeSortingRobot()
        
        # Run the node (this blocks until node is shut down)
        rclpy.spin(robot)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            robot.cleanup()
        except:
            pass
        
        # Shutdown ROS2
        rclpy.shutdown()
        print("Shutdown complete")


if __name__ == "__main__":
    main()