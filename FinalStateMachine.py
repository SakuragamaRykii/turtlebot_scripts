#!/usr/bin/env python3
"""
TurtleBot3 WafflePi Cube Sorter State Machine
Sorts colored cubes (red/blue) into their correct zones
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
import time
from enum import Enum
import threading

# Servo control imports
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("WARNING: RPi.GPIO not available. Servo control disabled.")
    GPIO_AVAILABLE = False


class State(Enum):
    """State machine states"""
    WAIT_FOR_DATA = 0
    IDLE = 1
    SEARCH = 2
    TURN_TO_APPROACH = 3
    APPROACH = 4
    GRAB_CUBE = 5
    PLAN_DELIVERY = 6
    TURN_TO_DELIVER = 7
    DELIVER = 8
    RELEASE_CUBE = 9
    TURN_HOME = 10
    RETURN_HOME = 11


class CubeColor(Enum):
    """Cube color identifiers"""
    NONE = 0
    RED = 1
    BLUE = 2


class CubeSorterStateMachine(Node):
    """Main state machine for TurtleBot3 cube sorting"""
    
    def __init__(self):
        super().__init__('cube_sorter_statemachine')
        
        # Initialize state
        self.current_state = State.WAIT_FOR_DATA
        self.previous_state = None
        
        # Robot position tracking (odometry-based)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0  # Current heading in radians
        self.odom_initialized = False
        
        # Arena boundaries
        self.arena_size = 2.0  # 2m x 2m arena
        
        # Data availability flags
        self.lidar_received = False
        self.odom_received = False
        self.camera_received = False
        
        # Cube detection state
        self.target_cube_color = CubeColor.NONE
        self.target_cube_heading = None  # Angle to cube when first detected
        self.held_cube_color = CubeColor.NONE
        self.delivery_target_x = None
        self.delivery_target_y = None
        
        # Visual servoing state
        self.last_cube_x_center = None  # Pixel x-coordinate of cube center
        self.last_cube_detected_time = None
        self.cube_visible = False
        
        # LiDAR state
        self.lidar_ranges = None
        self.min_obstacle_distance = float('inf')
        
        # User input flag
        self.start_requested = False
        
        # Motion parameters
        self.SEARCH_ANGULAR_VEL = 0.30  # rad/s for rotation during search
        self.APPROACH_LINEAR_VEL = 0.12  # m/s for forward motion
        self.TURN_ANGULAR_VEL = 0.25  # rad/s for turning
        self.APPROACH_STOP_DISTANCE = 0.15  # meters
        self.DELIVERY_STOP_DISTANCE = 0.10  # meters
        self.HOME_STOP_DISTANCE = 0.15  # meters
        self.ANGLE_TOLERANCE = 0.05  # radians (~3 degrees)
        
        # Visual servoing parameters
        self.IMAGE_WIDTH = 640
        self.IMAGE_CENTER_X = self.IMAGE_WIDTH / 2
        self.CROP_WIDTH = 200
        self.CROP_HEIGHT = 480
        self.CENTERING_TOLERANCE = 30  # pixels
        self.VISUAL_SERVO_GAIN = 0.003  # angular velocity gain
        
        # Servo control
        self.SERVO_PIN = 12
        self.SERVO_OPEN_DUTY = 7.5  # Duty cycle for open (lifted) position
        self.SERVO_CLOSED_DUTY = 2.5  # Duty cycle for closed (lowered) position
        self.servo_pwm = None
        self.setup_servo()
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # ROS2 Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # ROS2 Subscribers with appropriate QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, qos_profile)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback, 10)
        
        # State machine timer (10 Hz update rate)
        self.state_timer = self.create_timer(0.1, self.state_machine_update)
        
        # Keyboard input thread
        self.input_thread = threading.Thread(target=self.keyboard_input_loop, daemon=True)
        self.input_thread.start()
        
        self.get_logger().info("Cube Sorter State Machine initialized")
        self.get_logger().info("Waiting for sensor data...")
    
    def setup_servo(self):
        """Initialize GPIO and servo PWM"""
        if not GPIO_AVAILABLE:
            self.get_logger().warn("Servo control unavailable - RPi.GPIO not found")
            return
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.SERVO_PIN, GPIO.OUT)
            self.servo_pwm = GPIO.PWM(self.SERVO_PIN, 50)  # 50Hz for servo
            self.servo_pwm.start(self.SERVO_OPEN_DUTY)
            self.get_logger().info("Servo initialized on GPIO pin 12")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize servo: {e}")
    
    def set_servo_position(self, open_position=True):
        """Control servo position"""
        if self.servo_pwm is None:
            self.get_logger().warn("Servo PWM not initialized")
            return
        
        try:
            duty_cycle = self.SERVO_OPEN_DUTY if open_position else self.SERVO_CLOSED_DUTY
            self.servo_pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(0.5)  # Allow servo to move
            self.servo_pwm.ChangeDutyCycle(0)  # Stop sending signal to prevent jitter
        except Exception as e:
            self.get_logger().error(f"Servo control error: {e}")
    
    def keyboard_input_loop(self):
        """Background thread for keyboard input"""
        while rclpy.ok():
            try:
                user_input = input()
                if user_input.lower() == 's':
                    self.start_requested = True
                    self.get_logger().info("Start command received!")
            except:
                break
    
    # ==================== ROS2 Callbacks ====================
    
    def lidar_callback(self, msg):
        """Process LiDAR scan data"""
        self.lidar_ranges = np.array(msg.ranges)
        # Replace inf values with max range
        self.lidar_ranges[np.isinf(self.lidar_ranges)] = msg.range_max
        self.min_obstacle_distance = np.min(self.lidar_ranges)
        
        if not self.lidar_received:
            self.lidar_received = True
            self.get_logger().info("LiDAR data received")
    
    def odom_callback(self, msg):
        """Process odometry data"""
        # Extract position
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Extract orientation (quaternion to euler)
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.current_theta = math.atan2(siny_cosp, cosy_cosp)
        
        if not self.odom_received:
            self.odom_received = True
            self.odom_initialized = True
            self.get_logger().info("Odometry data received")
    
    def camera_callback(self, msg):
        """Process camera images for cube detection"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect cube in current frame
            cube_color, cube_x_center, is_misplaced = self.detect_cube(cv_image)
            
            if cube_color != CubeColor.NONE:
                self.cube_visible = True
                self.last_cube_x_center = cube_x_center
                self.last_cube_detected_time = time.time()
                
                # Store target if in SEARCH state and cube is misplaced
                if self.current_state == State.SEARCH and is_misplaced:
                    if self.target_cube_color == CubeColor.NONE:
                        self.target_cube_color = cube_color
                        self.target_cube_heading = self.current_theta
                        self.get_logger().info(
                            f"Locked onto {cube_color.name} cube at heading {self.target_cube_heading:.2f} rad"
                        )
            else:
                self.cube_visible = False
            
            if not self.camera_received:
                self.camera_received = True
                self.get_logger().info("Camera data received")
                
        except Exception as e:
            self.get_logger().error(f"Camera callback error: {e}")
    
    # ==================== Cube Detection ====================
    
    def detect_cube(self, image):
        """
        Detect colored cubes in image and determine if misplaced
        Returns: (CubeColor, x_center, is_misplaced)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for red and blue cubes
        # Red has two ranges (wraps around hue=0)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours
        red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (minimum size threshold)
        MIN_CONTOUR_AREA = 500
        red_contours = [c for c in red_contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        blue_contours = [c for c in blue_contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        
        # Determine which cube to focus on
        detected_color = CubeColor.NONE
        x_center = None
        is_misplaced = False
        
        # Prioritize red cubes, then blue
        if len(red_contours) > 0:
            # Get largest red contour
            largest_contour = max(red_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                x_center = int(M['m10'] / M['m00'])
                detected_color = CubeColor.RED
                # Red cube is misplaced if robot is in blue zone (x <= 0)
                is_misplaced = (self.current_x <= 0)
        
        elif len(blue_contours) > 0:
            # Get largest blue contour
            largest_contour = max(blue_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                x_center = int(M['m10'] / M['m00'])
                detected_color = CubeColor.BLUE
                # Blue cube is misplaced if robot is in red zone (x >= 0)
                is_misplaced = (self.current_x >= 0)
        
        return detected_color, x_center, is_misplaced
    
    def estimate_cube_distance(self):
        """
        Estimate distance to cube using LiDAR (front-facing readings)
        Returns distance in meters
        """
        if self.lidar_ranges is None or len(self.lidar_ranges) == 0:
            return float('inf')
        
        # Use front 30-degree cone (±15 degrees)
        num_readings = len(self.lidar_ranges)
        angle_increment = 360.0 / num_readings
        front_range = int(15.0 / angle_increment)
        
        # Front readings (wrap around index 0)
        front_indices = list(range(0, front_range)) + list(range(num_readings - front_range, num_readings))
        front_distances = self.lidar_ranges[front_indices]
        
        # Filter out invalid readings
        valid_distances = front_distances[front_distances < 3.5]  # Max valid range
        
        if len(valid_distances) > 0:
            return np.min(valid_distances)
        return float('inf')
    
    # ==================== Motion Control ====================
    
    def stop_robot(self):
        """Stop all robot motion"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
    
    def rotate_robot(self, angular_vel):
        """Rotate robot in place"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)
    
    def drive_robot(self, linear_vel, angular_vel=0.0):
        """Drive robot forward/backward with optional turning"""
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def calculate_angle_to_point(self, target_x, target_y):
        """Calculate angle from current position to target point"""
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        return math.atan2(dy, dx)
    
    def calculate_distance_to_point(self, target_x, target_y):
        """Calculate Euclidean distance to target point"""
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        return math.sqrt(dx**2 + dy**2)
    
    # ==================== State Machine ====================
    
    def state_machine_update(self):
        """Main state machine update loop (called at 10 Hz)"""
        
        # State transition logic
        if self.current_state == State.WAIT_FOR_DATA:
            self.handle_wait_for_data()
        
        elif self.current_state == State.IDLE:
            self.handle_idle()
        
        elif self.current_state == State.SEARCH:
            self.handle_search()
        
        elif self.current_state == State.TURN_TO_APPROACH:
            self.handle_turn_to_approach()
        
        elif self.current_state == State.APPROACH:
            self.handle_approach()
        
        elif self.current_state == State.GRAB_CUBE:
            self.handle_grab_cube()
        
        elif self.current_state == State.PLAN_DELIVERY:
            self.handle_plan_delivery()
        
        elif self.current_state == State.TURN_TO_DELIVER:
            self.handle_turn_to_deliver()
        
        elif self.current_state == State.DELIVER:
            self.handle_deliver()
        
        elif self.current_state == State.RELEASE_CUBE:
            self.handle_release_cube()
        
        elif self.current_state == State.TURN_HOME:
            self.handle_turn_home()
        
        elif self.current_state == State.RETURN_HOME:
            self.handle_return_home()
    
    def transition_to(self, new_state):
        """Transition to a new state"""
        if new_state != self.current_state:
            self.get_logger().info(f"State transition: {self.current_state.name} -> {new_state.name}")
            self.previous_state = self.current_state
            self.current_state = new_state
    
    # ==================== State Handlers ====================
    
    def handle_wait_for_data(self):
        """Wait for all sensor topics to publish data"""
        if self.lidar_received and self.odom_received and self.camera_received:
            self.get_logger().info("All sensor data received. Ready to start.")
            self.transition_to(State.IDLE)
    
    def handle_idle(self):
        """Robot stopped, servo open, waiting for start command"""
        self.stop_robot()
        self.set_servo_position(open_position=True)
        
        if self.start_requested:
            self.start_requested = False
            self.get_logger().info("Starting cube search...")
            self.transition_to(State.SEARCH)
    
    def handle_search(self):
        """Rotate in place scanning for misplaced cubes"""
        # Rotate at constant speed
        self.rotate_robot(self.SEARCH_ANGULAR_VEL)
        
        # Check if we've locked onto a target
        if self.target_cube_color != CubeColor.NONE:
            self.stop_robot()
            self.transition_to(State.TURN_TO_APPROACH)
    
    def handle_turn_to_approach(self):
        """Turn toward the target cube heading"""
        if self.target_cube_heading is None:
            self.get_logger().error("No target heading available!")
            self.transition_to(State.IDLE)
            return
        
        # If cube is visible, use visual servoing instead
        if self.cube_visible and self.last_cube_x_center is not None:
            error = self.last_cube_x_center - self.IMAGE_CENTER_X
            
            if abs(error) < self.CENTERING_TOLERANCE:
                # Cube is centered, transition to approach
                self.stop_robot()
                self.transition_to(State.APPROACH)
            else:
                # Apply visual servoing
                angular_vel = -self.VISUAL_SERVO_GAIN * error
                angular_vel = np.clip(angular_vel, -self.TURN_ANGULAR_VEL, self.TURN_ANGULAR_VEL)
                self.rotate_robot(angular_vel)
        else:
            # Turn to stored heading
            angle_error = self.normalize_angle(self.target_cube_heading - self.current_theta)
            
            if abs(angle_error) < self.ANGLE_TOLERANCE:
                self.stop_robot()
                self.transition_to(State.APPROACH)
            else:
                angular_vel = self.TURN_ANGULAR_VEL if angle_error > 0 else -self.TURN_ANGULAR_VEL
                self.rotate_robot(angular_vel)
    
    def handle_approach(self):
        """Drive toward cube with visual servoing"""
        # Check stop conditions
        estimated_distance = self.estimate_cube_distance()
        
        if estimated_distance <= self.APPROACH_STOP_DISTANCE:
            self.stop_robot()
            self.get_logger().info(f"Reached cube (distance: {estimated_distance:.2f}m)")
            self.transition_to(State.GRAB_CUBE)
            return
        
        # If cube disappears, assume we're at grab distance
        if not self.cube_visible:
            if self.last_cube_detected_time is not None:
                time_since_last_seen = time.time() - self.last_cube_detected_time
                if time_since_last_seen > 1.0:  # Lost for >1 second
                    self.stop_robot()
                    self.get_logger().info("Cube disappeared - auto-grab")
                    self.transition_to(State.GRAB_CUBE)
                    return
        
        # Visual servoing: keep cube centered while driving forward
        angular_vel = 0.0
        if self.cube_visible and self.last_cube_x_center is not None:
            error = self.last_cube_x_center - self.IMAGE_CENTER_X
            angular_vel = -self.VISUAL_SERVO_GAIN * error
            angular_vel = np.clip(angular_vel, -0.3, 0.3)
        
        self.drive_robot(self.APPROACH_LINEAR_VEL, angular_vel)
    
    def handle_grab_cube(self):
        """Lower servo to grab the cube"""
        self.stop_robot()
        
        self.get_logger().info("Grabbing cube...")
        self.set_servo_position(open_position=False)
        
        # Record which cube we're holding
        self.held_cube_color = self.target_cube_color
        
        time.sleep(1.0)  # Allow grab to complete
        
        self.transition_to(State.PLAN_DELIVERY)
    
    def handle_plan_delivery(self):
        """Calculate delivery coordinates based on cube color"""
        if self.held_cube_color == CubeColor.RED:
            # Red cube → red zone (x >= 0, y = -0.1)
            self.delivery_target_x = 0.5  # Safely into red zone
            self.delivery_target_y = -0.1
        elif self.held_cube_color == CubeColor.BLUE:
            # Blue cube → blue zone (x <= 0, y = +0.1)
            self.delivery_target_x = -0.5  # Safely into blue zone
            self.delivery_target_y = 0.1
        else:
            self.get_logger().error("No cube held during delivery planning!")
            self.transition_to(State.IDLE)
            return
        
        self.get_logger().info(
            f"Delivery target: ({self.delivery_target_x:.2f}, {self.delivery_target_y:.2f})"
        )
        self.transition_to(State.TURN_TO_DELIVER)
    
    def handle_turn_to_deliver(self):
        """Turn to face delivery point"""
        if self.delivery_target_x is None or self.delivery_target_y is None:
            self.get_logger().error("No delivery target set!")
            self.transition_to(State.IDLE)
            return
        
        target_angle = self.calculate_angle_to_point(
            self.delivery_target_x, self.delivery_target_y
        )
        angle_error = self.normalize_angle(target_angle - self.current_theta)
        
        if abs(angle_error) < self.ANGLE_TOLERANCE:
            self.stop_robot()
            self.transition_to(State.DELIVER)
        else:
            angular_vel = self.TURN_ANGULAR_VEL if angle_error > 0 else -self.TURN_ANGULAR_VEL
            self.rotate_robot(angular_vel)
    
    def handle_deliver(self):
        """Drive to delivery location"""
        distance_to_target = self.calculate_distance_to_point(
            self.delivery_target_x, self.delivery_target_y
        )
        
        # Stop if close enough or obstacle detected
        if distance_to_target <= self.DELIVERY_STOP_DISTANCE or self.min_obstacle_distance < 0.15:
            self.stop_robot()
            self.get_logger().info("Reached delivery point")
            self.transition_to(State.RELEASE_CUBE)
            return
        
        # Maintain heading while driving
        target_angle = self.calculate_angle_to_point(
            self.delivery_target_x, self.delivery_target_y
        )
        angle_error = self.normalize_angle(target_angle - self.current_theta)
        angular_vel = 2.0 * angle_error  # Proportional heading correction
        angular_vel = np.clip(angular_vel, -0.3, 0.3)
        
        self.drive_robot(self.APPROACH_LINEAR_VEL, angular_vel)
    
    def handle_release_cube(self):
        """Open servo to release the cube"""
        self.stop_robot()
        
        self.get_logger().info("Releasing cube...")
        self.set_servo_position(open_position=True)
        
        # Clear held cube and target
        self.held_cube_color = CubeColor.NONE
        self.target_cube_color = CubeColor.NONE
        self.target_cube_heading = None
        self.delivery_target_x = None
        self.delivery_target_y = None
        
        time.sleep(1.0)  # Allow release to complete
        
        self.transition_to(State.TURN_HOME)
    
    def handle_turn_home(self):
        """Turn to face origin (0, 0)"""
        target_angle = self.calculate_angle_to_point(0.0, 0.0)
        angle_error = self.normalize_angle(target_angle - self.current_theta)
        
        if abs(angle_error) < self.ANGLE_TOLERANCE:
            self.stop_robot()
            self.transition_to(State.RETURN_HOME)
        else:
            angular_vel = self.TURN_ANGULAR_VEL if angle_error > 0 else -self.TURN_ANGULAR_VEL
            self.rotate_robot(angular_vel)
    
    def handle_return_home(self):
        """Drive back to origin"""
        distance_to_home = self.calculate_distance_to_point(0.0, 0.0)
        
        if distance_to_home <= self.HOME_STOP_DISTANCE:
            self.stop_robot()
            self.get_logger().info("Returned home. Ready for next cube.")
            self.transition_to(State.IDLE)
            return
        
        # Maintain heading toward home
        target_angle = self.calculate_angle_to_point(0.0, 0.0)
        angle_error = self.normalize_angle(target_angle - self.current_theta)
        angular_vel = 2.0 * angle_error
        angular_vel = np.clip(angular_vel, -0.3, 0.3)
        
        self.drive_robot(self.APPROACH_LINEAR_VEL, angular_vel)
    
    # ==================== Cleanup ====================
    
    def cleanup(self):
        """Cleanup resources on shutdown"""
        self.stop_robot()
        if self.servo_pwm is not None:
            self.servo_pwm.stop()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        self.get_logger().info("Cleanup complete")


def main(args=None):
    rclpy.init(args=args)
    
    node = CubeSorterStateMachine()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()