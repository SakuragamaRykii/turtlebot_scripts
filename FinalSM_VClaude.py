#!/usr/bin/env python3
"""
Cube Sorter State Machine for TurtleBot3 Waffle Pi
==================================================
States follow the blueprint exactly. The robot starts at (0,0) facing +x.
Red zone: right side (y < 0), Blue zone: left side (y > 0).
Delivery points are chosen within 2m x 2m arena.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import numpy as np
import threading
import time
import math
import sys
import termios
import tty
import select

# ---------- Hardware Control ----------
try:
    import RPi.GPIO as GPIO
except ImportError:
    print("RPi.GPIO not available – using mock servo")
    class GPIOMock:
        BCM = "BCM"
        OUT = "OUT"
        @staticmethod
        def setmode(*args): pass
        @staticmethod
        def setup(*args, **kwargs): pass
        @staticmethod
        def PWM(*args, **kwargs): pass
        @staticmethod
        def cleanup(): pass
    GPIO = GPIOMock()
    # Provide dummy PWM class
    class PWM:
        def start(self, dc): pass
        def ChangeDutyCycle(self, dc): pass
        def stop(self): pass

# ---------- Servo Configuration ----------
SERVO_PIN = 12
SERVO_OPEN_DUTY = 7.5    # 90° - grabber open (lifted)
SERVO_CLOSE_DUTY = 5.0   # 0°  - grabber closed (lowered)

# ---------- Arena and Delivery ----------
ARENA_SIZE = 2.0
# Red cube delivery: x>=0, y=-0.1  -> pick (0.3, -0.1)
# Blue cube delivery: x<=0, y=+0.1 -> pick (-0.3, 0.1)
DELIVERY_RED  = (0.3, -0.1)
DELIVERY_BLUE = (-0.3, 0.1)
HOME_POS = (0.0, 0.0)

# ---------- Motion Parameters ----------
ANGULAR_SEARCH = 0.30          # rad/s, rotate in place for search
LINEAR_APPROACH = 0.12         # m/s
LINEAR_DELIVER = 0.12          # m/s
LINEAR_RETURN = 0.12           # m/s

STOP_DISTANCE_GRAB = 0.15      # stop and grab if LiDAR distance <= this
STOP_DISTANCE_DELIVER = 0.10   # stop delivery within this
STOP_DISTANCE_HOME = 0.15      # stop home return within this
WALL_DISTANCE = 0.10           # emergency stop if forward distance < this

# ---------- Turning Gains ----------
ANGLE_P_GAIN = 1.0             # P gain for heading alignment
VIS_P_GAIN = 0.5               # visual servoing proportional gain
ANGLE_TOLERANCE = 0.05         # rad, considered aligned
VIS_TOLERANCE = 0.05           # normalized (-1..1) error dead zone
APPROACH_CUBE_LOST_TIME = 0.5  # seconds without cube → auto grab

# ---------- Camera & Vision ----------
CROP_H = 200
CROP_W = 480
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CROP_Y_START = (IMAGE_HEIGHT - CROP_H) // 2   # 140
CROP_X_START = (IMAGE_WIDTH - CROP_W) // 2    # 80
# HSV colour ranges (can be tuned)
RED_LOWER1 = np.array([0, 100, 100])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 100, 100])
RED_UPPER2 = np.array([180, 255, 255])
BLUE_LOWER = np.array([100, 150, 0])
BLUE_UPPER = np.array([140, 255, 255])

# ---------- State Machine ----------
from enum import Enum
class State(Enum):
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


class CubeSorterNode(Node):
    def __init__(self):
        super().__init__('cube_sorter')
        # Publishers / Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)

        self.bridge = CvBridge()
        self.state = State.WAIT_FOR_DATA

        # Data containers (latest values)
        self.latest_odom = None
        self.latest_scan = None
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.camera_ready = False
        self.odom_ready = False
        self.scan_ready = False

        # Robot state
        self.current_pose = (0.0, 0.0, 0.0)  # x, y, yaw
        self.initial_pose_set = False
        self.initial_x = 0.0
        self.initial_y = 0.0
        self.initial_yaw = 0.0

        # Searching
        self.search_heading = 0.0
        self.target_color = None          # 'red' or 'blue'
        self.held_color = None

        # Delivery target
        self.target_pos = (0.0, 0.0)

        # Visual servoing
        self.cube_detected = False
        self.cube_centroid_x = 0.0
        self.last_cube_time = 0.0

        # Keyboard control
        self.start_search = threading.Event()
        self.keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        # Servo setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        self.servo = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz
        self.servo.start(SERVO_OPEN_DUTY)      # start in open position
        time.sleep(0.5)

        # Timer for control loop
        self.create_timer(1.0/30.0, self.control_loop)  # 30 Hz
        self.get_logger().info("Cube Sorter node started")

    # ----- Callbacks -----
    def odom_cb(self, msg: Odometry):
        self.latest_odom = msg
        self.odom_ready = True
        # Extract pose
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        # quaternion to yaw
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self.current_pose = (px, py, yaw)

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg
        self.scan_ready = True

    def image_cb(self, msg: Image):
        try:
            with self.image_lock:
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.camera_ready = True
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    # ----- Keyboard thread -----
    def _keyboard_listener(self):
        """Non‑blocking keyboard read for 's' key."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == 's':
                        self.start_search.set()
                        self.get_logger().info("Start search command received")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # ----- Helpers -----
    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

    def move(self, linear=0.0, angular=0.0):
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_pub.publish(twist)

    def set_servo(self, open: bool):
        duty = SERVO_OPEN_DUTY if open else SERVO_CLOSE_DUTY
        self.servo.ChangeDutyCycle(duty)
        time.sleep(0.3)  # allow servo to move

    def wrap_angle(self, a):
        while a > math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def angle_diff(self, target, current):
        return self.wrap_angle(target - current)

    def get_yaw(self):
        return self.current_pose[2]

    def get_position(self):
        return self.current_pose[0], self.current_pose[1]

    def forward_distance(self):
        """Minimum distance in a narrow forward cone from LiDAR."""
        if self.latest_scan is None:
            return float('inf')
        ranges = np.array(self.latest_scan.ranges)
        angles = self.latest_scan.angle_min + \
                 np.arange(len(ranges)) * self.latest_scan.angle_increment
        # forward cone ±10°
        forward_mask = np.abs(angles) < 0.175
        if not np.any(forward_mask):
            return float('inf')
        dist = np.min(ranges[forward_mask])
        if np.isnan(dist) or np.isinf(dist):
            return float('inf')
        return dist

    def detect_cube(self, color_to_find=None):
        """
        Process latest image and return:
          - If color_to_find is None: first misplaced cube (dict) or None.
          - If color_to_find is 'red' or 'blue': centroid_x in crop coordinates,
            normalized error from center (-1..1), and bounding box info.
        Returns a dict with:
          'color': 'red'/'blue', 'centroid_x', 'centroid_y',
          'error_norm': (centroid_x - center)/half_w,
          'area': contour area, 'height': bounding box height.
        """
        with self.image_lock:
            if self.latest_image is None:
                return None
            img = self.latest_image.copy()
        # Crop region
        crop = img[CROP_Y_START:CROP_Y_START+CROP_H,
                   CROP_X_START:CROP_X_START+CROP_W]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        red_mask = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1) | \
                   cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

        half_w = CROP_W / 2.0
        center_x = half_w

        def process_mask(mask, color):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < 50:  # ignore noise
                return None
            M = cv2.moments(largest)
            if M['m00'] == 0:
                return None
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            x, y, w, h = cv2.boundingRect(largest)
            return {
                'color': color,
                'centroid_x': cx,
                'centroid_y': cy,
                'error_norm': (cx - center_x) / half_w,
                'area': area,
                'height': h
            }

        red_det = process_mask(red_mask, 'red')
        blue_det = process_mask(blue_mask, 'blue')

        if color_to_find is not None:
            if color_to_find == 'red':
                return red_det
            else:
                return blue_det

        # No specific colour – check for misplaced cube using yaw
        yaw = self.get_yaw()
        # Determine which side the robot is facing:
        # yaw > 0  -> left side (blue zone), yaw < 0 -> right side (red zone)
        def is_misplaced(detection):
            if detection is None:
                return False
            col = detection['color']
            # Red cube misplaced if seen while facing blue zone (yaw > 0)
            if col == 'red' and yaw > 0:
                return True
            # Blue cube misplaced if seen while facing red zone (yaw < 0)
            if col == 'blue' and yaw < 0:
                return True
            return False

        if red_det and is_misplaced(red_det):
            return red_det
        if blue_det and is_misplaced(blue_det):
            return blue_det
        return None

    # ----- State Machine -----
    def control_loop(self):
        # WAIT_FOR_DATA: ensure all topics are publishing
        if self.state == State.WAIT_FOR_DATA:
            if self.odom_ready and self.scan_ready and self.camera_ready:
                self.get_logger().info("All data received – entering IDLE")
                self.state = State.IDLE
                self.stop_robot()
            return

        # In all operational states, update the robot pose is already done in callback.
        # Common actions depending on state:

        if self.state == State.IDLE:
            # Servo open, stationary
            self.stop_robot()
            self.set_servo(open=True)
            if self.start_search.is_set():
                self.get_logger().info("Starting SEARCH")
                self.start_search.clear()
                self.state = State.SEARCH
            return

        elif self.state == State.SEARCH:
            # Rotate in place and scan for misplaced cube
            self.move(angular=ANGULAR_SEARCH)
            detection = self.detect_cube()  # None or first misplaced cube
            if detection is not None:
                # Found misplaced cube
                self.search_heading = self.get_yaw()      # save robot heading
                self.target_color = detection['color']
                self.get_logger().info(f"Found misplaced {self.target_color} cube, heading {self.search_heading:.2f}")
                self.stop_robot()
                self.state = State.TURN_TO_APPROACH
                self.last_cube_time = self.get_clock().now().nanoseconds / 1e9
            return

        elif self.state == State.TURN_TO_APPROACH:
            # Turn to stored heading, visual servoing if cube reappears
            current_yaw = self.get_yaw()
            target_yaw = self.search_heading
            ang_err = self.angle_diff(target_yaw, current_yaw)

            # Try to see the cube again for fine alignment
            detection = self.detect_cube(color_to_find=self.target_color)
            if detection is not None:
                # Cube is visible: use visual error to adjust
                vis_err = detection['error_norm']
            else:
                vis_err = 0.0

            # Combine heading error and visual correction
            cmd_ang = ANGLE_P_GAIN * ang_err + VIS_P_GAIN * vis_err
            cmd_ang = max(min(cmd_ang, 0.5), -0.5)   # limit angular speed
            self.move(angular=cmd_ang)

            # Transition condition: heading error small and cube centered (or we trust heading)
            if abs(ang_err) < ANGLE_TOLERANCE and (detection is None or abs(vis_err) < VIS_TOLERANCE):
                self.get_logger().info("Turn to approach complete")
                self.stop_robot()
                self.state = State.APPROACH
                self.cube_detected = detection is not None
                self.last_cube_time = self.get_clock().now().nanoseconds / 1e9
            return

        elif self.state == State.APPROACH:
            # Drive forward using visual servoing, stop when close or cube lost
            # Visual servoing
            detection = self.detect_cube(color_to_find=self.target_color)
            now = self.get_clock().now().nanoseconds / 1e9
            if detection is not None:
                vis_err = detection['error_norm']
                self.cube_detected = True
                self.last_cube_time = now
            else:
                vis_err = 0.0
                self.cube_detected = False

            # Check if cube lost too long
            if not self.cube_detected and (now - self.last_cube_time) > APPROACH_CUBE_LOST_TIME:
                self.get_logger().info("Cube lost during approach – auto-grab")
                self.stop_robot()
                self.state = State.GRAB_CUBE
                return

            # LiDAR distance
            front_dist = self.forward_distance()

            # Stop conditions
            if front_dist <= STOP_DISTANCE_GRAB or (self.cube_detected and self.get_logger().info("Distance reached")):
                self.get_logger().info(f"Approach stop: dist={front_dist:.3f}")
                self.stop_robot()
                self.state = State.GRAB_CUBE
                return

            # Drive
            ang_cmd = VIS_P_GAIN * vis_err
            self.move(linear=LINEAR_APPROACH, angular=ang_cmd)
            return

        elif self.state == State.GRAB_CUBE:
            # Lower servo to grab
            self.stop_robot()
            self.set_servo(open=False)         # close grabber
            self.held_color = self.target_color
            self.get_logger().info(f"Grabbed {self.held_color} cube")
            time.sleep(0.5)                     # allow grab
            self.state = State.PLAN_DELIVERY
            return

        elif self.state == State.PLAN_DELIVERY:
            if self.held_color == 'red':
                self.target_pos = DELIVERY_RED
            else:
                self.target_pos = DELIVERY_BLUE
            self.get_logger().info(f"Delivery target: {self.target_pos}")
            self.state = State.TURN_TO_DELIVER
            return

        elif self.state == State.TURN_TO_DELIVER:
            # Rotate to face delivery point
            current_x, current_y = self.get_position()
            dx = self.target_pos[0] - current_x
            dy = self.target_pos[1] - current_y
            target_yaw = math.atan2(dy, dx)
            ang_err = self.angle_diff(target_yaw, self.get_yaw())
            cmd_ang = ANGLE_P_GAIN * ang_err
            cmd_ang = max(min(cmd_ang, 0.5), -0.5)
            self.move(angular=cmd_ang)
            if abs(ang_err) < ANGLE_TOLERANCE:
                self.stop_robot()
                self.state = State.DELIVER
            return

        elif self.state == State.DELIVER:
            # Drive to delivery point while maintaining heading
            current_x, current_y = self.get_position()
            dx = self.target_pos[0] - current_x
            dy = self.target_pos[1] - current_y
            dist = math.hypot(dx, dy)
            if dist <= STOP_DISTANCE_DELIVER or self.forward_distance() <= WALL_DISTANCE:
                self.get_logger().info("Delivery stop")
                self.stop_robot()
                self.state = State.RELEASE_CUBE
                return
            target_yaw = math.atan2(dy, dx)
            ang_err = self.angle_diff(target_yaw, self.get_yaw())
            cmd_ang = ANGLE_P_GAIN * ang_err
            self.move(linear=LINEAR_DELIVER, angular=cmd_ang)
            return

        elif self.state == State.RELEASE_CUBE:
            self.stop_robot()
            self.set_servo(open=True)         # open grabber
            self.held_color = None
            self.target_color = None
            self.get_logger().info("Cube released")
            time.sleep(0.5)
            self.state = State.TURN_HOME
            return

        elif self.state == State.TURN_HOME:
            # Turn to face origin (0,0)
            current_x, current_y = self.get_position()
            dx = HOME_POS[0] - current_x
            dy = HOME_POS[1] - current_y
            target_yaw = math.atan2(dy, dx)
            ang_err = self.angle_diff(target_yaw, self.get_yaw())
            cmd_ang = ANGLE_P_GAIN * ang_err
            cmd_ang = max(min(cmd_ang, 0.5), -0.5)
            self.move(angular=cmd_ang)
            if abs(ang_err) < ANGLE_TOLERANCE:
                self.stop_robot()
                self.state = State.RETURN_HOME
            return

        elif self.state == State.RETURN_HOME:
            # Drive to origin
            current_x, current_y = self.get_position()
            dx = HOME_POS[0] - current_x
            dy = HOME_POS[1] - current_y
            dist = math.hypot(dx, dy)
            if dist <= STOP_DISTANCE_HOME or self.forward_distance() <= WALL_DISTANCE:
                self.get_logger().info("Returned home")
                self.stop_robot()
                self.state = State.IDLE
                return
            target_yaw = math.atan2(dy, dx)
            ang_err = self.angle_diff(target_yaw, self.get_yaw())
            cmd_ang = ANGLE_P_GAIN * ang_err
            self.move(linear=LINEAR_RETURN, angular=cmd_ang)
            return

    def destroy_node(self):
        self.stop_robot()
        self.servo.stop()
        GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CubeSorterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()