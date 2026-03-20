#!/usr/bin/env python3
import math
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class ArenaMapper(Node):
    def __init__(self):
        super().__init__('arena_mapper')

        # =========================
        # Parameters
        # =========================
        self.SEARCH_SPEED = -0.25  # Negative for Clockwise rotation
        self.MIN_AREA = 500        # Minimum pixel area for both colors
        self.CENTER_FOV = 0.40     # Only log cubes that enter the middle 40% of the camera
        self.CONTROL_DT = 0.05

        # =========================
        # ROS2 Communication
        # =========================
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.control_timer = self.create_timer(self.CONTROL_DT, self.control_loop)

        # =========================
        # State & Sensor Variables
        # =========================
        self.has_odom = False
        self.has_image = False

        self.world_yaw = 0.0
        self.init_world_yaw = None
        self.local_yaw = 0.0

        self.image_width = None

        # Color tracking data
        self.vis = {'red': False, 'blue': False}
        self.cx = {'red': None, 'blue': None}
        self.area = {'red': 0.0, 'blue': 0.0}

        # Multi-cube debouncing trackers
        self.trackers = {
            'red': {'active': False, 'peak_area': 0.0, 'peak_yaw': 0.0},
            'blue': {'active': False, 'peak_area': 0.0, 'peak_yaw': 0.0}
        }
        self.found_cubes = [] # List to store the final logged cubes

        self.state = 'WAIT_FOR_DATA'
        self.search_prev_yaw = None
        self.search_accum_yaw = 0.0
        self.shutdown_requested = False

        print('[BOOT] Arena Mapper started. Waiting for sensors...')

    # =========================
    # Math & Control Utils
    # =========================
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_cmd(self, angular_z=0.0):
        try:
            msg = Twist()
            msg.angular.z = float(angular_z)
            self.cmd_pub.publish(msg)
        except Exception:
            pass

    def stop_robot(self):
        for _ in range(5):
            self.publish_cmd(0.0)
            time.sleep(0.02)

    def ready(self):
        return self.has_odom and self.has_image

    def set_state(self, new_state, text=None):
        if self.state == new_state: return
        self.state = new_state
        if text: print(f'\n[STATE] {text}')

        if new_state == 'SWEEP_ARENA':
            self.search_prev_yaw = self.local_yaw
            self.search_accum_yaw = 0.0

    # =========================
    # Callbacks
    # =========================
    def odom_callback(self, msg):
        self.world_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        if self.init_world_yaw is None:
            self.init_world_yaw = self.world_yaw
            print('[EVENT] Placed on line. World frame initialized.')

        self.local_yaw = self.normalize_angle(self.world_yaw - self.init_world_yaw)
        self.has_odom = True

    def image_callback(self, msg):
        self.has_image = True
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.image_width = frame.shape[1]
        except Exception:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Red Mask
        mask1 = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 80, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask1, mask2)

        # 2. Blue Mask
        lower_blue = np.array([100, 80, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Process and Extract Data
        self.process_color('red', mask_red)
        self.process_color('blue', mask_blue)

    def process_color(self, color, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area, best_cx = 0.0, None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_AREA: continue
            x, y, w, h = cv2.boundingRect(contour)
            if area > best_area:
                best_area = area
                best_cx = x + w / 2.0

        if best_cx is not None:
            self.vis[color] = True
            self.cx[color] = best_cx
            self.area[color] = best_area
        else:
            self.vis[color] = False

    # =========================
    # Tracking & State Logic
    # =========================
    def manage_tracker(self, color):
        """ Watches a color, finds its peak area in the FOV, and logs it when it leaves. """
        tracker = self.trackers[color]
        is_visible = self.vis[color]
        cx = self.cx[color]
        area = self.area[color]

        # Check if the cube is within the central vertical band of the camera
        error = abs(cx - self.image_width / 2.0) if is_visible else float('inf')
        allowed_pixels = self.image_width * (self.CENTER_FOV / 2.0)
        in_center_fov = error <= allowed_pixels

        if is_visible and in_center_fov:
            if not tracker['active']:
                tracker['active'] = True
                tracker['peak_area'] = area
                tracker['peak_yaw'] = self.local_yaw
            else:
                if area > tracker['peak_area']:
                    tracker['peak_area'] = area
                    tracker['peak_yaw'] = self.local_yaw
        else:
            if tracker['active']:
                # Cube has left the center FOV. Save it to our permanent list!
                self.found_cubes.append({
                    'color': color.upper(),
                    'yaw': tracker['peak_yaw'],
                    'area': tracker['peak_area']
                })
                tracker['active'] = False

    def handle_sweep(self):
        # 1. Feed camera data to our debouncing trackers
        self.manage_tracker('red')
        self.manage_tracker('blue')

        # 2. Accumulate Rotation
        delta = self.normalize_angle(self.local_yaw - self.search_prev_yaw)
        self.search_accum_yaw += abs(delta)
        self.search_prev_yaw = self.local_yaw

        # 3. Check for 360 completion
        remaining = 2.0 * math.pi - self.search_accum_yaw
        if remaining <= math.radians(4):
            self.stop_robot()
            
            # Force close any active trackers in case a cube was caught at the very end
            self.vis['red'], self.vis['blue'] = False, False
            self.manage_tracker('red')
            self.manage_tracker('blue')
            
            self.set_state('REPORT_RESULTS', 'Sweep complete. Analyzing data...')
            return

        # 4. Rotate
        self.publish_cmd(self.SEARCH_SPEED)

    def handle_report(self):
        print("\n=================================================")
        print("             ARENA SURVEY COMPLETE               ")
        print("=================================================")
        
        wrong_cubes = []

        if not self.found_cubes:
            print("No cubes found in the arena!")
        else:
            for idx, cube in enumerate(self.found_cubes):
                c_color = cube['color']
                deg = math.degrees(cube['yaw'])
                
                # Left Side (Blue Zone) = Positive Yaws
                # Right Side (Red Zone) = Negative Yaws
                if cube['yaw'] >= 0:
                    zone = "LEFT (Blue Zone)"
                    is_wrong = (c_color == 'RED')
                else:
                    zone = "RIGHT (Red Zone)"
                    is_wrong = (c_color == 'BLUE')

                print(f"Cube {idx+1}: [{c_color}] found at {deg:4.0f} deg -> Location: {zone}")
                
                if is_wrong:
                    wrong_cubes.append((idx+1, c_color, zone))

        print("\n-------------------------------------------------")
        print("                MISPLACED CUBES                  ")
        print("-------------------------------------------------")
        if not wrong_cubes:
            print("All cubes are in their correct zones! Good job.")
        else:
            for bad_cube in wrong_cubes:
                print(f"WARNING: Cube {bad_cube[0]} is a {bad_cube[1]} cube, but is located in the {bad_cube[2]}!")
        
        print("=================================================\n")
        self.shutdown_requested = True


    def control_loop(self):
        if self.shutdown_requested:
            self.stop_robot()
            rclpy.shutdown()
            return

        if not self.ready():
            self.stop_robot()
            return

        if self.state == 'WAIT_FOR_DATA':
            self.stop_robot()
            self.set_state('SWEEP_ARENA', 'Sensors ready. Starting 360 clockwise sweep...')
        elif self.state == 'SWEEP_ARENA':
            self.handle_sweep()
        elif self.state == 'REPORT_RESULTS':
            self.handle_report()

def main(args=None):
    rclpy.init(args=args)
    node = ArenaMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()