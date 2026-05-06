import time
import math
import cv2
import numpy as np
from enum import Enum
try:
    from gpiozero import Servo
    from gpiozero.pins.pigpio import PiGPIOFactory
except ImportError:
    # Fallback or stub for environment without gpiozero installed
    Servo = None

# =====================================================================
# State Enum Definition
# =====================================================================
class State(Enum):
    WAIT_FOR_DATA = 1
    IDLE = 2
    SEARCH = 3
    TURN_TO_APPROACH = 4
    APPROACH = 5
    GRAB_CUBE = 6
    PLAN_DELIVERY = 7
    TURN_TO_DELIVER = 8
    DELIVER = 9
    RELEASE_CUBE = 10
    TURN_HOME = 11
    RETURN_HOME = 12

# =====================================================================
# Hardware Abstraction Layer & Interfaces
# =====================================================================
class TurtleBotController:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.held_cube_color = None # 'red' or 'blue'
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_heading = 0.0
        self.cube_seen = False
        
        # Initialize OpenCV Video Capture (use 0 or the video path/index for Picamera stream)
        self.cap = cv2.VideoCapture(0)
        
        # Initialize Servo on Pin 12 (GPIO 18 / Pin 12 on Raspberry Pi 5)
        # Assuming PiGPIOFactory is used to prevent issues on Pi 5
        if Servo:
            # factory = PiGPIOFactory() # Uncomment if using pigpio
            self.servo = Servo(18, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
            self.servo.detach()
        else:
            self.servo = None
            
    def read_sensors(self):
        """Simulates/Reads from OpenCR board, LiDAR, and Camera."""
        # 1. Camera check for cube colors
        ret, frame = self.cap.read()
        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Define color bounds for red and blue
            # Note: Red can wrap around the HSV 0 degree mark, so we use two ranges or a general detector.
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([130, 255, 255])
            
            mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            
            if np.sum(mask_red) > 500:
                self.cube_seen = True
                self.target_color = "red"
            elif np.sum(mask_blue) > 500:
                self.cube_seen = True
                self.target_color = "blue"
            else:
                self.cube_seen = False
                
    def set_speeds(self, linear_velocity, angular_velocity):
        """Send velocity commands to OpenCR Board."""
        # Interface to OpenCR (e.g., using serial or ROS 2 publishers)
        pass

    def get_distance(self):
        """Read from LiDAR / Ultrasonic or visual distance estimator."""
        return 1.0 # Mock distance
        
    def close(self):
        self.cap.release()
        if self.servo:
            self.servo.detach()

# =====================================================================
# State Machine Implementation
# =====================================================================
class CubeSorterStateMachine:
    def __init__(self):
        self.state = State.WAIT_FOR_DATA
        self.robot = TurtleBotController()
        
        # Workspace Parameters
        self.arena_size = 2.0
        
        # State Data
        self.stored_heading = 0.0
        self.held_color = None
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_color = None
        
    def execute(self):
        while True:
            if self.state == State.WAIT_FOR_DATA:
                print("[STATE] WAIT_FOR_DATA: Waiting for topics and sensor initialization...")
                self.robot.read_sensors()
                # Dummy wait condition simulation
                time.sleep(1)
                self.state = State.IDLE
                
            elif self.state == State.IDLE:
                print("[STATE] IDLE: Waiting for user to press 's' to start...")
                # To capture keyboard presses in Python without a GUI:
                # Use standard terminal input or keep it automated for demo.
                user_input = 's' # Simulated key press
                if user_input.lower() == 's':
                    self.state = State.SEARCH
                time.sleep(1)
                
            elif self.state == State.SEARCH:
                print("[STATE] SEARCH: Rotating in place to find a cube...")
                self.robot.set_speeds(0.0, 0.30)
                self.robot.read_sensors()
                
                # Check for misplaced cube:
                # Right is red (+y in arena, or +x relative to zone), Left is blue (-y in arena, or -x).
                # We save the heading and color once found.
                if self.robot.cube_seen:
                    # Determine if it is misplaced:
                    if self.robot.target_color == "red" and self.robot.y > 0.0: # Misplaced in Blue zone
                        self.target_color = "red"
                        self.stored_heading = self.robot.theta
                        self.state = State.TURN_TO_APPROACH
                    elif self.robot.target_color == "blue" and self.robot.y < 0.0: # Misplaced in Red zone
                        self.target_color = "blue"
                        self.stored_heading = self.robot.theta
                        self.state = State.TURN_TO_APPROACH
                time.sleep(0.5)
                
            elif self.state == State.TURN_TO_APPROACH:
                print(f"[STATE] TURN_TO_APPROACH: Turning to heading {self.stored_heading}...")
                # Rotate toward the stored heading.
                self.robot.set_speeds(0.0, 0.2)
                # Simulated alignment complete
                self.state = State.APPROACH
                time.sleep(1)
                
            elif self.state == State.APPROACH:
                print("[STATE] APPROACH: Driving forward while using visual servoing...")
                self.robot.set_speeds(0.12, 0.0)
                distance = self.robot.get_distance()
                
                # Check condition for stopping
                if distance <= 0.15 or not self.robot.cube_seen:
                    self.robot.set_speeds(0.0, 0.0)
                    self.state = State.GRAB_CUBE
                time.sleep(0.5)
                
            elif self.state == State.GRAB_CUBE:
                print("[STATE] GRAB_CUBE: Actuating servo at pin 12...")
                if self.robot.servo is not None:
                    self.robot.servo.value = 0.5 # Lower grabber
                time.sleep(1)
                self.held_color = self.robot.target_color
                self.state = State.PLAN_DELIVERY
                
            elif self.state == State.PLAN_DELIVERY:
                print("[STATE] PLAN_DELIVERY: Calculating drop-off coordinates...")
                if self.held_color == "red":
                    self.target_x = 0.0
                    self.target_y = -0.1
                elif self.held_color == "blue":
                    self.target_x = 0.0
                    self.target_y = 0.1
                self.state = State.TURN_TO_DELIVER
                
            elif self.state == State.TURN_TO_DELIVER:
                print("[STATE] TURN_TO_DELIVER: Turning toward delivery point...")
                self.state = State.DELIVER
                time.sleep(1)
                
            elif self.state == State.DELIVER:
                print("[STATE] DELIVER: Moving to delivery coordinates...")
                self.robot.set_speeds(0.12, 0.0)
                # Distance logic simulation
                time.sleep(2)
                self.state = State.RELEASE_CUBE
                
            elif self.state == State.RELEASE_CUBE:
                print("[STATE] RELEASE_CUBE: Releasing held cube...")
                if self.robot.servo is not None:
                    self.robot.servo.value = -0.5 # Open grabber
                self.held_color = None
                self.target_color = None
                self.state = State.TURN_HOME
                
            elif self.state == State.TURN_HOME:
                print("[STATE] TURN_HOME: Turning to face origin (0,0)...")
                self.stored_heading = 0.0
                self.state = State.RETURN_HOME
                time.sleep(1)
                
            elif self.state == State.RETURN_HOME:
                print("[STATE] RETURN_HOME: Driving back to origin...")
                self.robot.set_speeds(0.12, 0.0)
                # Simulated arrival at home
                self.state = State.IDLE
                time.sleep(1)
                break # Stops for safe shutdown (or loop infinitely)

# =====================================================================
# Main Execution
# =====================================================================
if __name__ == '__main__':
    try:
        sm = CubeSorterStateMachine()
        sm.execute()
    except KeyboardInterrupt:
        print("\nExiting cube sorter state machine.")