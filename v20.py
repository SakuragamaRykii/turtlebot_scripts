#!/usr/bin/env python3
import math
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Float32, Int32, String

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None

try:
    from flask import Flask, Response
except Exception:
    Flask = None
    Response = None

try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None


POINTBLANK_Y = 228.0
COLOR_TO_WALL = {"RED": 0, "BLUE": 23}

SCAN_SPEED = 0.22
CENTRE_SPEED = 0.18
APPROACH_FWD = 0.07
APPROACH_P_GAIN = 0.003
PUSH_FWD = 0.15
PUSH_P_GAIN = 0.002
PIVOT_SPEED = 0.22

WALL_STOP_DIST = 0.52
REVERSE_FWD = -0.20
REVERSE_DURATION = 1.8

CENTER_TARGET_WIDTH = 41.0
WIDTH_TOLERANCE = 2.0

CENTRE_PIXEL_THRESH = 20.0
PIVOT_PIXEL_THRESH = 20.0
APPROACH_LOST_LIMIT = 8
PUSH_TIMEOUT = 12.0
PIVOT_TIMEOUT = 35.0
PUSH_BLIND_TIME = 0.5

SERVO_GPIO_BCM = 12
SERVO_PWM_HZ = 50
SERVO_START_DUTY = 0.0
SERVO_UP_DUTY = 9.0
SERVO_DOWN_DUTY = 4.2
SERVO_SETTLE_SEC = 0.55
SERVO_DOUBLE_DOWN = True


class ServoHelper:
    def __init__(self):
        self.enabled = GPIO is not None
        self.servo = None
        if not self.enabled:
            print("[SERVO] RPi.GPIO not available, servo disabled")
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_GPIO_BCM, GPIO.OUT)
        self.servo = GPIO.PWM(SERVO_GPIO_BCM, SERVO_PWM_HZ)
        self.servo.start(SERVO_START_DUTY)
        time.sleep(0.1)
        self.servo.ChangeDutyCycle(0.0)
        print(f"[SERVO] BCM {SERVO_GPIO_BCM}, {SERVO_PWM_HZ}Hz ready")

    def set_duty(self, duty):
        if not self.enabled or self.servo is None:
            print(f"[SERVO] duty={duty:.2f} skipped")
            return
        self.servo.ChangeDutyCycle(float(duty))
        time.sleep(SERVO_SETTLE_SEC)
        self.servo.ChangeDutyCycle(0.0)

    def servo_up(self):
        print(f"[SERVO] UP duty={SERVO_UP_DUTY}")
        self.set_duty(SERVO_UP_DUTY)

    def servo_down(self):
        print(f"[SERVO] DOWN duty={SERVO_DOWN_DUTY}")
        self.set_duty(SERVO_DOWN_DUTY)
        if SERVO_DOUBLE_DOWN:
            time.sleep(0.12)
            self.set_duty(SERVO_DOWN_DUTY)

    def cleanup(self):
        if not self.enabled:
            return
        try:
            if self.servo is not None:
                self.servo.ChangeDutyCycle(0.0)
                self.servo.stop()
            GPIO.cleanup(SERVO_GPIO_BCM)
        except Exception:
            pass


class StateMachineV20(Node):
    def __init__(self):
        super().__init__("state_machine_v20")

        self.create_subscription(Bool, "/target_found", self._cb_t_found, 10)
        self.create_subscription(Float32, "/color_error", self._cb_t_error, 10)
        self.create_subscription(Float32, "/target_y", self._cb_t_y, 10)
        self.create_subscription(String, "/target_color", self._cb_t_color, 10)
        self.create_subscription(Bool, "/aruco_detected", self._cb_a_found, 10)
        self.create_subscription(Int32, "/aruco_id", self._cb_a_id, 10)
        self.create_subscription(Float32, "/aruco_error", self._cb_a_error, 10)
        self.create_subscription(Float32, "/aruco_width", self._cb_a_width, 10)
        self.create_subscription(LaserScan, "/scan", self._cb_scan, qos_profile_sensor_data)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.servo = ServoHelper()

        self.t_found = False
        self.t_error = 0.0
        self.t_y = 0.0
        self.t_color = "NONE"
        self.a_found = False
        self.a_id = -1
        self.a_error = 0.0
        self.a_width = 0.0
        self.forward_dist = 999.0

        self.state = "SCAN"
        self.carried_color = "NONE"
        self.target_wall_id = -1
        self.lost_count = 0
        self.reversing = False
        self.reverse_start = 0.0
        self.push_start = 0.0
        self.pivot_start = 0.0
        self.deliveries = 0

        self.current_zone = "UNKNOWN"
        self.confidence_counter = 0
        self.last_y = 0.0
        self.last_color = "NONE"
        self.last_seen_a_id = -1
        self.arm_is_down = False

        self.servo.servo_up()

        self.timer = self.create_timer(0.1, self._loop)
        self.get_logger().info("========================================")
        self.get_logger().info("[SYSTEM] V20 Brain Started with RPi.GPIO Servo Integration")
        self.get_logger().info("[SERVO] Boot arm raised. Capture lowers arm. Wall push raises arm.")
        self.get_logger().info("========================================")

    def _cb_t_found(self, msg):
        self.t_found = msg.data

    def _cb_t_error(self, msg):
        self.t_error = msg.data

    def _cb_t_y(self, msg):
        self.t_y = msg.data

    def _cb_t_color(self, msg):
        self.t_color = msg.data

    def _cb_a_found(self, msg):
        self.a_found = msg.data

    def _cb_a_id(self, msg):
        self.a_id = msg.data

    def _cb_a_error(self, msg):
        self.a_error = msg.data

    def _cb_a_width(self, msg):
        self.a_width = msg.data

    def _cb_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return

        inc = msg.angle_increment if msg.angle_increment > 0 else 0.0175
        idx_20 = int(math.radians(20) / inc)
        idx_45 = int(math.radians(45) / inc)

        valid_ranges = []
        for i in range(idx_20, idx_45):
            dist = msg.ranges[i % n]
            if math.isfinite(dist) and dist > 0.15:
                valid_ranges.append(dist)

        for i in range(-idx_45, -idx_20):
            dist = msg.ranges[i % n]
            if math.isfinite(dist) and dist > 0.15:
                valid_ranges.append(dist)

        self.forward_dist = min(valid_ranges) if valid_ranges else 999.0

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _spin(self, speed):
        cmd = Twist()
        cmd.angular.z = float(speed)
        self.cmd_pub.publish(cmd)

    def _drive(self, linear, angular=0.0):
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)

    def _reset_approach(self):
        self.lost_count = 0
        self.last_y = 0.0
        self.last_color = "NONE"

    def _go_scan(self):
        self._reset_approach()
        self.carried_color = "NONE"
        self.target_wall_id = -1
        self.confidence_counter = 0
        self.state = "SCAN"
        self._stop()
        self.get_logger().info("[STATE CHANGE] Back to SCAN")

    def _start_reverse(self):
        self.reverse_start = time.time()
        self.reversing = True
        self.get_logger().info("[STATE CHANGE] REVERSE")

    def _lower_arm_for_capture(self):
        self._stop()
        self.servo.servo_down()
        self.arm_is_down = True

    def _raise_arm_for_release(self, reason):
        self._stop()
        if self.arm_is_down:
            self.get_logger().info(f"[SERVO] Release block: {reason}")
            self.servo.servo_up()
            self.arm_is_down = False

    def _loop(self):
        if self.a_found:
            if self.a_id == 0:
                self.current_zone = "RED_ZONE"
            elif self.a_id == 7 and self.a_error > 100.0:
                self.current_zone = "RED_ZONE"
            elif self.a_id == 23:
                self.current_zone = "BLUE_ZONE"
            elif self.a_id == 42 and self.a_error > 100.0:
                self.current_zone = "BLUE_ZONE"

        if self.reversing:
            if time.time() - self.reverse_start < REVERSE_DURATION:
                self._drive(REVERSE_FWD)
            else:
                self.reversing = False
                self._stop()
                self.state = "RETURN_CENTER"
                self.get_logger().info("[STATE CHANGE] RETURN_CENTER")
                self.last_seen_a_id = -1
            return

        if self.state == "SCAN":
            self._state_scan()
        elif self.state == "CENTRE":
            self._state_centre()
        elif self.state == "APPROACH":
            self._state_approach()
        elif self.state == "PIVOT":
            self._state_pivot()
        elif self.state == "PUSH":
            self._state_push()
        elif self.state == "RETURN_CENTER":
            self._state_return_center()

    def _state_return_center(self):
        if self.a_found:
            if self.last_seen_a_id != self.a_id:
                self.get_logger().info(f"[PIXEL RETURN] Wall ID={self.a_id}, width={self.a_width:.1f}")
                self.last_seen_a_id = self.a_id

            err = self.a_error
            steer = -(err / 160.0) * PIVOT_SPEED
            steer = max(-PIVOT_SPEED, min(PIVOT_SPEED, steer))

            if abs(self.a_width - CENTER_TARGET_WIDTH) <= WIDTH_TOLERANCE:
                if abs(err) < PIVOT_PIXEL_THRESH:
                    self.get_logger().info(f"[RETURN_CENTER] Center reached. deliveries={self.deliveries + 1}")
                    self.deliveries += 1
                    self._go_scan()
                else:
                    self._drive(0.0, steer)
            elif self.a_width < (CENTER_TARGET_WIDTH - WIDTH_TOLERANCE):
                self._drive(0.12, steer)
            elif self.a_width > (CENTER_TARGET_WIDTH + WIDTH_TOLERANCE):
                self._drive(-0.12, steer)
        else:
            self.last_seen_a_id = -1
            self._spin(PIVOT_SPEED)

    def _state_scan(self):
        if self.t_found and self.t_color in ("RED", "BLUE"):
            if (self.t_color == "RED" and self.current_zone == "RED_ZONE") or (
                self.t_color == "BLUE" and self.current_zone == "BLUE_ZONE"
            ):
                self._spin(SCAN_SPEED)
            else:
                self.confidence_counter += 1
                if self.confidence_counter > 5:
                    self.get_logger().info(f"[STATE CHANGE] SCAN -> CENTRE ({self.t_color})")
                    self._stop()
                    self.confidence_counter = 0
                    self.state = "CENTRE"
                else:
                    self._spin(0.1)
        else:
            self.confidence_counter = 0
            self._spin(SCAN_SPEED)

    def _state_centre(self):
        if not self.t_found or self.t_color not in ("RED", "BLUE"):
            self.get_logger().warn("[CENTRE -> SCAN] Block lost while centring")
            self._go_scan()
            return

        error = self.t_error
        if abs(error) < CENTRE_PIXEL_THRESH:
            self.get_logger().info("[STATE CHANGE] CENTRE -> APPROACH")
            self._reset_approach()
            self._stop()
            self.state = "APPROACH"
        else:
            steer = -float(error) * CENTRE_SPEED / 80.0
            steer = max(-CENTRE_SPEED, min(CENTRE_SPEED, steer))
            self._spin(steer)

    def _state_approach(self):
        if self.t_found and self.t_color in ("RED", "BLUE"):
            self.lost_count = 0
            self.last_y = self.t_y
            self.last_color = self.t_color
            steer = -(self.t_error * APPROACH_P_GAIN)
            self._drive(APPROACH_FWD, steer)
        else:
            self.lost_count += 1
            if self.lost_count >= APPROACH_LOST_LIMIT:
                if self.last_y > POINTBLANK_Y:
                    self._confirm_capture(f"blind spot capture, last_y={self.last_y:.0f}")
                else:
                    self.get_logger().warn("[APPROACH -> SCAN] Target lost too early")
                    self._go_scan()
            else:
                self._drive(APPROACH_FWD)

    def _confirm_capture(self, reason: str):
        color = self.last_color
        if color not in COLOR_TO_WALL:
            self._go_scan()
            return

        self.carried_color = color
        self.target_wall_id = COLOR_TO_WALL[color]
        self._reset_approach()
        self.get_logger().info(
            f"[STATE CHANGE] APPROACH -> ARM_DOWN -> PIVOT ({reason}, target wall {self.target_wall_id})"
        )
        self._lower_arm_for_capture()
        self.pivot_start = time.time()
        self.state = "PIVOT"

    def _state_pivot(self):
        elapsed = time.time() - self.pivot_start
        if elapsed > PIVOT_TIMEOUT:
            self.get_logger().error("[PIVOT -> REVERSE] timeout, releasing block")
            self._raise_arm_for_release("pivot timeout")
            self._start_reverse()
            return

        if self.a_found:
            if self.last_seen_a_id != self.a_id:
                self.get_logger().info(f"[VISION RADAR] ArUco ID={self.a_id}")
                self.last_seen_a_id = self.a_id

            if self.a_id == self.target_wall_id:
                err = self.a_error
                if abs(err) < PIVOT_PIXEL_THRESH:
                    self.get_logger().info("[STATE CHANGE] PIVOT -> PUSH")
                    self._stop()
                    self.push_start = time.time()
                    self.state = "PUSH"
                else:
                    steer = -(err / 160.0) * PIVOT_SPEED
                    steer = max(-PIVOT_SPEED, min(PIVOT_SPEED, steer))
                    self._spin(steer)
            else:
                self._spin(PIVOT_SPEED)
        else:
            self.last_seen_a_id = -1
            self._spin(PIVOT_SPEED)

    def _state_push(self):
        elapsed = time.time() - self.push_start

        if elapsed > PUSH_BLIND_TIME and self.forward_dist < WALL_STOP_DIST:
            self.get_logger().warn(f"[PUSH -> RELEASE/REVERSE] wall reached dist={self.forward_dist:.3f}")
            self._raise_arm_for_release("wall reached")
            self._start_reverse()
            return

        if elapsed > PUSH_TIMEOUT:
            self.get_logger().error("[PUSH -> RELEASE/REVERSE] timeout")
            self._raise_arm_for_release("push timeout")
            self._start_reverse()
            return

        if self.a_found and self.a_id == self.target_wall_id:
            steer = -(self.a_error * PUSH_P_GAIN)
        else:
            steer = 0.0
        self._drive(PUSH_FWD, steer)

    def destroy_node(self):
        try:
            self._stop()
        except Exception:
            pass
        try:
            self.servo.cleanup()
        except Exception:
            pass
        super().destroy_node()


class ColorTracking(Node):
    def __init__(self):
        super().__init__("color_tracking_v20")

        if CvBridge is None:
            raise RuntimeError("cv_bridge is required for ColorTracking")
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.target_pub = self.create_publisher(Bool, "/target_found", 10)
        self.error_pub = self.create_publisher(Float32, "/color_error", 10)
        self.area_pub = self.create_publisher(Float32, "/target_area", 10)
        self.y_pub = self.create_publisher(Float32, "/target_y", 10)
        self.color_pub = self.create_publisher(String, "/target_color", 10)

        self.green_pub = self.create_publisher(Bool, "/green_detected", 10)
        self.green_y_pub = self.create_publisher(Float32, "/green_y", 10)

        self.aruco_pub = self.create_publisher(Bool, "/aruco_detected", 10)
        self.aruco_id_pub = self.create_publisher(Int32, "/aruco_id", 10)
        self.aruco_error_pub = self.create_publisher(Float32, "/aruco_error", 10)
        self.aruco_width_pub = self.create_publisher(Float32, "/aruco_width", 10)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_aruco_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_aruco_api = False

        self.app = Flask(__name__) if Flask is not None else None
        self.frame = None
        self.frame_lock = threading.Lock()
        if self.app is not None:
            threading.Thread(target=self.run_flask, daemon=True).start()

        self.last_print_time = 0.0
        self.last_target = {"found": False, "color": "NONE", "error": 0.0, "area": 0.0, "y": 0.0}
        self.lost_count = 0
        self.MAX_LOST_FRAMES = 8

        self.get_logger().info("Integrated Color Tracking Node Started (V20)")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            try:
                frame = self.bridge.imgmsg_to_cv2(msg)
                if msg.encoding == "nv21":
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV21)
                else:
                    return
            except Exception:
                return

        if frame is None:
            return

        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        frame = cv2.LUT(frame, table)

        height, width, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 140, 60])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 140, 60])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red += cv2.inRange(hsv, lower_red2, upper_red2)

        lower_blue = np.array([102, 145, 55])
        upper_blue = np.array([132, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        lower_green = np.array([40, 80, 80])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        targets = []
        for contour in contours_red:
            area = cv2.contourArea(contour)
            if area < 200:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.5 or aspect_ratio > 2.2:
                continue
            roi = hsv[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            mean_s = np.mean(roi[:, :, 1])
            mean_v = np.mean(roi[:, :, 2])
            if mean_s < 120 or mean_v < 50:
                continue
            targets.append({"color": "RED", "area": area, "cx": x + w // 2, "bottom_y": y + h, "bbox": (x, y, w, h)})

        for contour in contours_blue:
            area = cv2.contourArea(contour)
            if area < 160:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.55 or aspect_ratio > 1.9:
                continue
            if area > 25000:
                continue
            if x < 5 or x + w > width - 5:
                continue
            roi = hsv[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            mean_s = np.mean(roi[:, :, 1])
            mean_v = np.mean(roi[:, :, 2])
            if mean_s < 115 or mean_v < 45:
                continue
            targets.append({"color": "BLUE", "area": area, "cx": x + w // 2, "bottom_y": y + h, "bbox": (x, y, w, h)})

        target_msg = Bool()
        error_msg = Float32()
        area_msg = Float32()
        y_msg = Float32()
        color_msg = String()

        if targets:
            red_targets = [t for t in targets if t["color"] == "RED"]
            blue_targets = [t for t in targets if t["color"] == "BLUE"]
            billiard_mode = False
            target = None

            if red_targets and blue_targets:
                best_red = max(red_targets, key=lambda t: t["area"])
                best_blue = max(blue_targets, key=lambda t: t["area"])
                dist = math.hypot(best_red["cx"] - best_blue["cx"], best_red["bottom_y"] - best_blue["bottom_y"])
                if dist < 120.0:
                    billiard_mode = True
                    fake_cx = best_red["cx"] - 240 if best_blue["cx"] > best_red["cx"] else best_red["cx"] + 240
                    target = {
                        "color": "RED",
                        "area": best_red["area"],
                        "cx": fake_cx,
                        "bottom_y": best_red["bottom_y"],
                        "bbox": best_red["bbox"],
                    }

            if not billiard_mode:
                target = max(targets, key=lambda t: t["area"])

            error = target["cx"] - width / 2
            target_msg.data = True
            error_msg.data = float(error)
            area_msg.data = float(target["area"])
            y_msg.data = float(target["bottom_y"])
            color_msg.data = target["color"]
            self.lost_count = 0
            self.last_target = {
                "found": True,
                "color": target["color"],
                "error": float(error),
                "area": float(target["area"]),
                "y": float(target["bottom_y"]),
            }
        else:
            self.lost_count += 1
            if self.lost_count <= self.MAX_LOST_FRAMES and self.last_target["found"]:
                target_msg.data = True
                error_msg.data = float(self.last_target["error"])
                area_msg.data = float(self.last_target["area"])
                y_msg.data = float(self.last_target["y"])
                color_msg.data = self.last_target["color"]
            else:
                target_msg.data = False
                error_msg.data = 0.0
                area_msg.data = 0.0
                y_msg.data = 0.0
                color_msg.data = "NONE"
                self.last_target["found"] = False

        green_detected = False
        green_y = 0.0
        best_green = None
        best_green_area = 0.0
        for c in contours_green:
            area = cv2.contourArea(c)
            if area < 1500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 2 * h:
                continue
            if area > best_green_area:
                best_green_area = area
                best_green = (x, y, w, h)
        if best_green is not None:
            x, y, w, h = best_green
            green_detected = True
            green_y = float(y + h / 2.0)

        green_msg = Bool()
        green_y_msg = Float32()
        green_msg.data = green_detected
        green_y_msg.data = green_y

        aruco_found_msg = Bool()
        aruco_id_msg = Int32()
        aruco_error_msg = Float32()
        aruco_width_msg = Float32()
        aruco_detected = False
        aruco_id = -1
        aruco_error = 0.0
        aruco_width = 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.use_new_aruco_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            best_i = 0
            best_width = 0.0
            for i in range(len(ids)):
                pts = corners[i][0]
                width_px = float(np.linalg.norm(pts[0] - pts[1]))
                if width_px > best_width:
                    best_width = width_px
                    best_i = i
            pts = corners[best_i][0]
            aruco_id = int(ids[best_i][0])
            cx = int(np.mean(pts[:, 0]))
            aruco_error = float(cx - width / 2)
            aruco_width = best_width
            aruco_detected = True

        aruco_found_msg.data = aruco_detected
        aruco_id_msg.data = aruco_id
        aruco_error_msg.data = aruco_error
        aruco_width_msg.data = aruco_width

        self.target_pub.publish(target_msg)
        self.error_pub.publish(error_msg)
        self.area_pub.publish(area_msg)
        self.y_pub.publish(y_msg)
        self.color_pub.publish(color_msg)
        self.green_pub.publish(green_msg)
        self.green_y_pub.publish(green_y_msg)
        self.aruco_pub.publish(aruco_found_msg)
        self.aruco_id_pub.publish(aruco_id_msg)
        self.aruco_error_pub.publish(aruco_error_msg)
        self.aruco_width_pub.publish(aruco_width_msg)

        with self.frame_lock:
            self.frame = frame.copy()

        now = time.time()
        if now - self.last_print_time > 0.5:
            self.last_print_time = now
            print("\n===== VISION STATUS =====")
            print(f"Target found : {target_msg.data}")
            print(f"Target color : {color_msg.data}")
            print(f"Target area  : {area_msg.data:.1f}")
            print(f"Target error : {error_msg.data:.1f}")
            print(f"Target y     : {y_msg.data:.1f}")
            print(f"Lost count   : {self.lost_count}")
            print(f"Green line   : {green_detected}, y={green_y:.1f}")
            print(f"ArUco found  : {aruco_detected}, id={aruco_id}, err={aruco_error:.1f}, width={aruco_width:.1f}")
            print("=========================\n")

    def generate_frames(self):
        while True:
            with self.frame_lock:
                frame = None if self.frame is None else self.frame.copy()
            if frame is not None:
                ret, buffer = cv2.imencode(".jpg", frame)
                if ret:
                    jpg = buffer.tobytes()
                    yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n\r\n")
            time.sleep(0.03)

    def run_flask(self):
        @self.app.route("/video_feed")
        def video_feed():
            return Response(self.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

        self.app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


def main(args=None):
    rclpy.init(args=args)
    vision_node = ColorTracking()
    brain_node = StateMachineV20()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(vision_node)
    executor.add_node(brain_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            brain_node.destroy_node()
        except Exception:
            pass
        try:
            vision_node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
