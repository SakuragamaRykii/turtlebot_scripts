#!/usr/bin/env python3
import math
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan

from v11 import (
    APPROACH_SPEED,
    BACKUP_AFTER_RELEASE_SEC,
    BACKUP_SPEED,
    CONTROL_DT,
    CubeDetector,
    EMERGENCY_STOP_M,
    EXTRA_FORWARD_AFTER_LOST_SEC,
    EXTRA_FORWARD_SPEED,
    FORWARD_AFTER_TURN_SEC,
    FORWARD_AFTER_TURN_SPEED,
    MIN_CENTER_FRAMES,
    SEARCH_ANG,
    SEARCH_CENTER_BAND_PX,
    SERVO_DOWN_DUTY,
    SERVO_GPIO_BCM,
    SERVO_UP_DUTY,
    STATUS_DT,
    TURN_180_TOL_DEG,
    TURN_AFTER_GRAB_MAX_ANG,
    TURN_RETURN_MAX_ANG,
    ServoHelper,
)


class SimpleCubeMissionV12(Node):
    def __init__(self):
        super().__init__("simple_cube_mission_v12")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.control_timer = self.create_timer(CONTROL_DT, self.control_loop)
        self.status_timer = self.create_timer(STATUS_DT, self.status_loop)

        self.detector = CubeDetector()
        self.servo = ServoHelper()

        self.has_scan = False
        self.has_odom = False
        self.has_image = False
        self.front_dist = float("inf")
        self.image_width = None
        self.image_height = None
        self.image_logged = False

        self.world_yaw = 0.0
        self.init_yaw = None
        self.local_yaw = 0.0
        self.turn_target_yaw = 0.0

        self.target_visible = False
        self.target_obs = None
        self.red_obs = None
        self.blue_obs = None
        self.center_seen_frames = 0
        self.grab_color = None

        self.state = "WAIT_FOR_DATA"
        self.state_enter_time = time.monotonic()
        self.shutdown_requested = False
        self.shutdown_reason = ""
        self.shutdown_count = 0
        self.wait_status_last = None

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        print("[BOOT] SimpleCubeMission v12")
        print("[FIX] compressed image only, same camera path as v5/v7")
        print("[MODE] center-band lock -> straight approach -> target lost => forward 1.5s -> servo sequence")
        print(f"[SERVO] UP={SERVO_UP_DUTY} DOWN={SERVO_DOWN_DUTY} BCM={SERVO_GPIO_BCM}")
        print("[CMD] type H then Enter to stop and exit")

    def ready(self):
        return self.has_scan and self.has_odom and self.has_image

    def state_age(self):
        return time.monotonic() - self.state_enter_time

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def clamp_abs(self, value, min_abs, max_abs):
        if abs(value) < 1e-9:
            return 0.0
        return math.copysign(min(max(abs(value), min_abs), max_abs), value)

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        if not rclpy.ok():
            return
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_robot_once(self):
        self.publish_cmd(0.0, 0.0)

    def stop_robot_reliable(self, repeat=12, delay=0.025):
        if not rclpy.ok():
            return
        for _ in range(repeat):
            self.stop_robot_once()
            time.sleep(delay)

    def set_state(self, new_state, text=""):
        if self.state == new_state:
            return
        self.state = new_state
        self.state_enter_time = time.monotonic()
        if new_state == "SEARCH":
            self.center_seen_frames = 0
            self.grab_color = None
        if text:
            print(f"[STATE] {new_state} | {text}")
        else:
            print(f"[STATE] {new_state}")

    def start_turn_relative(self, radians_delta, state_name):
        self.turn_target_yaw = self.normalize_angle(self.local_yaw + radians_delta)
        self.set_state(state_name, f"target_yaw={self.turn_target_yaw:.2f}")

    def request_shutdown(self, reason):
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.shutdown_reason = reason
        self.shutdown_count = 0
        print(f"[STOP] {reason}")

    def console_loop(self):
        while True:
            try:
                cmd = input().strip().lower()
            except Exception:
                return
            if cmd == "h":
                self.request_shutdown("manual emergency stop")
                return

    def scan_callback(self, msg):
        ranges = list(msg.ranges)
        n = len(ranges)
        if n == 0:
            return
        vals = list(ranges[0:min(10, n)]) + list(ranges[max(0, n - 10):n])
        good = [x for x in vals if math.isfinite(x) and x > 0.04]
        raw_front = min(good) if good else float("inf")
        if not self.has_scan:
            self.front_dist = raw_front
        else:
            alpha = 0.55
            self.front_dist = alpha * self.front_dist + (1.0 - alpha) * raw_front
        self.has_scan = True

    def odom_callback(self, msg):
        self.world_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        if self.init_yaw is None:
            self.init_yaw = self.world_yaw
        self.local_yaw = self.normalize_angle(self.world_yaw - self.init_yaw)
        self.has_odom = True

    def image_callback(self, msg):
        self.has_image = True
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("decoded frame is None")
        except Exception as exc:
            print(f"[IMAGE] compressed decode failed: {exc}")
            self.target_obs = None
            self.target_visible = False
            return

        self.image_height, self.image_width = frame.shape[:2]
        if not self.image_logged:
            print(f"[IMAGE] compressed size={self.image_width}x{self.image_height}")
            self.image_logged = True

        if self.state in ("WAIT_FOR_DATA", "SEARCH", "APPROACH"):
            chosen, red_obs, blue_obs = self.detector.detect(frame)
            self.target_obs = chosen
            self.red_obs = red_obs
            self.blue_obs = blue_obs
            self.target_visible = chosen is not None
        else:
            self.target_obs = None
            self.red_obs = None
            self.blue_obs = None
            self.target_visible = False

    def target_error_pixels(self):
        if not self.target_visible or self.target_obs is None or self.image_width is None:
            return None
        return self.target_obs.cx - self.image_width / 2.0

    def target_centered(self):
        err = self.target_error_pixels()
        return err is not None and abs(err) <= SEARCH_CENTER_BAND_PX

    def handle_search(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        if self.target_visible and self.target_obs is not None:
            if self.target_centered():
                self.center_seen_frames += 1
                self.stop_robot_once()
                if self.center_seen_frames >= MIN_CENTER_FRAMES:
                    self.grab_color = self.target_obs.color
                    self.set_state("APPROACH", f"{self.grab_color} centered, drive straight")
                return
            self.center_seen_frames = 0

        self.publish_cmd(0.0, SEARCH_ANG)

    def handle_approach(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        if not self.target_visible or self.target_obs is None:
            self.set_state("EXTRA_FORWARD_AFTER_LOST", "target disappeared, forward 1.5s")
            return

        self.publish_cmd(APPROACH_SPEED, 0.0)

    def handle_extra_forward_after_lost(self):
        if self.state_age() < EXTRA_FORWARD_AFTER_LOST_SEC:
            self.publish_cmd(EXTRA_FORWARD_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.set_state("SERVO_DOWN", "lower arm")

    def handle_servo_down(self):
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_down()
        self.start_turn_relative(math.pi, "TURN_180_AFTER_GRAB")

    def handle_turn_180_after_grab(self):
        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= math.radians(TURN_180_TOL_DEG):
            self.stop_robot_once()
            self.set_state("FORWARD_AFTER_TURN", "drive forward 3s")
            return
        angular = self.clamp_abs(0.65 * error, 0.04, TURN_AFTER_GRAB_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_forward_after_turn(self):
        if self.state_age() < FORWARD_AFTER_TURN_SEC:
            self.publish_cmd(FORWARD_AFTER_TURN_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.set_state("SERVO_UP_RELEASE", "raise arm")

    def handle_servo_up_release(self):
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_up()
        self.set_state("BACKUP_AFTER_RELEASE", "backup 1s")

    def handle_backup_after_release(self):
        if self.state_age() < BACKUP_AFTER_RELEASE_SEC:
            self.publish_cmd(BACKUP_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.start_turn_relative(math.pi, "TURN_180_RETURN")

    def handle_turn_180_return(self):
        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= math.radians(TURN_180_TOL_DEG):
            self.stop_robot_once()
            self.set_state("SEARCH", "continue searching")
            return
        angular = self.clamp_abs(0.75 * error, 0.05, TURN_RETURN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def control_loop(self):
        if self.shutdown_requested:
            self.stop_robot_once()
            self.shutdown_count += 1
            if self.shutdown_count >= 10:
                self.stop_robot_reliable()
                self.servo.cleanup()
                rclpy.shutdown()
            return

        if not self.ready():
            self.stop_robot_once()
            return

        if self.state == "WAIT_FOR_DATA":
            self.stop_robot_once()
            self.servo.servo_up()
            self.set_state("SEARCH", "data ready, arm up")
            return

        handlers = {
            "SEARCH": self.handle_search,
            "APPROACH": self.handle_approach,
            "EXTRA_FORWARD_AFTER_LOST": self.handle_extra_forward_after_lost,
            "SERVO_DOWN": self.handle_servo_down,
            "TURN_180_AFTER_GRAB": self.handle_turn_180_after_grab,
            "FORWARD_AFTER_TURN": self.handle_forward_after_turn,
            "SERVO_UP_RELEASE": self.handle_servo_up_release,
            "BACKUP_AFTER_RELEASE": self.handle_backup_after_release,
            "TURN_180_RETURN": self.handle_turn_180_return,
            "STOPPED": self.stop_robot_once,
        }
        handlers.get(self.state, self.stop_robot_once)()

    def status_loop(self):
        if not self.ready():
            wait_now = (self.has_scan, self.has_odom, self.has_image)
            if wait_now != self.wait_status_last:
                print(f"[WAIT] scan={self.has_scan} odom={self.has_odom} image={self.has_image}")
                self.wait_status_last = wait_now
            return

        target_text = "target=none"
        if self.target_visible and self.target_obs is not None:
            err = self.target_error_pixels()
            target_text = (
                f"target={self.target_obs.color} err={err:.0f} "
                f"bbox=({self.target_obs.bbox_w}x{self.target_obs.bbox_h}) "
                f"holes={self.target_obs.holes} dist={self.target_obs.distance_cm:.1f}cm "
                f"center_frames={self.center_seen_frames}"
            )

        print(
            f"[STATUS] {self.state} {target_text} grab={self.grab_color} "
            f"front={self.front_dist:.3f}m yaw={math.degrees(self.local_yaw):.0f} "
            f"image=compressed {self.image_width}x{self.image_height}"
        )

    def destroy_node(self):
        try:
            self.stop_robot_reliable()
        except Exception:
            pass
        try:
            self.servo.cleanup()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SimpleCubeMissionV12()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.request_shutdown("KeyboardInterrupt")
        node.stop_robot_reliable()
    finally:
        try:
            if rclpy.ok():
                node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
