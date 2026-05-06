#!/usr/bin/env python3
import math
import time

import cv2
import numpy as np
import rclpy

import v11 as v11_base
import v12 as base


# Force red-only for the blue-floor test environment.
v11_base.TARGET_COLOR = "red"
v11_base.PREFER_BLUE = False


class SimpleCubeMissionV13(base.SimpleCubeMissionV12):
    def __init__(self):
        super().__init__()
        self.get_logger().info("SimpleCubeMission v13 active")
        print("[BOOT] SimpleCubeMission v13")
        print("[FIX] red-only, stop 1s after center hit, fine-align before approach")
        print("[FIX] servo_down double pulse, backup 2s at 2x speed")

    def set_state(self, new_state, text=""):
        if self.state == new_state:
            return
        self.state = new_state
        self.state_enter_time = time.monotonic()
        if new_state == "SEARCH":
            self.center_seen_frames = 0
            self.grab_color = None
        if new_state in ("CENTER_STOP", "FINE_ALIGN"):
            self.center_seen_frames = 0
        if text:
            print(f"[STATE] {new_state} | {text}")
        else:
            print(f"[STATE] {new_state}")

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

        if self.state in ("WAIT_FOR_DATA", "SEARCH", "CENTER_STOP", "FINE_ALIGN", "APPROACH"):
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

    def handle_search(self):
        if self.front_dist <= base.EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        if self.target_visible and self.target_obs is not None:
            if self.target_centered():
                self.grab_color = self.target_obs.color
                self.stop_robot_once()
                self.set_state("CENTER_STOP", f"{self.grab_color} centered, stop 1s")
                return

        self.publish_cmd(0.0, base.SEARCH_ANG)

    def handle_center_stop(self):
        self.stop_robot_once()
        if self.state_age() >= base.CENTER_STOP_SEC:
            self.set_state("FINE_ALIGN", "fine align before straight approach")

    def handle_fine_align(self):
        if self.front_dist <= base.EMERGENCY_STOP_M:
            self.set_state("STOPPED", "emergency front distance")
            return

        if not self.target_visible or self.target_obs is None:
            self.center_seen_frames = 0
            self.set_state("SEARCH", "target lost during fine align")
            return

        err = self.target_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) <= base.FINE_ALIGN_PIXEL_TOL:
            self.center_seen_frames += 1
            self.stop_robot_once()
            if self.center_seen_frames >= base.FINE_ALIGN_HOLD_FRAMES:
                self.set_state("APPROACH", f"{self.target_obs.color} fine aligned, drive straight")
            return

        self.center_seen_frames = 0
        err_norm = err / max(self.image_width / 2.0, 1.0)
        angular = self.clamp_abs(
            -0.18 * err_norm,
            base.FINE_ALIGN_MIN_ANG,
            base.FINE_ALIGN_MAX_ANG,
        )
        self.publish_cmd(0.0, angular)

    def handle_servo_down(self):
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_down()
        time.sleep(0.12)
        self.servo.servo_down()
        self.start_turn_relative(math.radians(base.TURN_AFTER_GRAB_TARGET_DEG), "TURN_180_AFTER_GRAB")

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
            "CENTER_STOP": self.handle_center_stop,
            "FINE_ALIGN": self.handle_fine_align,
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


def main(args=None):
    rclpy.init(args=args)
    node = SimpleCubeMissionV13()
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
