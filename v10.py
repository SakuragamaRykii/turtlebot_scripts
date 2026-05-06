#!/usr/bin/env python3
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan

try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None


# =========================
# Motion tuning
# =========================
SEARCH_ANG = 0.16
SEARCH_TRACK_MAX_ANG = 0.12
ALIGN_MIN_ANG = 0.03
ALIGN_MAX_ANG = 0.14
RECOVER_ANG = 0.07

APPROACH_FAST_SPEED = 0.045
APPROACH_SLOW_SPEED = 0.025
EXTRA_FORWARD_SPEED = 0.04
DELIVERY_FORWARD_SPEED = 0.07
BACKUP_SPEED = -0.13

ALIGN_PIXEL_TOL = 20
SEARCH_LOCK_PIXEL_TOL = 170
SEARCH_TRACK_PIXEL_TOL = 300
REACQUIRE_PIXEL_TOL = 110
CENTER_HOLD_FRAMES = 3
CONFIRM_FRAMES = 3
LOST_TARGET_TIMEOUT = 1.0

EXTRA_FORWARD_AFTER_LOST_SEC = 1.5
DELIVERY_FORWARD_SEC = 3.0
BACKUP_AFTER_RELEASE_SEC = 1.0

TURN_180_TOL_DEG = 4.0
TURN_180_SLOW_MAX_ANG = 0.14
TURN_180_NORMAL_MAX_ANG = 0.20

EMERGENCY_STOP_CM = 3.0
EMERGENCY_STOP_M = EMERGENCY_STOP_CM / 100.0


# =========================
# Red cube detection tuning
# =========================
RED_MIN_AREA = 650.0
RED_LOCK_MIN_AREA = 800.0
RED_MIN_BBOX_W = 18
RED_MIN_BBOX_H = 18
RED_MIN_ASPECT = 0.50
RED_MAX_ASPECT = 2.10
RED_MIN_FILL_RATIO = 0.20
RED_MIN_EXTENT = 0.16
RED_MIN_SOLIDITY = 0.50
RED_MIN_CENTER_Y_RATIO = 0.12
RED_MAX_RAW_JUMP_PX = 135.0

CLOSE_RED_BBOX_H_PX = 150


# =========================
# Servo tuning
# =========================
SERVO_GPIO_BCM = 12
SERVO_PWM_HZ = 50
SERVO_START_DUTY = 0.0

# v9 boot-up moved the arm down on your robot, so defaults are swapped here.
# If direction is still reversed, swap these two values.
SERVO_UP_DUTY = 9.0
SERVO_DOWN_DUTY = 5.5
SERVO_SETTLE_SEC = 0.55


CONTROL_DT = 0.05
STATUS_DT = 0.8


@dataclass
class RedObservation:
    cx: float
    cy: float
    area: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    fill_ratio: float
    extent: float
    solidity: float
    score: float


class ServoHelper:
    def __init__(self):
        self.enabled = GPIO is not None
        self.servo = None
        if not self.enabled:
            print('[SERVO] RPi.GPIO not available, servo disabled')
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_GPIO_BCM, GPIO.OUT)
        self.servo = GPIO.PWM(SERVO_GPIO_BCM, SERVO_PWM_HZ)
        self.servo.start(SERVO_START_DUTY)
        time.sleep(0.1)
        self.servo.ChangeDutyCycle(0.0)
        print(f'[SERVO] BCM {SERVO_GPIO_BCM}, {SERVO_PWM_HZ} Hz ready')

    def set_duty(self, duty):
        if not self.enabled or self.servo is None:
            print(f'[SERVO] duty={duty:.2f} skipped')
            return
        self.servo.ChangeDutyCycle(float(duty))
        time.sleep(SERVO_SETTLE_SEC)
        self.servo.ChangeDutyCycle(0.0)

    def servo_up(self):
        print(f'[SERVO] up duty={SERVO_UP_DUTY}')
        self.set_duty(SERVO_UP_DUTY)

    def servo_down(self):
        print(f'[SERVO] down duty={SERVO_DOWN_DUTY}')
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


class RedDetector:
    def __init__(self):
        self.image_width = 0
        self.image_height = 0

    def build_red_mask(self, hsv, bgr):
        lower_red_1 = np.array([0, 95, 55], dtype=np.uint8)
        upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([168, 95, 55], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2),
        )

        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)
        bgr_mask = np.zeros_like(hsv_mask)
        bgr_mask[(r >= 78) & (r > g + 28) & (r > b + 28)] = 255

        mask = cv2.bitwise_and(hsv_mask, bgr_mask)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask

    def detect(self, frame):
        self.image_height, self.image_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.build_red_mask(hsv, frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < RED_MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < RED_MIN_BBOX_W or h < RED_MIN_BBOX_H:
                continue

            cy = y + h / 2.0
            if cy < self.image_height * RED_MIN_CENTER_Y_RATIO:
                continue

            aspect = w / float(h)
            if aspect < RED_MIN_ASPECT or aspect > RED_MAX_ASPECT:
                continue

            bbox_area = float(w * h)
            hull = cv2.convexHull(contour)
            solidity = float(area / max(cv2.contourArea(hull), 1.0))
            if solidity < RED_MIN_SOLIDITY:
                continue

            shifted = contour - np.array([[x, y]])
            shape_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(shape_mask, [shifted], -1, 255, thickness=-1)
            fill_ratio = float(np.count_nonzero(shape_mask) / max(bbox_area, 1.0))
            extent = float(area / max(bbox_area, 1.0))
            if fill_ratio < RED_MIN_FILL_RATIO or extent < RED_MIN_EXTENT:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
            if len(approx) < 4 or len(approx) > 12:
                continue

            center_err = abs((x + w / 2.0) - self.image_width / 2.0)
            bottom_bonus = 120.0 if y + h > self.image_height * 0.35 else 0.0
            score = (
                1.8 * area
                + 420.0 * fill_ratio
                + 280.0 * extent
                + 220.0 * solidity
                + bottom_bonus
                - 0.12 * center_err
            )

            obs = RedObservation(
                cx=float(x + w / 2.0),
                cy=float(cy),
                area=float(area),
                bbox_x=int(x),
                bbox_y=int(y),
                bbox_w=int(w),
                bbox_h=int(h),
                fill_ratio=float(fill_ratio),
                extent=float(extent),
                solidity=float(solidity),
                score=float(score),
            )

            if best is None or obs.score > best.score:
                best = obs

        return best


class RedCubeServoMission(Node):
    def __init__(self):
        super().__init__('red_cube_servo_mission_v10')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.control_timer = self.create_timer(CONTROL_DT, self.control_loop)
        self.status_timer = self.create_timer(STATUS_DT, self.status_loop)

        self.detector = RedDetector()
        self.servo = ServoHelper()

        self.has_scan = False
        self.has_odom = False
        self.has_image = False
        self.front_dist = float('inf')
        self.image_width = None
        self.image_height = None
        self.image_logged = False

        self.world_yaw = 0.0
        self.init_yaw = None
        self.local_yaw = 0.0
        self.turn_target_yaw = 0.0

        self.red_visible = False
        self.red_obs: Optional[RedObservation] = None
        self.prev_red_obs: Optional[RedObservation] = None
        self.red_seen_frames = 0
        self.centered_frames = 0
        self.last_red_seen_time = 0.0
        self.last_red_dir = 1.0
        self.last_red_bbox_h = 0

        self.state = 'WAIT_FOR_DATA'
        self.state_enter_time = time.monotonic()
        self.wait_status_last = None
        self.shutdown_requested = False
        self.shutdown_reason = ''
        self.shutdown_count = 0

        self.console_thread = threading.Thread(target=self.console_loop, daemon=True)
        self.console_thread.start()

        print('[BOOT] Red cube servo mission v10')
        print('[MODE] red cube only, no blue, no marker delivery')
        print(f'[SERVO PARAM] UP={SERVO_UP_DUTY} DOWN={SERVO_DOWN_DUTY} BCM={SERVO_GPIO_BCM}')
        print('[CMD] type H then Enter to stop and exit')

    def ready(self):
        return self.has_scan and self.has_odom and self.has_image

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
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_robot_once(self):
        self.publish_cmd(0.0, 0.0)

    def stop_robot_reliable(self, repeat=12, delay=0.025):
        for _ in range(repeat):
            self.stop_robot_once()
            time.sleep(delay)

    def set_state(self, new_state, text=''):
        if self.state == new_state:
            return
        self.state = new_state
        self.state_enter_time = time.monotonic()
        self.centered_frames = 0
        if text:
            print(f'[STATE] {new_state} | {text}')
        else:
            print(f'[STATE] {new_state}')

    def start_turn_relative(self, radians_delta, state_name):
        self.turn_target_yaw = self.normalize_angle(self.local_yaw + radians_delta)
        self.set_state(state_name, f'target_yaw={self.turn_target_yaw:.2f}')

    def request_shutdown(self, reason):
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.shutdown_reason = reason
        self.shutdown_count = 0
        print(f'[STOP] {reason}')

    def console_loop(self):
        while True:
            try:
                cmd = input().strip().lower()
            except Exception:
                return
            if cmd == 'h':
                self.request_shutdown('manual emergency stop')
                return

    def scan_callback(self, msg):
        ranges = list(msg.ranges)
        n = len(ranges)
        if n == 0:
            return
        vals = list(ranges[0:min(10, n)]) + list(ranges[max(0, n - 10):n])
        good = [x for x in vals if math.isfinite(x) and x > 0.04]
        raw_front = min(good) if good else float('inf')
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
                raise ValueError('decoded frame is None')
        except Exception as exc:
            print(f'[IMAGE] decode failed: {exc}')
            self.clear_red(keep_recent=True)
            return

        self.image_height, self.image_width = frame.shape[:2]
        if not self.image_logged:
            print(f'[IMAGE] compressed size={self.image_width}x{self.image_height}')
            self.image_logged = True

        if self.state in ('WAIT_FOR_DATA', 'SEARCH_RED', 'ALIGN_RED', 'APPROACH_RED'):
            self.update_red(self.detector.detect(frame))
        else:
            self.clear_red(keep_recent=True)

    def update_red(self, obs):
        if obs is None:
            self.clear_red(keep_recent=True)
            return

        stable = False
        if self.prev_red_obs is not None:
            raw_jump = abs(obs.cx - self.prev_red_obs.cx)
            area_ratio = obs.area / max(self.prev_red_obs.area, 1.0)
            stable = raw_jump <= RED_MAX_RAW_JUMP_PX and 0.35 <= area_ratio <= 2.9

        self.red_seen_frames = self.red_seen_frames + 1 if stable else 1
        self.prev_red_obs = obs
        self.red_visible = True
        self.red_obs = obs
        self.last_red_seen_time = time.monotonic()
        self.last_red_bbox_h = obs.bbox_h

        err = obs.cx - self.image_width / 2.0
        if abs(err) > 2.0:
            self.last_red_dir = -1.0 if err > 0.0 else 1.0

    def clear_red(self, keep_recent=False):
        self.red_visible = False
        self.red_obs = None
        self.red_seen_frames = 0
        self.centered_frames = 0
        self.prev_red_obs = None
        if not keep_recent:
            self.last_red_seen_time = 0.0

    def red_error_pixels(self):
        if not self.red_visible or self.red_obs is None or self.image_width is None:
            return None
        return self.red_obs.cx - self.image_width / 2.0

    def red_recently_seen(self):
        return (time.monotonic() - self.last_red_seen_time) <= LOST_TARGET_TIMEOUT

    def red_lockworthy(self):
        if not self.red_visible or self.red_obs is None:
            return False
        if self.red_seen_frames < CONFIRM_FRAMES:
            return False
        if self.red_obs.area < RED_LOCK_MIN_AREA:
            return False
        err = self.red_error_pixels()
        if err is None:
            return False
        return abs(err) <= SEARCH_LOCK_PIXEL_TOL

    def handle_search_red(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if self.red_lockworthy():
            self.stop_robot_once()
            self.set_state('ALIGN_RED', 'red locked')
            return

        if self.red_visible and self.red_obs is not None and self.image_width is not None:
            err = self.red_error_pixels()
            if err is not None and abs(err) <= SEARCH_TRACK_PIXEL_TOL:
                err_norm = err / max(self.image_width / 2.0, 1.0)
                angular = self.clamp_abs(-0.20 * err_norm, ALIGN_MIN_ANG, SEARCH_TRACK_MAX_ANG)
                self.publish_cmd(0.0, angular)
                return

        self.publish_cmd(0.0, SEARCH_ANG)

    def handle_align_red(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if not self.red_visible or self.red_obs is None:
            if self.red_recently_seen():
                self.publish_cmd(0.0, self.last_red_dir * RECOVER_ANG)
                return
            self.set_state('SEARCH_RED', 'red lost before approach')
            return

        err = self.red_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) <= ALIGN_PIXEL_TOL:
            self.centered_frames += 1
            self.stop_robot_once()
            if self.centered_frames >= CENTER_HOLD_FRAMES:
                self.set_state('APPROACH_RED', 'red centered')
            return

        self.centered_frames = 0
        err_norm = err / max(self.image_width / 2.0, 1.0)
        angular = self.clamp_abs(-0.24 * err_norm, ALIGN_MIN_ANG, ALIGN_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_approach_red(self):
        if self.front_dist <= EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if not self.red_visible or self.red_obs is None:
            self.set_state('EXTRA_FORWARD_AFTER_LOST', 'red disappeared, forward 1.5s')
            return

        err = self.red_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) >= REACQUIRE_PIXEL_TOL:
            self.stop_robot_once()
            self.set_state('ALIGN_RED', 'red off center')
            return

        err_norm = err / max(self.image_width / 2.0, 1.0)
        linear = APPROACH_SLOW_SPEED if self.red_obs.bbox_h >= CLOSE_RED_BBOX_H_PX or self.front_dist < 0.20 else APPROACH_FAST_SPEED
        angular = 0.0 if abs(err) <= ALIGN_PIXEL_TOL else self.clamp_abs(-0.16 * err_norm, 0.02, 0.08)
        self.publish_cmd(linear, angular)

    def handle_extra_forward_after_lost(self):
        if time.monotonic() - self.state_enter_time < EXTRA_FORWARD_AFTER_LOST_SEC:
            self.publish_cmd(EXTRA_FORWARD_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.set_state('SERVO_DOWN', 'lower arm')

    def handle_servo_down(self):
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_down()
        self.start_turn_relative(math.pi, 'TURN_180_AFTER_GRAB')

    def handle_turn_180_after_grab(self):
        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= math.radians(TURN_180_TOL_DEG):
            self.stop_robot_once()
            self.set_state('FORWARD_AFTER_TURN', 'drive forward 3s')
            return
        angular = self.clamp_abs(0.65 * error, 0.04, TURN_180_SLOW_MAX_ANG)
        self.publish_cmd(0.0, angular)

    def handle_forward_after_turn(self):
        if time.monotonic() - self.state_enter_time < DELIVERY_FORWARD_SEC:
            self.publish_cmd(DELIVERY_FORWARD_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.set_state('SERVO_UP_RELEASE', 'raise arm')

    def handle_servo_up_release(self):
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_up()
        self.set_state('BACKUP_AFTER_RELEASE', 'backup 1s')

    def handle_backup_after_release(self):
        if time.monotonic() - self.state_enter_time < BACKUP_AFTER_RELEASE_SEC:
            self.publish_cmd(BACKUP_SPEED, 0.0)
            return
        self.stop_robot_once()
        self.start_turn_relative(math.pi, 'TURN_180_RETURN')

    def handle_turn_180_return(self):
        error = self.normalize_angle(self.turn_target_yaw - self.local_yaw)
        if abs(error) <= math.radians(TURN_180_TOL_DEG):
            self.stop_robot_once()
            self.clear_red(keep_recent=False)
            self.set_state('SEARCH_RED', 'continue searching')
            return
        angular = self.clamp_abs(0.75 * error, 0.05, TURN_180_NORMAL_MAX_ANG)
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

        if self.state == 'WAIT_FOR_DATA':
            self.stop_robot_once()
            self.servo.servo_up()
            self.set_state('SEARCH_RED', 'data ready, arm up')
            return

        handlers = {
            'SEARCH_RED': self.handle_search_red,
            'ALIGN_RED': self.handle_align_red,
            'APPROACH_RED': self.handle_approach_red,
            'EXTRA_FORWARD_AFTER_LOST': self.handle_extra_forward_after_lost,
            'SERVO_DOWN': self.handle_servo_down,
            'TURN_180_AFTER_GRAB': self.handle_turn_180_after_grab,
            'FORWARD_AFTER_TURN': self.handle_forward_after_turn,
            'SERVO_UP_RELEASE': self.handle_servo_up_release,
            'BACKUP_AFTER_RELEASE': self.handle_backup_after_release,
            'TURN_180_RETURN': self.handle_turn_180_return,
            'STOPPED': self.stop_robot_once,
        }
        handlers.get(self.state, self.stop_robot_once)()

    def status_loop(self):
        if not self.ready():
            wait_now = (self.has_scan, self.has_odom, self.has_image)
            if wait_now != self.wait_status_last:
                print(f'[WAIT] scan={self.has_scan} odom={self.has_odom} image={self.has_image}')
                self.wait_status_last = wait_now
            return

        red_text = 'red=none'
        if self.red_visible and self.red_obs is not None:
            err = self.red_error_pixels()
            red_text = (
                f'red=yes err={err:.0f} area={self.red_obs.area:.0f} '
                f'bbox=({self.red_obs.bbox_w}x{self.red_obs.bbox_h}) '
                f'seen={self.red_seen_frames}'
            )

        print(
            f'[STATUS] {self.state} {red_text} '
            f'front={self.front_dist:.3f}m yaw={math.degrees(self.local_yaw):.0f} '
            f'image={self.image_width}x{self.image_height}'
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
    node = RedCubeServoMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.request_shutdown('KeyboardInterrupt')
        node.stop_robot_reliable()
    finally:
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
